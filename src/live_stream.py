import math
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import tensorflow as tf
import tpqoa
import time

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
LSTM_MODEL_PATH = BASE_DIR / "models" / "lstm_model.h5"
LSTM_STATS_PATH = BASE_DIR / "models" / "lstm_feature_stats.pkl"
TRADING_TIMEZONE = ZoneInfo("Europe/London")

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
# News integration removed — using LSTM-only signals for now.

# ── Signal combination weights (LSTM-only) ─────────────────────────────────
LSTM_WEIGHT = 1.0

# ── Minimum combined confidence to enter a trade (dead zone) ──────────────
# Combined signal must exceed this threshold in either direction.
# Below it the model is considered undecided and no trade is placed.
MIN_CONFIDENCE = 0.10   # 10 % of the [-1, +1] scale


# get_news_signal removed — news-based signals are disabled.


def load_lstm_artifacts(model_path, stats_path):
    with open(stats_path, "rb") as file_handle:
        feature_stats = pickle.load(file_handle)

    model = tf.keras.models.load_model(model_path, compile=False)
    mean = feature_stats["mean"]
    std = feature_stats["std"].replace(0, 1)
    feature_cols = feature_stats["feature_cols"]
    lookback = int(feature_stats["lookback"])
    window = int(feature_stats["window"])
    return model, mean, std, feature_cols, lookback, window


def generate_features(data, window=20):
    data = data.copy()
    data["returns"] = np.log(data["price"] / data["price"].shift(1))
    data["direction"] = np.where(data["returns"] > 0, 1, 0)
    data["sma"] = data["price"].rolling(window).mean() - data["price"].rolling(150).mean()
    data["boll"] = (data["price"] - data["price"].rolling(window).mean()) / data["price"].rolling(window).std()
    data["min"] = data["price"].rolling(window).min() / data["price"] - 1
    data["max"] = data["price"].rolling(window).max() / data["price"] - 1
    data["mom"] = data["returns"].rolling(3).mean()
    data["vol"] = data["returns"].rolling(window).std()
    data.dropna(inplace=True)
    return data


def infer_granularity(bar_length):
    mapping = {
        pd.Timedelta(minutes=1):  "M1",
        pd.Timedelta(minutes=5):  "M5",
        pd.Timedelta(minutes=15): "M15",
        pd.Timedelta(hours=1):    "H1",
        pd.Timedelta(hours=4):    "H4",
    }
    if bar_length in mapping:
        return mapping[bar_length]
    raise ValueError(f"Unsupported bar length for OANDA history bootstrap: {bar_length}.")




class MLTrader(tpqoa.tpqoa):
    
    def __init__(self, config_file, instrument, bar_length, units, model, mean, std, feature_cols, lookback, window,
                 stop_loss=0.005, take_profit=0.010):
        super().__init__(config_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.granularity = infer_granularity(self.bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = pd.DataFrame()
        self.data = None
        self.last_bar = pd.Timestamp.now(tz=TRADING_TIMEZONE).floor(self.bar_length)
        self.units = units
        self.position = 0
        self.profits = []
        self.stop_loss   = stop_loss    # fraction, e.g. 0.005 = 0.5 %
        self.take_profit = take_profit  # fraction, e.g. 0.010 = 1.0 %
        self.entry_price: float | None = None
        
        self.model = model
        self.mean = mean
        self.std = std
        self.feature_cols = feature_cols
        self.lookback = lookback
        self.window = window

        self.bootstrap_history()

    def bootstrap_history(self):
        # Request completed bars via count-based OANDA v20 call.
        # This avoids the 'to is in the future' 400 error from time-range requests.
        warmup_bars = max(350, 160 + self.lookback + self.window)
        try:
            resp = self.ctx.instrument.candles(
                self.instrument,
                price="M",
                granularity=self.granularity,
                count=warmup_bars,
            )
            if resp.status != 200:
                print(f"History bootstrap HTTP {resp.status}: {resp.body}")
                return
            raw_candles = resp.body.get("candles", [])
        except Exception as exc:
            print(f"History bootstrap failed: {exc}")
            return

        rows = []
        for c in raw_candles:
            if not c.complete:
                continue
            rows.append({
                "time":          pd.to_datetime(c.time, utc=True).tz_convert(TRADING_TIMEZONE),
                self.instrument: float(c.mid.c),
            })

        if not rows:
            print("History bootstrap: no completed bars returned.")
            return

        history = pd.DataFrame(rows).set_index("time")
        self.raw_data = history.copy()
        bar_seconds = int(self.bar_length.total_seconds())
        last_ts = history.index[-1]
        # Align last_bar to the most recent completed bar boundary
        self.last_bar = pd.Timestamp(last_ts)
    
    def on_success(self, time, bid, ask):
        print(self.ticks, end = " ",flush=True)
        
        # collect and store tick data
        recent_tick = pd.to_datetime(time, utc=True).tz_convert(TRADING_TIMEZONE)
        df = pd.DataFrame({self.instrument:(ask + bid)/2}, 
                          index = [recent_tick])
        self.tick_data = pd.concat([self.tick_data, df]) # new with pd.concat()
        
        # if a time longer than the bar_length has elapsed between last full bar and the most recent tick
        if recent_tick - self.last_bar >= self.bar_length:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()
            
    def resample_and_join(self):
        resampled = self.tick_data.resample(self.bar_length, label="right").last().ffill()
        complete_bars = resampled.iloc[:-1]
        if not complete_bars.empty:
            self.raw_data = pd.concat([self.raw_data, complete_bars])
        self.tick_data = self.tick_data.iloc[-1:]
        # Advance last_bar to the left edge of the current open bar so the
        # on_success condition won't fire again until the next 15-min boundary.
        self.last_bar = resampled.index[-1] - self.bar_length
       
    def define_strategy(self):
        df = self.raw_data.rename(columns={self.instrument: "price"}).copy()
        df = generate_features(df, window=self.window)

        if len(df) < self.lookback:
            return

        df[self.feature_cols] = (df[self.feature_cols] - self.mean) / self.std
        sequence = df[self.feature_cols].tail(self.lookback).to_numpy(dtype=np.float32)
        if len(sequence) < self.lookback:
            return

        # ── LSTM signal ───────────────────────────────────────────────────────
        prediction  = float(self.model.predict(sequence[np.newaxis, ...], verbose=0).reshape(-1)[0])
        lstm_signal = (prediction - 0.5) * 2   # scale [0,1] → [-1,+1]

        # ── Weighted combination (LSTM-only) ─────────────────────────────────
        combined = LSTM_WEIGHT * lstm_signal

        # Dead zone: if combined conviction is too low, stay flat
        if abs(combined) < MIN_CONFIDENCE:
            position = 0
            direction = "FLAT (low confidence)"
        elif combined > 0:
            position = 1
            direction = "LONG"
        else:
            position = -1
            direction = "SHORT"

        # Store for use in report_trade
        self._last_signal = {
            "lstm_prob":    prediction,
            "lstm_signal":  lstm_signal,
            "combined":     combined,
            "direction":    direction,
        }

        print(
            f"\nLSTM prob={prediction:.3f}  signal={lstm_signal:+.3f} | "
            f"Combined={combined:+.3f}  ->  {direction}"
        )

        df["pred"]     = np.nan
        df["position"] = np.nan
        df.loc[df.index[-1], "pred"]     = prediction
        df.loc[df.index[-1], "position"] = position

        self.data = df.copy()
        
    def execute_trades(self):
        if self.data is None or self.data.empty:
            return

        # ── Stop Loss / Take Profit check ─────────────────────────────────────
        current_price = float(self.data["price"].iloc[-1])
        if self.position != 0 and self.entry_price is not None:
            ret = (current_price - self.entry_price) / self.entry_price * self.position
            if ret <= -self.stop_loss:
                close_units = -self.position * self.units
                order = self.create_order(self.instrument, close_units, suppress=True, ret=True)
                self.report_trade(order, f"STOP LOSS HIT (ret={ret*100:.2f}%)")
                self.position   = 0
                self.entry_price = None
                return
            elif ret >= self.take_profit:
                close_units = -self.position * self.units
                order = self.create_order(self.instrument, close_units, suppress=True, ret=True)
                self.report_trade(order, f"TAKE PROFIT HIT (ret={ret*100:.2f}%)")
                self.position   = 0
                self.entry_price = None
                return

        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING LONG")
                self.entry_price = float(order["price"])
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress=True, ret=True)
                self.report_trade(order, "GOING LONG")
                self.entry_price = float(order["price"])
            self.position = 1
        elif self.data["position"].iloc[-1] == -1:
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING SHORT")
                self.entry_price = float(order["price"])
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress=True, ret=True)
                self.report_trade(order, "GOING SHORT")
                self.entry_price = float(order["price"])
            self.position = -1
        elif self.data["position"].iloc[-1] == 0:
            if self.position == -1:
                order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING NEUTRAL")
                self.entry_price = None
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING NEUTRAL")
                self.entry_price = None
            self.position = 0
            
    def report_trade(self, order, going):  # NEW
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100 * "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))

        # ── Trade decision constraints ────────────────────────────────────────
        sig = getattr(self, "_last_signal", None)
        if sig:
            lstm_conf   = abs(sig["lstm_signal"])   * 100   # 0-100 %
            comb_conf   = abs(sig["combined"])      * 100
            print("  Trade taken at the following signal constraints:")
            print(f"    Price model (LSTM) : prob={sig['lstm_prob']:.4f}  |  "
                f"signal={sig['lstm_signal']:+.4f}  |  confidence={lstm_conf:.1f}%  "
                f"(weight {LSTM_WEIGHT*100:.0f}%)")
            print(f"    Combined signal    : {sig['combined']:+.4f}  |  "
                f"final confidence={comb_conf:.1f}%  |  direction={sig['direction']}")
        print(100 * "-" + "\n")

if __name__ == "__main__":
    model, mean, std, feature_cols, lookback, window = load_lstm_artifacts(LSTM_MODEL_PATH, LSTM_STATS_PATH)

    trader = MLTrader(
        config_file=str(BASE_DIR / "src" / "oanda.cfg"),
        instrument="EUR_USD",
        bar_length="1min",  # ← TESTING: change back to "15min" for production
        units=100000,
        model=model, mean=mean, std=std,
        feature_cols=feature_cols, lookback=lookback, window=window,
        stop_loss=0.005,    # 0.5 % stop loss
        take_profit=0.010,  # 1.0 % take profit
    )

    while True:
        try:
            trader.stream_data(trader.instrument)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Stream disconnected: {e}. Reconnecting in 5s...")
            trader.bootstrap_history()   # refresh history after gap
            time.sleep(5)

    if trader.position != 0:
        close_order = trader.create_order(trader.instrument, units=-trader.position * trader.units,
                                          suppress=True, ret=True)
        trader.report_trade(close_order, "GOING NEUTRAL")
        trader.position = 0