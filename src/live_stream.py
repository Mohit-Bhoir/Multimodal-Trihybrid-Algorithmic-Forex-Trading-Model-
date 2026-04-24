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
from news import fetch_news, analyze_sentiment

# ── News signal configuration ──────────────────────────────────────────────
NEWS_QUERIES = [
    "EURUSD news now",
    "EUR USD breaking news",
    "EURUSD live updates",
    "EURUSD price move news",
    "EURUSD reaction news",
    "EURUSD market moving news",
    "EURUSD volatility news now",
    "EURUSD spike news",
    "EURUSD drop news",
    "euro dollar breaking news",
    "USD breaking news forex",
    "euro breaking news forex",
    "forex market breaking news EUR USD",
    "EURUSD central bank comments live",
    "ECB comments live euro impact",
    "Fed comments live USD impact",
    "US economic data release EURUSD reaction",
    "Eurozone data release EURUSD reaction",
    "EURUSD headlines now",
    "forex live headlines EUR USD"
]
NEWS_FETCH_MINUTES = 120   # look-back window for news articles
NEWS_DECAY_LAMBDA  = 0.02  # exponential decay rate (half-weight at ~35 min)

# ── Signal combination weights ─────────────────────────────────────────────
LSTM_WEIGHT = 0.6
NEWS_WEIGHT  = 0.4

# ── Minimum combined confidence to enter a trade (dead zone) ──────────────
# Combined signal must exceed this threshold in either direction.
# Below it the model is considered undecided and no trade is placed.
MIN_CONFIDENCE = 0.10   # 10 % of the [-1, +1] scale


def get_news_signal(max_age_minutes=NEWS_FETCH_MINUTES,
                    decay_lambda=NEWS_DECAY_LAMBDA,
                    num_articles=5):
    """Return a time-decayed sentiment score in [-1, 1] from recent EUR/USD news.

    Exponential decay weights newer articles more heavily:
        weight = exp(-lambda * age_in_minutes)
    A score > 0 is bullish; < 0 is bearish.  Returns (score, n_unique_articles).
    """
    now = datetime.now(timezone.utc)
    weighted_sum = 0.0
    total_weight  = 0.0
    seen = set()

    for query in NEWS_QUERIES:
        try:
            articles = fetch_news(query,
                                  num_articles=num_articles,
                                  max_age_minutes=max_age_minutes,
                                  fetch_content=False)  # titles only – fast
        except Exception:
            continue

        for article in articles:
            title = article["title"]
            if title in seen:
                continue
            seen.add(title)

            published_dt = article.get("published_dt")
            if published_dt is None:
                age_minutes = float(max_age_minutes)
            else:
                age_minutes = max(0.0, (now - published_dt).total_seconds() / 60.0)

            weight = math.exp(-decay_lambda * age_minutes)
            polarity, _ = analyze_sentiment(title)
            weighted_sum += weight * polarity
            total_weight  += weight

    if total_weight == 0.0:
        return 0.0, 0

    score = weighted_sum / total_weight  # normalised to [-1, 1]
    return score, len(seen)


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
    if bar_length == pd.Timedelta(minutes=15):
        return "M15"
    raise ValueError(f"Unsupported bar length for OANDA history bootstrap: {bar_length}.")




class MLTrader(tpqoa.tpqoa):
    
    def __init__(self, config_file, instrument, bar_length, units, model, mean, std, feature_cols, lookback, window):
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
        
        self.model = model
        self.mean = mean
        self.std = std
        self.feature_cols = feature_cols
        self.lookback = lookback
        self.window = window

        self.bootstrap_history()

    def bootstrap_history(self):
        warmup_bars = max(250, 150 + self.lookback + self.window)
        bar_seconds = int(self.bar_length.total_seconds())

        now_local = datetime.now(TRADING_TIMEZONE).replace(second=0, microsecond=0)
        now_epoch = int(now_local.timestamp())
        aligned_local_epoch = now_epoch - (now_epoch % bar_seconds)
        aligned_local = datetime.fromtimestamp(aligned_local_epoch, tz=TRADING_TIMEZONE)

        aligned_end_utc = aligned_local.astimezone(timezone.utc)
        start_utc = aligned_end_utc - timedelta(seconds=warmup_bars * bar_seconds)

        # Use naive UTC datetimes for tpqoa/OANDA request formatting compatibility.
        aligned_end = aligned_end_utc.replace(tzinfo=None)
        start = start_utc.replace(tzinfo=None)

        try:
            history = self.get_history(
                instrument=self.instrument,
                start=start,
                end=aligned_end,
                granularity=self.granularity,
                price="M",
            )
        except Exception as exc:
            print(f"History bootstrap failed: {exc}")
            return

        if history is None or history.empty:
            return

        history = history.rename(columns={"c": self.instrument})[[self.instrument]].dropna()
        history.index = pd.to_datetime(history.index, utc=True).tz_convert(TRADING_TIMEZONE)
        self.raw_data = history.copy()
        self.last_bar = pd.Timestamp(aligned_local)
    
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

        # ── News sentiment signal (time-decayed) ──────────────────────────────
        news_signal, n_articles = get_news_signal()

        # ── Weighted combination ──────────────────────────────────────────────
        combined = LSTM_WEIGHT * lstm_signal + NEWS_WEIGHT * news_signal

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
            "news_signal":  news_signal,
            "n_articles":   n_articles,
            "combined":     combined,
            "direction":    direction,
        }

        print(
            f"\nLSTM prob={prediction:.3f}  signal={lstm_signal:+.3f} | "
            f"News score={news_signal:+.3f} ({n_articles} articles) | "
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
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING LONG")  # NEW
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress = True, ret = True) 
                self.report_trade(order, "GOING LONG")  # NEW
            self.position = 1
        elif self.data["position"].iloc[-1] == -1: 
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")  # NEW
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")  # NEW
            self.position = -1
        elif self.data["position"].iloc[-1] == 0: 
            if self.position == -1:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True) 
                self.report_trade(order, "GOING NEUTRAL")  # NEW
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING NEUTRAL")  # NEW
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
            news_conf   = abs(sig["news_signal"])   * 100
            comb_conf   = abs(sig["combined"])      * 100
            print("  Trade taken at the following signal constraints:")
            print(f"    Price model (LSTM) : prob={sig['lstm_prob']:.4f}  |  "
                  f"signal={sig['lstm_signal']:+.4f}  |  confidence={lstm_conf:.1f}%  "
                  f"(weight {LSTM_WEIGHT*100:.0f}%)")
            print(f"    News sentiment     : score={sig['news_signal']:+.4f}  |  "
                  f"confidence={news_conf:.1f}%  |  "
                  f"articles={sig['n_articles']}  (weight {NEWS_WEIGHT*100:.0f}%)")
            print(f"    Combined signal    : {sig['combined']:+.4f}  |  "
                  f"final confidence={comb_conf:.1f}%  |  direction={sig['direction']}")
        print(100 * "-" + "\n")

if __name__ == "__main__":
    model, mean, std, feature_cols, lookback, window = load_lstm_artifacts(LSTM_MODEL_PATH, LSTM_STATS_PATH)

    trader = MLTrader(
        config_file=str(BASE_DIR / "src" / "oanda.cfg"),
        instrument="EUR_USD",
        bar_length="15min",
        units=100000,
        model=model, mean=mean, std=std,
        feature_cols=feature_cols, lookback=lookback, window=window
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