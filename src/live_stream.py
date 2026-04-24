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

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
LSTM_MODEL_PATH = BASE_DIR / "models" / "lstm_model.h5"
LSTM_STATS_PATH = BASE_DIR / "models" / "lstm_feature_stats.pkl"
TRADING_TIMEZONE = ZoneInfo("Europe/London")


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
        self.last_bar = self.raw_data.index[-1]
    
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
        self.raw_data = pd.concat([self.raw_data, self.tick_data.resample(self.bar_length, 
                                                                          label="right").last().ffill().iloc[:-1]]) 
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]
       
    def define_strategy(self):
        df = self.raw_data.rename(columns={self.instrument: "price"}).copy()
        df = generate_features(df, window=self.window)

        if len(df) < self.lookback:
            return

        df[self.feature_cols] = (df[self.feature_cols] - self.mean) / self.std
        sequence = df[self.feature_cols].tail(self.lookback).to_numpy(dtype=np.float32)
        if len(sequence) < self.lookback:
            return

        prediction = float(self.model.predict(sequence[np.newaxis, ...], verbose=0).reshape(-1)[0])
        df["pred"] = np.nan
        df["position"] = np.nan
        df.loc[df.index[-1], "pred"] = prediction
        df.loc[df.index[-1], "position"] = 1 if prediction >= 0.5 else -1

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
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
        print(100 * "-" + "\n") 

if __name__ == "__main__":
    model, mean, std, feature_cols, lookback, window = load_lstm_artifacts(LSTM_MODEL_PATH, LSTM_STATS_PATH)

    trader = MLTrader(config_file = str(BASE_DIR / "src" / "oanda.cfg"), 
                      instrument = "EUR_USD", 
                      bar_length = "15min", 
                      units = 100000, 
                      model = model,
                      mean = mean,
                      std = std,
                      feature_cols = feature_cols,
                      lookback = lookback,
                      window = window)
    trader.stream_data(trader.instrument)
    if trader.position != 0: 
        close_order = trader.create_order(trader.instrument, units = -trader.position * trader.units, 
                                          suppress = True, ret = True) 
        trader.report_trade(close_order, "GOING NEUTRAL")
        trader.position = 0