"""
trading_engine.py
-----------------
Shared utilities for the Streamlit frontend.
Provides model loading, feature engineering, OANDA price fetching,
and a batched BacktestEngine.
All timestamps are in British Summer Time (Europe/London).
"""

import collections
import os
import pickle
import threading
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # suppress TF INFO / WARNING logs

# ── Constants ──────────────────────────────────────────────────────────────────
BST = ZoneInfo("Europe/London")
ROOT = Path(__file__).resolve().parent.parent.parent   # project root

LSTM_MODEL_PATH = ROOT / "models" / "lstm_model.h5"
LSTM_STATS_PATH = ROOT / "models" / "lstm_feature_stats.pkl"
OANDA_CONFIG_PATH = ROOT / "src" / "oanda.cfg"

# Backtest data — single append-only CSV maintained by the Airflow DAG.
# Kept separate from the model-training pipeline (data/raw/, data/processed/).
BACKTEST_CSV = ROOT / "data" / "backtest" / "forex_data_backtest.csv"

FEATURE_COLS = ["returns", "sma", "boll", "min", "max", "mom", "vol"]


def get_oanda_config_path() -> str:
    """Return path to a valid oanda.cfg.

    - Local / Docker: uses the real file at src/oanda.cfg.
    - Streamlit Community Cloud: writes a temp file from st.secrets so the
      real cfg file never needs to be committed to the repo.
    """
    if OANDA_CONFIG_PATH.exists():
        return str(OANDA_CONFIG_PATH)
    # Cloud deployment — build config from st.secrets
    import configparser
    import tempfile
    import streamlit as st
    cfg = configparser.ConfigParser()
    cfg["oanda"] = {
        "account_id":   st.secrets["oanda"]["account_id"],
        "access_token": st.secrets["oanda"]["access_token"],
        "account_type": st.secrets["oanda"]["account_type"],
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".cfg", delete=False, prefix="oanda_"
    )
    cfg.write(tmp)
    tmp.close()
    return tmp.name


# ── Live tick streaming ────────────────────────────────────────────────────────
# Thread-safe deque accumulates ticks from the OANDA streaming thread.
# Streamlit drains it on each page refresh.

_tick_buffer: collections.deque = collections.deque(maxlen=2000)
_stream_thread: threading.Thread | None = None
_stream_lock = threading.Lock()
_stop_event   = threading.Event()   # set() to stop, clear() to allow running


def start_tick_stream(api, instrument: str) -> None:
    """Launch a daemon thread that streams live ticks into _tick_buffer.

    Auto-reconnects on any connection drop or error.
    Safe to call on every Streamlit rerun — no-op when thread is alive.
    """
    global _stream_thread

    _stop_event.clear()   # allow running (clears any previous stop)

    with _stream_lock:
        if _stream_thread is not None and _stream_thread.is_alive():
            return   # already streaming

        def _callback(instr, time, bid, ask):  # noqa: ARG001
            if _stop_event.is_set():
                api.stop_stream = True
                return
            bid_f = round(float(bid), 5)
            ask_f = round(float(ask), 5)
            mid   = round((bid_f + ask_f) / 2, 5)
            ts    = pd.to_datetime(time, utc=True).astimezone(BST)
            _tick_buffer.append({"time": ts, "bid": bid_f, "ask": ask_f, "price": mid})

        def _run():
            import time as _time
            while not _stop_event.is_set():
                try:
                    api.stop_stream = False
                    api.stream_data(instrument, callback=_callback)
                except Exception:
                    pass
                if not _stop_event.is_set():
                    _time.sleep(3)   # brief pause then reconnect

        _stream_thread = threading.Thread(target=_run, daemon=True, name="oanda-tick-stream")
        _stream_thread.start()


def stop_tick_stream(api) -> None:
    """Permanently stop the OANDA streaming thread."""
    _stop_event.set()
    api.stop_stream = True


def is_stream_running() -> bool:
    """Return True if the background tick stream thread is alive."""
    return _stream_thread is not None and _stream_thread.is_alive()


def drain_tick_buffer() -> list:
    """Remove and return all buffered ticks as a list of dicts {time, price}."""
    ticks: list = []
    while True:
        try:
            ticks.append(_tick_buffer.popleft())
        except IndexError:
            break
    return ticks



# ── Model helpers ──────────────────────────────────────────────────────────────

def load_lstm_artifacts():
    """Load LSTM model + normalisation stats.

    Returns
    -------
    model, mean (pd.Series), std (pd.Series), feature_cols (list), lookback (int), window (int)
    """
    import tensorflow as tf

    with open(LSTM_STATS_PATH, "rb") as fh:
        stats = pickle.load(fh)

    model = tf.keras.models.load_model(str(LSTM_MODEL_PATH), compile=False)
    mean  = stats["mean"]
    std   = stats["std"].replace(0, 1)
    return model, mean, std, list(stats["feature_cols"]), int(stats["lookback"]), int(stats["window"])


def generate_features(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add technical indicator columns to a DataFrame that has a 'price' column."""
    df = data.copy()
    df["returns"]   = np.log(df["price"] / df["price"].shift(1))
    df["direction"] = np.where(df["returns"] > 0, 1, 0)
    df["sma"]       = df["price"].rolling(window).mean() - df["price"].rolling(150).mean()
    df["boll"]      = (df["price"] - df["price"].rolling(window).mean()) / df["price"].rolling(window).std()
    df["min"]       = df["price"].rolling(window).min() / df["price"] - 1
    df["max"]       = df["price"].rolling(window).max() / df["price"] - 1
    df["mom"]       = df["returns"].rolling(3).mean()
    df["vol"]       = df["returns"].rolling(window).std()
    df.dropna(inplace=True)
    return df


# ── OANDA helpers ──────────────────────────────────────────────────────────────

def fetch_recent_bars(api, instrument: str, lookback: int, window: int,
                      granularity: str = "M15") -> pd.DataFrame:
    """Fetch enough completed historical bars for feature generation + LSTM lookback.

    Uses a count-based OANDA v20 candles request (avoids the 'to is in the
    future' 400 error that occurs with time-range requests close to now).
    Only returns bars where complete=True (excludes the current forming candle).

    Returns a DataFrame with a BST-aware DatetimeIndex and a 'price' column.
    """
    # Need 150 (rolling SMA warmup) + window + lookback completed bars minimum.
    # Request a comfortable extra buffer.
    n_bars = max(350, 160 + lookback + window)

    try:
        resp = api.ctx.instrument.candles(
            instrument,
            price="M",
            granularity=granularity,
            count=n_bars,
        )
        if resp.status != 200:
            return pd.DataFrame()
        raw_candles = resp.body.get("candles", [])
    except Exception:
        return pd.DataFrame()

    if not raw_candles:
        return pd.DataFrame()

    rows = []
    for c in raw_candles:
        if not c.complete:           # skip the still-forming current bar
            continue
        rows.append({
            "time":  pd.to_datetime(c.time, utc=True).tz_convert(BST),
            "price": float(c.mid.c),  # use close price
        })

    if not rows:
        return pd.DataFrame()

    hist = pd.DataFrame(rows).set_index("time")
    return hist


def get_bid_ask_price(api, instrument: str) -> dict | None:
    """Return current bid, ask and mid via OANDA REST pricing endpoint.

    Returns ``{"bid": float, "ask": float, "price": float}`` or ``None``.
    Uses the same endpoint as :func:`get_current_price` but exposes all three
    values so the tick feed can display a realistic spread without needing the
    streaming connection to be active.
    """
    try:
        r      = api.ctx.pricing.get(api.account_id, instruments=instrument)
        prices = r.get("prices", 200)
        if prices and len(prices) > 0:
            p   = prices[0]
            bid = round(float(p.bids[0].price), 5)
            ask = round(float(p.asks[0].price), 5)
            mid = round((bid + ask) / 2, 5)
            return {"bid": bid, "ask": ask, "price": mid}
    except Exception:
        pass
    return None


def get_current_price(api, instrument: str) -> float | None:
    """Return current mid price via OANDA REST pricing endpoint (works in real time).

    Falls back to the most recent M1 historical bar close if the pricing
    endpoint is unavailable.
    """
    # ── Primary: live streaming-prices endpoint ───────────────────────────────
    try:
        r      = api.ctx.pricing.get(api.account_id, instruments=instrument)
        prices = r.get("prices", 200)
        if prices and len(prices) > 0:
            p   = prices[0]
            bid = float(p.bids[0].price)
            ask = float(p.asks[0].price)
            return round((bid + ask) / 2, 5)
    except Exception:
        pass

    # ── Fallback: latest M1 historical bar ────────────────────────────────────
    end_utc   = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(minutes=10)
    try:
        hist = api.get_history(
            instrument=instrument,
            start=start_utc.replace(tzinfo=None),
            end=end_utc.replace(tzinfo=None),
            granularity="M1",
            price="M",
        )
        if hist is not None and not hist.empty:
            return round(float(hist["c"].iloc[-1]), 5)
    except Exception:
        pass
    return None


def get_account_info(api) -> dict:
    """Fetch real account balance, NAV, P&L and margin from OANDA demo account.

    Returns a dict with keys:
        balance, nav, unrealized_pl, realized_pl, currency, margin_used
    Returns an empty dict on failure.
    """
    try:
        r   = api.ctx.account.summary(api.account_id)
        acc = r.get("account", 200)
        return {
            "balance":       round(float(acc.balance),       2),
            "nav":           round(float(acc.NAV),            2),
            "unrealized_pl": round(float(acc.unrealizedPL),  4),
            "realized_pl":   round(float(acc.pl),            4),
            "currency":      str(acc.currency),
            "margin_used":   round(float(acc.marginUsed),    2),
        }
    except Exception:
        return {}


def get_open_position(api, instrument: str) -> dict:
    """Return the current open position for *instrument* from OANDA.

    Returns a dict:
        side          (+1 = long, -1 = short, 0 = flat)
        units         (absolute number of units)
        avg_price     (average open price, or None)
        unrealized_pl (unrealised P&L in account currency)
    """
    try:
        r   = api.ctx.position.get(api.account_id, instrument=instrument)
        pos = r.get("position", 200)
        long_units  = int(float(pos.long.units))
        short_units = int(float(pos.short.units))
        if long_units > 0:
            return {"side": 1,  "units": long_units,
                    "avg_price": round(float(pos.long.averagePrice), 5),
                    "unrealized_pl": round(float(pos.long.unrealizedPL), 4)}
        if short_units < 0:
            return {"side": -1, "units": abs(short_units),
                    "avg_price": round(float(pos.short.averagePrice), 5),
                    "unrealized_pl": round(float(pos.short.unrealizedPL), 4)}
    except Exception:
        pass
    return {"side": 0, "units": 0, "avg_price": None, "unrealized_pl": 0.0}


def predict_signal(model, mean, std, feature_cols, lookback, window,
                   price_data: pd.DataFrame,
                   long_threshold: float = 0.55,
                   short_threshold: float = 0.45) -> dict:
    """Run the LSTM on the most recent bars and return a signal dictionary."""
    df = generate_features(price_data[["price"]].copy(), window=window)
    if len(df) < lookback:
        return {"prob": None, "lstm_signal": None, "combined": None,
                "direction": "INSUFFICIENT DATA"}

    df_norm = df.copy()
    df_norm[feature_cols] = (df[feature_cols] - mean) / std

    seq  = df_norm[feature_cols].tail(lookback).to_numpy(dtype=np.float32)
    prob = float(model.predict(seq[np.newaxis, ...], verbose=0).reshape(-1)[0])
    sig  = (prob - 0.5) * 2   # scale [0,1] → [-1, +1]

    if   prob >= long_threshold:  direction = "LONG"
    elif prob <= short_threshold: direction = "SHORT"
    else:                         direction = "FLAT"

    return {"prob": prob, "lstm_signal": sig, "combined": sig, "direction": direction}


def get_recent_trades(api, instrument: str, count: int = 30) -> list:
    """Fetch recent open + closed trades for *instrument* from OANDA.

    Returns a list of dicts with keys:
        id, state, side, open_time, close_time, units,
        open_price, close_price, realised_pl, unrealised_pl
    """
    rows = []

    # ── Open trades ───────────────────────────────────────────────────────────
    try:
        r = api.ctx.trade.listOpen(api.account_id)
        if r.status == 200:
            for t in (r.body.get("trades") or []):
                if getattr(t, "instrument", None) != instrument:
                    continue
                units = int(float(t.currentUnits))
                rows.append({
                    "id":            str(t.id),
                    "state":         "OPEN",
                    "side":          "LONG" if units > 0 else "SHORT",
                    "open_time":     str(t.openTime)[:19].replace("T", " "),
                    "close_time":    None,
                    "units":         abs(units),
                    "open_price":    round(float(t.price), 5),
                    "close_price":   None,
                    "realised_pl":   None,
                    "unrealised_pl": round(float(t.unrealizedPL), 4),
                })
    except Exception:
        pass

    # ── Closed trades ─────────────────────────────────────────────────────────
    try:
        r = api.ctx.trade.list(
            api.account_id,
            instrument=instrument,
            state="CLOSED",
            count=count,
        )
        if r.status == 200:
            for t in (r.body.get("trades") or []):
                init_units = int(float(getattr(t, "initialUnits", 0)))
                rows.append({
                    "id":            str(t.id),
                    "state":         "CLOSED",
                    "side":          "LONG" if init_units > 0 else "SHORT",
                    "open_time":     str(t.openTime)[:19].replace("T", " "),
                    "close_time":    str(getattr(t, "closeTime", ""))[:19].replace("T", " ") or None,
                    "units":         abs(init_units),
                    "open_price":    round(float(t.price), 5),
                    "close_price":   round(float(getattr(t, "averageClosePrice", 0) or 0), 5) or None,
                    "realised_pl":   round(float(t.realizedPL), 4),
                    "unrealised_pl": None,
                })
    except Exception:
        pass

    # Sort by trade id descending (highest = most recent)
    rows.sort(key=lambda x: int(x["id"]) if x["id"].isdigit() else 0, reverse=True)
    return rows[:count]


# ── Backtesting ────────────────────────────────────────────────────────────────

class BacktestEngine:
    """Vectorised (batch-prediction) backtest for the LSTM strategy."""

    MIN_WARMUP = 200   # bars needed before the first prediction

    def __init__(
        self,
        model, mean, std, feature_cols, lookback, window,
        long_threshold:  float = 0.55,
        short_threshold: float = 0.45,
        stop_loss:       float = 0.005,
        take_profit:     float = 0.010,
        capital:         float = 100_000.0,
        min_confidence:  float = 0.10,
    ):
        self.model           = model
        self.mean            = mean
        self.std             = std
        self.feature_cols    = feature_cols
        self.lookback        = lookback
        self.window          = window
        self.long_threshold  = long_threshold
        self.short_threshold = short_threshold
        self.stop_loss       = stop_loss
        self.take_profit     = take_profit
        self.capital         = float(capital)
        self.min_confidence  = min_confidence

    # ------------------------------------------------------------------
    def run(self, raw_data: pd.DataFrame) -> dict:
        """Run backtest on *raw_data* (must have a 'price' column).

        Returns
        -------
        dict with keys: equity_curve, trades, metrics   (all DataFrames / dicts)
        """
        if "price" not in raw_data.columns:
            raw_data = raw_data.rename(columns={raw_data.columns[0]: "price"})

        # Preserve per-bar spread from CSV if available
        has_spread = "spread" in raw_data.columns

        df = generate_features(raw_data[["price"]].copy(), window=self.window)

        min_bars = self.lookback + self.MIN_WARMUP
        if len(df) < min_bars:
            return {"error": f"Need ≥ {min_bars} bars after feature engineering. Got {len(df)}."}

        # Normalise
        df_norm = df.copy()
        df_norm[self.feature_cols] = (df[self.feature_cols] - self.mean) / self.std

        n         = len(df)
        start_idx = self.lookback

        # ── Batch-predict all sequences in one model call ─────────────────────
        sequences = np.stack([
            df_norm[self.feature_cols].iloc[i - self.lookback:i].to_numpy(dtype=np.float32)
            for i in range(start_idx, n)
        ])                                      # shape: (N, lookback, features)
        all_probs = self.model.predict(sequences, verbose=0, batch_size=512).reshape(-1)

        # ── Simulate bar-by-bar ───────────────────────────────────────────────
        nav       = self.capital
        position  = 0       # 0 = flat, 1 = long, -1 = short
        entry_px  = None
        equity    = []
        trades    = []

        for idx, i in enumerate(range(start_idx, n)):
            ts    = df.index[i]
            price = float(df["price"].iloc[i])

            # ── Stop-loss / Take-profit ───────────────────────────────────────
            if position != 0 and entry_px is not None:
                ret = (price - entry_px) / entry_px * position
                if ret <= -self.stop_loss:
                    pnl  = nav * (-self.stop_loss)
                    nav += pnl
                    trades.append(_trade(ts, "SL_CLOSE",  entry_px, price, pnl, position, None))
                    position = 0;  entry_px = None
                    equity.append({"time": ts, "nav": nav})
                    continue
                elif ret >= self.take_profit:
                    pnl  = nav * self.take_profit
                    nav += pnl
                    trades.append(_trade(ts, "TP_CLOSE",  entry_px, price, pnl, position, None))
                    position = 0;  entry_px = None
                    equity.append({"time": ts, "nav": nav})
                    continue

            # ── Model signal ──────────────────────────────────────────────────
            prob    = float(all_probs[idx])
            sig     = (prob - 0.5) * 2
            new_pos = _resolve_position(prob, sig, self.long_threshold,
                                        self.short_threshold, self.min_confidence)

            # ── Trade on position change ──────────────────────────────────────
            if new_pos != position:
                if position != 0 and entry_px is not None:
                    ret  = (price - entry_px) / entry_px * position
                    # Use actual per-bar spread from data; fall back to 0.
                    # Look up by label (df index is a subset of raw_data after dropna)
                    bar_ts = df.index[i]
                    bar_spread = (float(raw_data.at[bar_ts, "spread"])
                                  if has_spread and bar_ts in raw_data.index else 0.0)
                    pnl  = nav * ret - abs(nav * bar_spread / price)
                    nav += pnl
                    trades.append(_trade(ts, f"CLOSE_{'LONG' if position==1 else 'SHORT'}",
                                         entry_px, price, pnl, position, round(prob, 4)))
                    position = 0;  entry_px = None

                if new_pos != 0:
                    position = new_pos;  entry_px = price
                    trades.append(_trade(ts, "LONG" if new_pos == 1 else "SHORT",
                                         price, None, 0.0, new_pos, round(prob, 4)))

            equity.append({"time": ts, "nav": nav})

        # Close any open position at end of data
        if position != 0 and entry_px is not None:
            last_px = float(df["price"].iloc[-1])
            ret  = (last_px - entry_px) / entry_px * position
            pnl  = nav * ret
            nav += pnl
            trades.append(_trade(df.index[-1], "END_CLOSE",
                                 entry_px, last_px, pnl, position, None))
            equity.append({"time": df.index[-1], "nav": nav})

        equity_df = (pd.DataFrame(equity).set_index("time")
                     if equity else pd.DataFrame(columns=["nav"]))
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        return {
            "equity_curve": equity_df,
            "trades":       trades_df,
            "metrics":      _compute_metrics(equity_df, trades_df, self.capital),
        }


# ── Private helpers ────────────────────────────────────────────────────────────

def _trade(ts, action, entry, exit_, pnl, side, prob):
    return dict(time=ts, action=action, entry=entry, exit=exit_,
                pnl=round(float(pnl), 4), side=side, prob=prob)


def _resolve_position(prob, sig, long_thr, short_thr, min_conf):
    if   prob >= long_thr:       return 1
    elif prob <= short_thr:      return -1
    else:                        return 0


def _compute_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame,
                     capital: float) -> dict:
    if equity_df.empty or "nav" not in equity_df.columns:
        return {}

    nav          = equity_df["nav"]
    total_ret    = (nav.iloc[-1] - nav.iloc[0]) / nav.iloc[0] * 100
    bar_rets     = nav.pct_change().dropna()
    # Annualise assuming 96 × M15 bars per trading day
    sharpe       = (float(bar_rets.mean() / bar_rets.std()) * (252 * 96) ** 0.5
                    if bar_rets.std() != 0 else 0.0)
    dd           = (nav - nav.cummax()) / nav.cummax()
    max_dd       = float(dd.min() * 100)
    final_nav    = float(nav.iloc[-1])

    if trades_df.empty or "pnl" not in trades_df.columns:
        return dict(total_return_pct=round(total_ret, 2), sharpe=round(sharpe, 3),
                    max_drawdown_pct=round(max_dd, 2), trade_count=0,
                    win_rate_pct=0.0, avg_win=0.0, avg_loss=0.0,
                    final_nav=round(final_nav, 2))

    closed   = trades_df[trades_df["pnl"] != 0.0]
    tc       = len(closed)
    wins     = int((closed["pnl"] > 0).sum())
    wr       = wins / tc * 100 if tc else 0.0
    avg_win  = float(closed.loc[closed["pnl"] > 0, "pnl"].mean()) if wins        else 0.0
    avg_loss = float(closed.loc[closed["pnl"] < 0, "pnl"].mean()) if tc - wins  else 0.0
    sl_count = int((trades_df["action"] == "SL_CLOSE").sum())
    tp_count = int((trades_df["action"] == "TP_CLOSE").sum())

    return dict(
        total_return_pct = round(total_ret, 2),
        sharpe           = round(sharpe, 3),
        max_drawdown_pct = round(max_dd, 2),
        trade_count      = tc,
        win_rate_pct     = round(wr, 2),
        avg_win          = round(avg_win, 4),
        avg_loss         = round(avg_loss, 4),
        final_nav        = round(final_nav, 2),
        sl_count         = sl_count,
        tp_count         = tp_count,
    )
