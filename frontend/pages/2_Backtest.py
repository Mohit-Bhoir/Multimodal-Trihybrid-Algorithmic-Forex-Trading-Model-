"""
Iterative Backtesting Engine — Page 2
=======================================
Configure strategy parameters, run a backtest on historical EUR/USD data,
inspect equity curve / drawdown, trade log, and deploy to live.
All times in British Summer Time (Europe/London).
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "frontend"))

from utils.trading_engine import (
    BST, TRAIN_CSV, TEST_CSV,
    BacktestEngine, load_lstm_artifacts,
)

BST_TZ = ZoneInfo("Europe/London")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Backtest Engine",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Iterative Backtesting Engine")
st.caption("Configure strategy parameters, run a simulation, then deploy to live trading.")

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading LSTM model…")
def _get_model():
    return load_lstm_artifacts()


# Slider defaults — initialised once per session
_BT_UI_DEFAULTS = {
    "long_threshold":  0.55,
    "short_threshold": 0.45,
    "sl_pct":          0.10,
    "tp_pct":          0.20,
    "capital":         100_000.0,
}
for _k, _v in _BT_UI_DEFAULTS.items():
    if f"bt_{_k}" not in st.session_state:
        st.session_state[f"bt_{_k}"] = _v


@st.cache_resource
def _get_bt_results() -> dict:
    """Persist last backtest result + params across refreshes and page navigation."""
    return {"result": None, "params": {}}


try:
    model, mean, std, feature_cols, lookback, window = _get_model()
except Exception as exc:
    st.error(f"Cannot load LSTM model: {exc}. Train the model first.")
    st.stop()

# ── Load historical data (cached) ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading historical data…")
def _load_data() -> pd.DataFrame:
    frames = []
    for path in [TRAIN_CSV, TEST_CSV]:
        if path.exists():
            df = pd.read_csv(path, parse_dates=["time"])
            cols = ["time", "price"] + (["spread"] if "spread" in df.columns else [])
            frames.append(df[cols])
    if not frames:
        return pd.DataFrame()
    data = (pd.concat(frames)
              .sort_values("time")
              .drop_duplicates("time")
              .set_index("time"))
    data.index = pd.to_datetime(data.index)
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize("UTC").tz_convert(BST_TZ)
    else:
        data.index = data.index.tz_convert(BST_TZ)
    return data


all_data = _load_data()

if all_data.empty:
    st.error("No historical data found. Run the data pipeline (fetch + preprocess) first.")
    st.stop()

_d_min = all_data.index.min().date()
_d_max = all_data.index.max().date()

# ── Sidebar: all strategy inputs ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Strategy Parameters")

    # ── Date range ────────────────────────────────────────────────────────────
    st.subheader("📅 Date Range")
    start_date = st.date_input("Start Date", value=_d_min,
                               min_value=_d_min, max_value=_d_max)
    end_date   = st.date_input("End Date",   value=_d_max,
                               min_value=_d_min, max_value=_d_max)

    # ── Signal thresholds ─────────────────────────────────────────────────────
    st.subheader("🎯 Signal Thresholds")
    long_threshold  = st.slider("Long Threshold",  0.50, 1.00,
                                 key="bt_long_threshold",
                                 help="Prob ≥ this → LONG signal",
                                 step=0.01)
    short_threshold = st.slider("Short Threshold", 0.00, 0.50,
                                 key="bt_short_threshold",
                                 help="Prob ≤ this → SHORT signal",
                                 step=0.01)

    # ── Risk management ───────────────────────────────────────────────────────
    st.subheader("\U0001f6e1\ufe0f Risk Management")
    _sl_pct = st.slider("Stop Loss (%)",   0.10, 5.0,
                        key="bt_sl_pct",
                        step=0.10,
                        help="Close losing position when return from entry falls by this %. "
                             "For EUR/USD: 0.10% \u2248 12 pips.")
    _tp_pct = st.slider("Take Profit (%)", 0.10, 5.0,
                        key="bt_tp_pct",
                        step=0.10,
                        help="Close winning position when return from entry rises by this %. "
                             "For EUR/USD: 0.20% \u2248 23 pips.")
    stop_loss   = _sl_pct / 100
    take_profit = _tp_pct / 100
    st.caption("\U0001f4cb Spread is taken from the actual per-bar values in the CSV data.")

    # ── Capital ───────────────────────────────────────────────────────────────
    st.subheader("💰 Capital")
    capital = st.number_input("Starting Capital (£)", min_value=1_000.0,
                               key="bt_capital", step=1_000.0, format="%.2f")

    st.divider()
    run_btn = st.button("▶️  Run Backtest", type="primary", use_container_width=True)

# ── Persistent backtest results (survive refresh + page navigation) ───────────
_bt_store = _get_bt_results()

# Hardcoded defaults — not exposed in UI
min_confidence = 0.10
bar_window     = window   # use the window from model stats

# ── Run backtest ───────────────────────────────────────────────────────────────
if run_btn:
    if start_date >= end_date:
        st.error("Start date must be before end date.")
    else:
        _mask   = (all_data.index.date >= start_date) & (all_data.index.date <= end_date)
        _subset = all_data[_mask].copy()

        _min_needed = lookback + 200
        if len(_subset) < _min_needed:
            st.error(
                f"Only {len(_subset):,} bars in the selected range. "
                f"Need at least {_min_needed:,}. Widen the date range."
            )
        else:
            _engine = BacktestEngine(
                model=model, mean=mean, std=std,
                feature_cols=feature_cols, lookback=lookback, window=bar_window,
                long_threshold  = long_threshold,
                short_threshold = short_threshold,
                stop_loss       = stop_loss,
                take_profit     = take_profit,
                capital         = capital,
                min_confidence  = min_confidence,
            )
            with st.spinner(f"Running backtest on {len(_subset):,} bars…"):
                _result = _engine.run(_subset)

            if "error" in _result:
                st.error(_result["error"])
            else:
                _bt_store["result"] = _result
                _bt_store["params"] = dict(
                    long_threshold  = long_threshold,
                    short_threshold = short_threshold,
                    stop_loss       = stop_loss,
                    take_profit     = take_profit,
                    capital         = capital,
                    start_date      = str(start_date),
                    end_date        = str(end_date),
                )
                st.success("✅ Backtest complete!")

# ── No results yet ─────────────────────────────────────────────────────────────
if _bt_store["result"] is None:
    st.info("Configure parameters in the sidebar, then click **▶️ Run Backtest**.")
    with st.expander("📂 Available Data Preview"):
        st.write(
            f"**{len(all_data):,} bars** available — "
            f"{_d_min} → {_d_max} (M15, BST)"
        )
        st.dataframe(all_data.tail(30), use_container_width=True)
    st.stop()

# ── Results ────────────────────────────────────────────────────────────────────
result    = _bt_store["result"]
metrics   = result.get("metrics", {})
equity_df = result.get("equity_curve", pd.DataFrame())
trades_df = result.get("trades",       pd.DataFrame())

# ── Metrics panel ──────────────────────────────────────────────────────────────
st.subheader("📊 Performance Metrics")

_m1, _m2, _m3, _m4, _m5, _m6 = st.columns(6)

with _m1:
    _val = metrics.get("total_return_pct", 0.0)
    st.metric("Total Return", f"{_val:+.2f}%",
              delta_color="normal" if _val >= 0 else "inverse")
with _m2:
    st.metric("Final NAV", f"£{metrics.get('final_nav', capital):,.2f}")
with _m3:
    st.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.3f}")
with _m4:
    st.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
with _m5:
    st.metric("Trade Count", metrics.get("trade_count", 0))
with _m6:
    st.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.1f}%")

# Secondary row
_s1, _s2, _s3, _s4, _s5 = st.columns(5)
with _s1:
    st.metric("Avg Win (£)", f"{metrics.get('avg_win', 0):+,.4f}")
with _s2:
    st.metric("Avg Loss (£)", f"{metrics.get('avg_loss', 0):+,.4f}")
with _s3:
    _aw = abs(metrics.get("avg_win", 0))
    _al = abs(metrics.get("avg_loss", 1)) or 1
    st.metric("Reward / Risk", f"{_aw / _al:.2f}\u00d7")
with _s4:
    st.metric("\U0001f534 SL Triggered", metrics.get("sl_count", 0),
              help="Number of positions closed by Stop Loss")
with _s5:
    st.metric("\U0001f7e2 TP Triggered", metrics.get("tp_count", 0),
              help="Number of positions closed by Take Profit")

# ── Parameters used in this run ──────────────────────────────────────────────
if _bt_store["params"]:
    _p = _bt_store["params"]
    st.caption(
        f"**Run parameters:** "
        f"Long ≥ {_p.get('long_threshold', '?'):.2f}  ·  "
        f"Short ≤ {_p.get('short_threshold', '?'):.2f}  ·  "
        f"SL {_p.get('stop_loss', 0)*100:.2f}%  ·  "
        f"TP {_p.get('take_profit', 0)*100:.2f}%  ·  "
        f"Capital £{_p.get('capital', 0):,.0f}  ·  "
        f"{_p.get('start_date', '?')} → {_p.get('end_date', '?')}"
    )

st.divider()

# ── Equity curve + Drawdown ────────────────────────────────────────────────────
st.subheader("📈 Equity Curve & Drawdown")

if not equity_df.empty and "nav" in equity_df.columns:
    _nav = equity_df["nav"]
    _dd  = (_nav - _nav.cummax()) / _nav.cummax() * 100

    # ── Buy-and-hold benchmark ────────────────────────────────────────────────
    _bah_nav = None
    try:
        _bah_s  = pd.to_datetime(_bt_store["params"]["start_date"]).date()
        _bah_e  = pd.to_datetime(_bt_store["params"]["end_date"]).date()
        _bah_m  = (all_data.index.date >= _bah_s) & (all_data.index.date <= _bah_e)
        _bah_px = all_data[_bah_m]["price"].dropna()
        if len(_bah_px) > 1:
            _bah_nav = capital * (_bah_px / float(_bah_px.iloc[0]))
    except Exception:
        pass

    _fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32],
        subplot_titles=("Portfolio NAV (£)", "Drawdown (%)"),
        vertical_spacing=0.07,
    )

    # Strategy NAV
    _fig.add_trace(
        go.Scatter(x=_nav.index, y=_nav.values,
                   mode="lines", name="Strategy",
                   line=dict(color="#4DA6FF", width=1.5)),
        row=1, col=1,
    )
    # Buy-and-hold benchmark
    if _bah_nav is not None:
        _fig.add_trace(
            go.Scatter(x=_bah_nav.index, y=_bah_nav.values,
                       mode="lines", name="Buy & Hold",
                       line=dict(color="#FFC107", width=1.2, dash="dot")),
            row=1, col=1,
        )
    # Starting capital baseline
    _fig.add_hline(y=capital, line_dash="dash",
                   line_color="rgba(255,255,255,0.3)", row=1, col=1)

    # Trade entry markers on equity curve
    if not trades_df.empty:
        _entries = trades_df[trades_df["action"].isin(["LONG", "SHORT"])]
        for _, _tr in _entries.iterrows():
            _nav_at = float(_nav.asof(_tr["time"])) if hasattr(_nav, "asof") else capital
            _col    = "#4CAF50" if _tr["action"] == "LONG" else "#F44336"
            _sym    = "triangle-up" if _tr["action"] == "LONG" else "triangle-down"
            _fig.add_trace(
                go.Scatter(
                    x=[_tr["time"]], y=[_nav_at],
                    mode="markers",
                    marker=dict(symbol=_sym, size=9, color=_col,
                                line=dict(width=1, color="white")),
                    showlegend=False, name=_tr["action"],
                ),
                row=1, col=1,
            )

    # Drawdown
    _fig.add_trace(
        go.Scatter(x=_dd.index, y=_dd.values,
                   mode="lines", name="Drawdown %",
                   line=dict(color="#F44336", width=1),
                   fill="tozeroy", fillcolor="rgba(244,67,54,0.12)"),
        row=2, col=1,
    )

    _fig.update_layout(
        template="plotly_dark", height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="left", x=0),
    )
    _fig.update_xaxes(title_text="Time (BST)", row=2, col=1)
    _fig.update_yaxes(title_text="£", row=1, col=1)
    _fig.update_yaxes(title_text="%", row=2, col=1)

    st.plotly_chart(_fig, use_container_width=True)
else:
    st.warning("No equity data to display.")

# ── Trade log ──────────────────────────────────────────────────────────────────
st.subheader("📋 Trade Log")

if not trades_df.empty:
    _display = trades_df.sort_values("time", ascending=False).reset_index(drop=True)
    st.dataframe(_display, use_container_width=True, height=320)
else:
    st.info("No trades were executed in the selected period.")

st.divider()

# ── Action buttons ─────────────────────────────────────────────────────────────
_b1, _b2, _b3 = st.columns(3)

with _b1:
    if st.button("💾  Save Strategy", use_container_width=True):
        _params    = _bt_store["params"]
        _save_dir  = ROOT / "strategies"
        _save_dir.mkdir(exist_ok=True)
        _fname     = _save_dir / f"strategy_{datetime.now(BST_TZ).strftime('%Y%m%d_%H%M%S')}.json"
        _fname.write_text(json.dumps(_params, indent=2))
        st.success(f"Strategy saved → `{_fname.relative_to(ROOT)}`")

with _b2:
    if st.button("🔄  Re-run Backtest", use_container_width=True):
        _bt_store["result"] = None
        st.rerun()

with _b3:
    if st.button("🚀  Deploy to Live", type="primary", use_container_width=True):
        st.session_state["deploy_params"] = _bt_store["params"]
        st.switch_page("pages/1_Live_Trading.py")
