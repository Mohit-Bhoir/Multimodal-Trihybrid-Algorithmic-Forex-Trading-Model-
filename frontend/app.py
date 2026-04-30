"""
EUR/USD LSTM Trading System — Home / Entry Point
=================================================
Run with:   streamlit run frontend/app.py
"""
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

from utils.ui import inject_page_chrome, render_disclaimer, render_footer

# ── Suppress noisy log spam ────────────────────────────────────────────────────
# TF / oneDNN warnings (must be set before tensorflow is imported anywhere)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Silence tornado WebSocketClosedError "Task exception was never retrieved" noise.
# This fires harmlessly whenever a browser tab closes mid-fragment-update.
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
logging.getLogger("tornado.access").setLevel(logging.ERROR)

BST_TZ = ZoneInfo("Europe/London")

st.set_page_config(
    page_title="EUR/USD LSTM Trading System",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_page_chrome()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("💹 EUR/USD LSTM Algorithmic Trading System")
st.caption(
    f"🕐 {datetime.now(BST_TZ).strftime('%d %b %Y  %H:%M:%S')} BST  |  "
    "All times displayed in British Summer Time (Europe/London)"
)
render_disclaimer()
st.divider()

# ── Navigation cards ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
### 📈 Live Trading Simulation

Connect to the **OANDA v20 API** and run the LSTM model in real time.

| What you see | What you can do |
|---|---|
| Live EUR/USD price & M15 bar close time | Enable **auto-execute** trades |
| Current position (Long / Short / Flat) | **Manually close** any open position |
| Portfolio NAV, P&L & bid/ask spread | Tune min-confidence threshold |
| LSTM model probability | Review full trade & decision logs |
| **Volatility regime** (Low / Medium / High) | Observe regime-aware signal context |

> Auto-refreshes every **15 seconds** via OANDA REST polling.
""")
    if st.button("📈  Open Live Trading", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Live_Trading.py")

with col2:
    st.markdown("""
### 🔬 Iterative Backtesting Engine

Simulate the strategy on **historical M15 EUR/USD data** before going live.

| Inputs | Outputs |
|---|---|
| Date range | Equity curve & drawdown chart |
| Long / Short thresholds | Total return, Sharpe, max drawdown |
| **Stop-loss & take-profit** (pip-calibrated) | Win rate & trade count |
| Feature window & actual per-bar spread | Full trade log with P&L |

> Results can be **saved as JSON** or **deployed directly to Live Trading**.
""")
    if st.button("🔬  Open Backtest Engine", use_container_width=True):
        st.switch_page("pages/2_Backtest.py")

st.divider()

# ── ML Pipeline card ───────────────────────────────────────────────────────────
st.markdown("### ⚙️ Automated ML Pipeline")

pipe_col1, pipe_col2 = st.columns(2, gap="large")

with pipe_col1:
    st.markdown("""
**Daily Data Refresh** *(Airflow DAG — Mon–Fri 22:00 UTC)*

Fetches the latest M15 EUR/USD candles from OANDA via count-based API and
appends them to the backtest CSV. Deduplicates and sorts automatically so
the Backtest Engine always has up-to-date market data.

| Component | Detail |
|---|---|
| Source | OANDA v20 REST — `EUR_USD` M15 candles |
| Output | `data/backtest/forex_data_backtest.csv` |
| Dedup | Timestamp-keyed, append-only |
""")

with pipe_col2:
    st.markdown("""
**Weekly LSTM Retraining** *(Airflow DAG — Sunday 02:00 UTC)*

End-to-end retraining pipeline with hyperparameter grid search and
walk-forward quality gating before the live model is replaced.

| Stage | What happens |
|---|---|
| Fetch | Refresh training data from OANDA (skips if fresh) |
| Preprocess | Train / test split — last 4 months held out |
| Tune & Train | Grid search over units, dropout, LR, batch size |
| Walk-Forward Backtest | 4-window iterative validation |
| Quality Gate | Sharpe ≥ 0.3 · Max DD ≥ −25 % · Trades ≥ 20 |
| Archive | Saves model, stats, metrics & params to `models/archive/` |
| Promote | Overwrites live `lstm_model.h5` only on gate PASS |
""")

st.divider()

# ── System overview ────────────────────────────────────────────────────────────
st.markdown("""
#### ℹ️ System Overview

| Component | Detail |
|---|---|
| **Model** | LSTM neural network — binary classifier (price up/down) on M15 EUR/USD |
| **Signal** | LSTM probability scaled to \[-1, +1\]; trades when \|signal\| > min-confidence |
| **Features** | Log-returns, SMA cross, Bollinger, min/max, momentum, volatility (7 features) |
| **Risk controls** | Per-trade stop-loss & take-profit; configurable in both live and backtest modes |
| **Volatility regime** | Rolling ATR classified as Low / Medium / High; displayed on Live Trading page |
| **Bar timing** | M15 bar **close** time displayed (bar start + 15 min) for precise signal attribution |
| **Retraining** | Weekly Airflow DAG — grid search → walk-forward backtest → quality gate → archive |
| **Broker** | OANDA v20 API via `tpqoa` (practice or live account) |
| **Orchestration** | Apache Airflow 2.10 on Astronomer (Astro CLI / Docker) |
| **Timezone** | British Summer Time — `Europe/London` (GMT in winter, BST+1 in summer) |
""")

render_footer()
