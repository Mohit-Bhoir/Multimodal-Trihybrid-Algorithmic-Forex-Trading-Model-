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
| Live EUR/USD price | Enable **auto-execute** trades |
| Current position (Long / Short / Flat) | **Manually close** any open position |
| Portfolio NAV & P&L | Tune min-confidence threshold |
| LSTM model probability | Review full trade & decision logs |

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
| Stop-loss & take-profit | Win rate & trade count |
| Feature window & spread | Full trade log |

> Results can be **saved as JSON** or **deployed directly to Live Trading**.
""")
    if st.button("🔬  Open Backtest Engine", use_container_width=True):
        st.switch_page("pages/2_Backtest.py")

st.divider()

# ── System overview ────────────────────────────────────────────────────────────
st.markdown("""
#### ℹ️ System Overview

| Component | Detail |
|---|---|
| **Model** | LSTM neural network — binary classifier (price up/down) on M15 EUR/USD |
| **Signal** | LSTM probability scaled to \[-1, +1\]; trades when \|signal\| > min-confidence |
| **Features** | Log-returns, SMA cross, Bollinger, min/max, momentum, volatility |
| **Broker** | OANDA v20 API via `tpqoa` (practice or live account) |
| **Timezone** | British Summer Time — `Europe/London` (GMT in winter, BST+1 in summer) |
""")

render_footer()
