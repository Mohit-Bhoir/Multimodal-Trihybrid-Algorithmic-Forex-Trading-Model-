"""
EUR/USD LSTM Trading System — Home / Entry Point
=================================================
Run with:   streamlit run frontend/app.py
"""
import logging
import os
from datetime import datetime
from pathlib import Path
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

LONDON_TZ = ZoneInfo("Europe/London")
ROOT = Path(__file__).resolve().parent.parent
BACKTEST_CSV = ROOT / "data" / "backtest" / "forex_data_backtest.csv"


@st.cache_data(ttl=300, show_spinner=False)
def _dataset_span() -> tuple[str, str, int]:
    """Read first/last timestamp and row count of the backtest CSV."""
    if not BACKTEST_CSV.exists():
        return ("N/A", "N/A", 0)
    try:
        lines = BACKTEST_CSV.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) < 2:
            return ("N/A", "N/A", 0)
        first_ts = lines[1].split(",")[0][:10]
        last_ts = lines[-1].split(",")[0][:10]
        return (first_ts, last_ts, len(lines) - 1)
    except Exception:
        return ("N/A", "N/A", 0)


st.set_page_config(
    page_title="EUR/USD LSTM Trading System",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_page_chrome()

# ── Header ─────────────────────────────────────────────────────────────────────
_now_london = datetime.now(LONDON_TZ)
_tz_abbr = _now_london.strftime("%Z")
_ds_start, _ds_end, _ds_rows = _dataset_span()

st.title("💹 EUR/USD LSTM Algorithmic Trading System")
st.caption(
    f"🕐 {_now_london.strftime('%d %b %Y  %H:%M:%S')} {_tz_abbr}  |  "
    "All times displayed in Europe/London (GMT in winter, BST in summer)"
)
render_disclaimer()
st.divider()

# ── Navigation cards ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
### 📈 Live Trading Simulation

Connect to the **OANDA v20 Practice API** and run the LSTM model in real time.

| What you see | What you can do |
|---|---|
| Live EUR/USD price & M15 bar close time | Enable **auto-execute** demo trades |
| Current position (Long / Short / Flat) | **Manually close** any open position |
| Portfolio NAV, P&L & bid/ask spread (pips) | Tune Long / Short probability thresholds |
| LSTM probability & session decision log | Configure Stop Loss & Take Profit (%) |
| **Volatility regime** (Low / Normal / High) | Observe regime-adjusted signal bounds |

> Auto-refreshes every **15 seconds** with live OANDA streaming ticks.
""")
    if st.button("📈  Open Live Trading", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Live_Trading.py")

with col2:
    st.markdown(f"""
### 🔬 Iterative Backtesting Engine

Simulate the LSTM strategy on **historical M15 EUR/USD data** before going live.

| Inputs | Outputs |
|---|---|
| Date range (defaults to out-of-sample) | Mark-to-market equity & drawdown chart |
| Long / Short probability thresholds | Total return, Sharpe & max drawdown |
| **Stop-loss & take-profit** (% of entry) | Win rate, reward/risk & SL/TP counts |
| Starting capital (£ simulated notional) | Full trade log with per-bar spread |
| Dataset: `{_ds_start}` → `{_ds_end}` | `{_ds_rows:,}` M15 bars (OOS from `2026-01-09`) |

> Results can be **exported as JSON** or **deployed directly to Live Trading**.
""")
    if st.button("🔬  Open Backtest Engine", use_container_width=True):
        st.switch_page("pages/2_Backtest.py")

st.divider()

# ── ML Pipeline card ───────────────────────────────────────────────────────────
st.markdown("### ⚙️ MLOps Pipeline *(Local Docker / Apache Airflow — Astro CLI)*")
st.caption(
    "The Airflow DAGs below run in the local Docker/Astronomer stack (`dags/`) rather than inside "
    "Streamlit Community Cloud. This hosted demo serves the committed model and dataset snapshot "
    f"(`{_ds_start}` → `{_ds_end}`)."
)

pipe_col1, pipe_col2 = st.columns(2, gap="large")

with pipe_col1:
    st.markdown("""
**Daily Data Refresh** *(`dags/daily_data_refresh.py` — Mon–Fri 22:00 UTC)*

Fetches the latest completed M15 EUR/USD candles from OANDA via the count-based
API and appends them to the backtest CSV with timestamp deduplication.

| Component | Detail |
|---|---|
| Source | OANDA v20 REST — `EUR_USD` M15 candles |
| Output | `data/backtest/forex_data_backtest.csv` |
| Dedup | Timestamp-keyed, append-only |
""")

with pipe_col2:
    st.markdown("""
**Weekly LSTM Retraining** *(`dags/retrain_lstm.py` — Sunday 02:00 UTC)*

End-to-end retraining DAG with `TimeSeriesSplit` hyperparameter grid search and
4-window walk-forward quality gating before promoting a candidate model.

| Stage | What happens |
|---|---|
| Fetch & Split | Refresh training CSV & hold out last 4 months as test |
| Tune & Train | Grid search over LSTM units, dropout, LR & batch size (isolated TF venv) |
| Walk-Forward | 4-window contiguous-warmup backtest with spread & MTM equity |
| Quality Gate | Avg Sharpe ≥ 0.3 · Worst DD ≥ −25% · Trades ≥ 20 |
| Archive & Promote | Archives artifacts locally and promotes `lstm_model.h5` on PASS |
""")

st.divider()

# ── System overview ────────────────────────────────────────────────────────────
st.markdown("""
#### ℹ️ System Overview

| Component | Detail |
|---|---|
| **Scope** | Phase 1 unimodal price-action LSTM trading & backtesting platform *(experimental NLP sentiment script in `src/news.py` is not yet fused into live signals)* |
| **Model** | 2-layer LSTM binary directional classifier (up/down) on M15 EUR/USD, 50-bar lookback |
| **Features (7)** | `returns`, `sma` (20 vs 150), `boll` (20-bar z-score), `min`, `max`, `mom` (3-bar), `vol` (20-bar) |
| **Signal Rule** | Go LONG if `prob ≥ long_threshold`; go SHORT if `prob ≤ short_threshold`; otherwise FLAT |
| **Volatility Regime** | 20-bar rolling M15 log-return std percentile (200-bar window) → Low / Normal / High; symmetrically widens/narrows live thresholds |
| **Risk Controls** | Per-trade Stop Loss & Take Profit (% of entry price); per-bar bid-ask spread applied on all backtest closes |
| **Broker** | OANDA v20 Practice (Demo) API via `tpqoa` (REST + live tick stream) |
| **Orchestration** | Apache Airflow on Astronomer (Astro CLI / Docker, local stack) + MLflow experiment tracking |
| **Timezone** | `Europe/London` — automatically displays GMT (UTC+0) in winter and BST (UTC+1) in summer |
""")

render_footer()
