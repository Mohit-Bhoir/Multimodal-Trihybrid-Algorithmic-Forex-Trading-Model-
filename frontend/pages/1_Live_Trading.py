"""Live Trading Simulation - Page 1"""

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from html import escape

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "frontend"))

from utils.trading_engine import (
    OANDA_CONFIG_PATH,
    drain_tick_buffer,
    fetch_recent_bars,
    get_account_info,
    get_bid_ask_price,
    get_current_price,
    get_oanda_config_path,
    get_open_position,
    get_recent_trades,
    get_volatility_regime,
    is_stream_running,
    load_lstm_artifacts,
    predict_signal,
    start_tick_stream,
)
from utils.ui import inject_page_chrome, render_disclaimer, render_footer

st.set_page_config(page_title="Live Trading", page_icon="📈", layout="wide")

try:
    from streamlit_autorefresh import st_autorefresh
    REFRESH_INTERVAL_S = 15
    st_autorefresh(interval=REFRESH_INTERVAL_S * 1000, key="live_autorefresh")
except ImportError:
    REFRESH_INTERVAL_S = 15

BST_TZ = ZoneInfo("Europe/London")

inject_page_chrome()
st.markdown("""
<style>
div[data-testid="metric-container"] {
    background: var(--secondary-background-color); border-radius: 10px;
    padding: 14px 16px; border-left: 4px solid #4DA6FF;
}
.stButton > button[kind="primary"] { background-color: #d32f2f; color: white; }
.bot-status-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(127, 127, 127, 0.18);
    border-left-width: 4px;
    padding: 8px 12px;
    border-radius: 8px;
    margin: 4px 0 10px 0;
    color: var(--text-color);
}
.tick-feed-box {
    height: 300px;
    overflow-y: auto;
    background: var(--secondary-background-color);
    border: 1px solid rgba(127, 127, 127, 0.22);
    border-radius: 8px;
    padding: 10px 14px;
    font-family: monospace;
    font-size: 12px;
    color: var(--text-color);
    line-height: 1.7;
}
.vol-regime-box {
    height: 300px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: var(--secondary-background-color);
    border: 1px solid rgba(127, 127, 127, 0.22);
    border-radius: 8px;
    padding: 16px 12px;
    text-align: center;
    gap: 10px;
}
.vol-regime-label {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.vol-regime-stat {
    font-size: 12px;
    opacity: 0.75;
    color: var(--text-color);
    line-height: 1.8;
}
</style>""", unsafe_allow_html=True)

_DEFAULTS = dict(trade_log=[], decision_log=[], price_history=[], tick_log=[], auto_trade=False)
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_DEFAULT_SLIDER_PARAMS = {
    "long_threshold":  0.55,
    "short_threshold": 0.45,
    "stop_loss":       0.50,   # percent
    "take_profit":     1.00,   # percent
    "trade_units":     100_000,
}

# ── Persistent app state (survives hard page refreshes & reconnections) ────────
@st.cache_resource
def _get_app_state() -> dict:
    return {
        "auto_trade":        False,
        "trade_log":         [],
        "decision_log":      [],
        "price_history":     [],
        "tick_log":          [],
        "equity_history":    [],
        "last_decision_bar": None,
        "latest_signal":     {},   # most recent predict_signal result
        # Slider values persisted across page navigation
        "slider_params":     _DEFAULT_SLIDER_PARAMS.copy(),
        # Snapshot locked in when bot starts (used for trade execution)
        "active_params":     _DEFAULT_SLIDER_PARAMS.copy(),
        # Volatility regime frozen at each M15 candle boundary
        "frozen_regime":     None,   # dict from get_volatility_regime() or None
        "last_candle_bar":   None,   # M15 boundary string "YYYY-MM-DD HH:MM"
    }

_app = _get_app_state()
# Back-compat: add new keys if running against an older cached state
for _fk, _fv in [("slider_params", _DEFAULT_SLIDER_PARAMS.copy()),
                  ("active_params",  _DEFAULT_SLIDER_PARAMS.copy()),
                  ("latest_signal",  {}),
                  ("last_pred_updated", None),
                  ("last_decision_bar_logged", None),
                  ("frozen_regime", None),
                  ("last_candle_bar", None)]:
    if _fk not in _app:
        _app[_fk] = _fv


def _effective_thresholds(
    base_long: float, base_short: float, frozen_regime: dict | None
) -> tuple[float, float]:
    """Return (long_thr, short_thr) adjusted by the frozen volatility regime.

    HIGH volatility → raise the bar (require stronger signal to trade).
    LOW  volatility → lower the bar (trade on weaker signal).
    NORMAL / UNKNOWN → use base values unchanged.
    """
    regime = (frozen_regime or {}).get("regime", "NORMAL")
    if regime == "HIGH":
        return min(base_long + 0.03, 1.0), max(base_short - 0.03, 0.0)
    if regime == "LOW":
        return max(base_long - 0.03, 0.50), min(base_short + 0.03, 0.50)
    return base_long, base_short

if "deploy_params" in st.session_state and st.session_state.get("deploy_params"):
    _dp = st.session_state.pop("deploy_params")
    # Probability thresholds arrive as-is (0–1 fractions)
    for _k in ("long_threshold", "short_threshold"):
        if _k in _dp:
            _app["slider_params"][_k] = float(_dp[_k])
    # Risk params: backtest stores fractions (e.g. 0.001), live uses percentages (e.g. 0.10)
    for _k in ("stop_loss", "take_profit"):
        if _k in _dp:
            _app["slider_params"][_k] = float(_dp[_k]) * 100
    st.success(
        f"Deployed from Backtest — "
        f"long_thr={_app['slider_params']['long_threshold']:.2f}, "
        f"short_thr={_app['slider_params']['short_threshold']:.2f}, "
        f"SL={_app['slider_params']['stop_loss']:.2f}%, "
        f"TP={_app['slider_params']['take_profit']:.2f}%",
        icon="🚀",
    )

INSTRUMENT = "EUR_USD"

with st.sidebar:
    st.header("\u2699\ufe0f Settings")
    _bot_on = _app["auto_trade"]
    _sp     = _app["slider_params"]   # editable values (bot OFF)
    _ap     = _app["active_params"]   # locked snapshot (bot ON)
    trade_units = st.number_input(
        "Trade Units", min_value=1_000, max_value=10_000_000,
        value=int(_ap["trade_units"] if _bot_on else _sp["trade_units"]),
        step=10_000, disabled=_bot_on,
    )
    if not _bot_on:
        _sp["trade_units"] = trade_units
    st.divider()
    # ── Bot ON / OFF ──────────────────────────────────────────────────────
    st.subheader("\U0001f916 Auto Trading Bot")
    _bot_label = "Bot: ON \U0001f7e2" if _app["auto_trade"] else "Bot: OFF \U0001f534"
    if st.button(_bot_label, use_container_width=True):
        _new_bot_state = not _app["auto_trade"]
        _app["auto_trade"] = _new_bot_state
        if _new_bot_state:   # turning ON — snapshot current slider values as active params
            _app["active_params"] = _app["slider_params"].copy()
        else:                # turning OFF — clear stale prediction so strip stays blank
            _app["latest_signal"] = {}
        st.rerun()
    _stream_alive = is_stream_running()
    if _app["auto_trade"] and _stream_alive:
        _bot_color, _bot_dot, _bot_status = "#4CAF50", "\U0001f7e2", "Running"
    elif _app["auto_trade"]:
        _bot_color, _bot_dot, _bot_status = "#FFC107", "\U0001f7e1", "Paused — Stream Connecting\u2026"
    else:
        _bot_color, _bot_dot, _bot_status = "#F44336", "\U0001f534", "Stopped"
    st.markdown(
        f'<div class="bot-status-card" style="border-left-color:{_bot_color};">'
        f'{_bot_dot} <b style="color:{_bot_color};font-size:14px;">{_bot_status}</b></div>',
        unsafe_allow_html=True,
    )
    # ── Close Position button (sidebar) ───────────────────────────────────
    _close_btn = st.button(
        "\U0001f6d1 Close All Positions", use_container_width=True,
        help="Immediately close any open OANDA position."
    )
    st.divider()
    if _bot_on:
        st.info(
            "🔒 **Bot is running** — settings are locked.\n\n"
            "Restart the bot to apply new values.",
        )
    st.subheader("\U0001f3af Signal Thresholds")
    long_threshold = st.slider(
        "Long Threshold", min_value=0.50, max_value=1.00,
        value=float(_ap["long_threshold"] if _bot_on else _sp["long_threshold"]),
        step=0.01, help="Model prob ≥ this → go LONG",
        disabled=_bot_on,
    )
    short_threshold = st.slider(
        "Short Threshold", min_value=0.00, max_value=0.50,
        value=float(_ap["short_threshold"] if _bot_on else _sp["short_threshold"]),
        step=0.01, help="Model prob ≤ this → go SHORT",
        disabled=_bot_on,
    )
    if not _bot_on:
        _sp["long_threshold"]  = long_threshold
        _sp["short_threshold"] = short_threshold
    _last_dir = (_app["decision_log"][-1].get("direction") if _app["decision_log"] else None)
    _z_long  = "rgba(76,175,80,0.22)"  if _last_dir == "LONG"  else "transparent"
    _z_short = "rgba(244,67,54,0.22)" if _last_dir == "SHORT" else "transparent"
    _z_flat  = "rgba(255,193,7,0.22)" if _last_dir == "FLAT"  else "transparent"
    _lt_disp = long_threshold
    _st_disp = short_threshold
    st.markdown(
        f'<div style="font-size:12px;line-height:2.3;margin-top:4px;">'
        f'<span style="display:block;background:{_z_long};padding:1px 8px;border-radius:4px;">'
        f'<span style="color:#4CAF50;font-weight:700;">● LONG</span>'
        f'&nbsp;&nbsp;prob ≥ <code>{_lt_disp:.2f}</code></span>'
        f'<span style="display:block;background:{_z_short};padding:1px 8px;border-radius:4px;">'
        f'<span style="color:#F44336;font-weight:700;">● SHORT</span>'
        f'&nbsp;&nbsp;prob ≤ <code>{_st_disp:.2f}</code></span>'
        f'<span style="display:block;background:{_z_flat};padding:1px 8px;border-radius:4px;">'
        f'<span style="color:#FFC107;font-weight:700;">● FLAT</span>'
        f'&nbsp;&nbsp;{_st_disp:.2f} &lt; prob &lt; {_lt_disp:.2f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    # ── Risk Management ───────────────────────────────────────────────────
    st.subheader("\U0001f6e1\ufe0f Risk Management")
    stop_loss = st.slider(
        "Stop Loss (%)", min_value=0.10, max_value=5.00,
        value=float(_ap["stop_loss"] if _bot_on else _sp["stop_loss"]),
        step=0.10,
        help="Auto-close position when loss reaches this % of entry price.",
        disabled=_bot_on,
    )
    take_profit = st.slider(
        "Take Profit (%)", min_value=0.10, max_value=10.00,
        value=float(_ap["take_profit"] if _bot_on else _sp["take_profit"]),
        step=0.10,
        help="Auto-close position when gain reaches this % of entry price.",
        disabled=_bot_on,
    )
    if not _bot_on:
        _sp["stop_loss"]   = stop_loss
        _sp["take_profit"] = take_profit
    st.caption(
        f"\U0001f534 Stop Loss: **{stop_loss:.2f}%**  \n"
        f"\U0001f7e2 Take Profit: **{take_profit:.2f}%**"
    )
    st.divider()
    if st.button("\U0001f504 Clear Session Logs", use_container_width=True):
        _app["trade_log"]         = []
        _app["decision_log"]      = []
        _app["price_history"]     = []
        _app["tick_log"]          = []
        _app["equity_history"]    = []
        _app["last_decision_bar"] = None
        st.rerun()


@st.cache_resource(show_spinner="Connecting to OANDA…")
def _get_api(cfg: str):
    import tpqoa
    return tpqoa.tpqoa(cfg)


@st.cache_resource(show_spinner="Loading LSTM model…")
def _get_model():
    return load_lstm_artifacts()


@st.cache_data(ttl=30, show_spinner=False)
def _get_recent_m15_history(_api, instrument: str, lookback_n: int, window_n: int) -> pd.DataFrame:
    """Fetch approximately 1 day of completed M15 bars for chart context."""
    hist = fetch_recent_bars(_api, instrument, lookback_n, window_n, granularity="M15")
    if hist.empty:
        return hist
    hist = hist[["price"]].sort_index().tail(96).copy()  # 96 * 15m ≈ 1 day
    return hist


api = model = None
_api_err = _model_err = None

try:
    api = _get_api(get_oanda_config_path())
except Exception as exc:
    _api_err = str(exc)

try:
    model, mean, std, feature_cols, lookback, window = _get_model()
except Exception as exc:
    _model_err = str(exc)

now_bst = datetime.now(BST_TZ)
st.title("📈 Live Trading Simulation — EUR/USD")
st.caption(
    f"🕒 **{now_bst.strftime('%d %b %Y  %H:%M:%S')} BST** | "
    f"Auto-refresh every {REFRESH_INTERVAL_S} s | "
    "Balance, Position & Price pulled live from OANDA Demo Account"
)
render_disclaimer()

_s1, _s2, _s3 = st.columns(3)
with _s1:
    if api:   st.success("OANDA  Connected", icon="🔗")
    else:     st.error(f"OANDA  {_api_err}", icon="❌")
with _s2:
    if model: st.success("LSTM Model  Loaded", icon="🧠")
    else:     st.error(f"Model  {_model_err}", icon="❌")
with _s3:
    _stream_ok = is_stream_running()
    if _app["auto_trade"] and _stream_ok:
        st.success("\U0001f7e2 Bot Running | Stream Live")
    elif _app["auto_trade"]:
        st.warning("\U0001f7e1 Bot Paused | Stream Connecting\u2026")
    else:
        st.error("\U0001f534 Bot Stopped")

if not api:
    st.stop()

st.divider()

_errs = []

latest_price = get_current_price(api, INSTRUMENT)
if latest_price is None:
    _errs.append("Live price unavailable — OANDA pricing endpoint returned nothing.")

acct = get_account_info(api)
if not acct:
    _errs.append("Account summary unavailable — check OANDA credentials.")

oanda_pos = get_open_position(api, INSTRUMENT)

bars = pd.DataFrame()
signal_result = {}

if api:
    # ── Start OANDA tick stream (no-op if already running) ─────────────────
    start_tick_stream(api, INSTRUMENT)

if model:
    bars = fetch_recent_bars(api, INSTRUMENT, lookback, window, granularity="M1")  # ← TESTING: change back to "M15" for production
    if not bars.empty:
        # Apply frozen-regime threshold adjustment before computing the signal.
        _eff_long, _eff_short = _effective_thresholds(
            long_threshold, short_threshold, _app.get("frozen_regime")
        )
        # Compute signal for auto-trade logic below — display is handled by _prediction_feed fragment
        signal_result = predict_signal(
            model, mean, std, feature_cols, lookback, window, bars,
            long_threshold=_eff_long, short_threshold=_eff_short,
        )
        # Store latest signal in app state so fragment + auto-trade can both read it
        _app["latest_signal"] = signal_result

for _e in _errs:
    st.warning(_e, icon="\u26a0\ufe0f")

_sl_frac = stop_loss  / 100.0
_tp_frac = take_profit / 100.0

# ── Sidebar Close Position button handler ─────────────────────────────────────
if _close_btn:
    _sb_side  = oanda_pos["side"]
    _sb_units = oanda_pos["units"]
    if _sb_side != 0:
        _cu = -_sb_side * _sb_units
        try:
            api.create_order(INSTRUMENT, _cu, suppress=True, ret=True)
            _app["trade_log"].append(dict(
                time=now_bst.strftime("%Y-%m-%d %H:%M:%S"),
                action="MANUAL_CLOSE", price=latest_price,
                units=_cu, upl_at_close=oanda_pos["unrealized_pl"],
            ))
            st.toast("Position closed on OANDA.", icon="\U0001f7e2")
        except Exception as _exc:
            st.sidebar.error(f"Close failed: {_exc}")
        oanda_pos = get_open_position(api, INSTRUMENT)
        acct      = get_account_info(api)
        st.rerun()
    else:
        st.sidebar.info("No open position to close.")

if _app["auto_trade"] and latest_price:
    _cur_side = oanda_pos["side"]
    _cur_avg  = oanda_pos["avg_price"]
    _now_str  = now_bst.strftime("%Y-%m-%d %H:%M:%S")

    # ── Stop Loss / Take Profit check ──────────────────────────────────────
    if _cur_side != 0 and _cur_avg:
        _ret = (latest_price - _cur_avg) / _cur_avg * _cur_side
        if _ret <= -_sl_frac:
            _cu = -_cur_side * oanda_pos["units"]
            try:
                api.create_order(INSTRUMENT, _cu, suppress=True, ret=True)
                _app["trade_log"].append(dict(
                    time=_now_str, action="SL_CLOSE",
                    price=latest_price, units=_cu,
                    upl_at_close=oanda_pos["unrealized_pl"],
                ))
                st.toast(f"Stop Loss hit at {latest_price:.5f}", icon="\U0001f534")
            except Exception as _exc:
                st.warning(f"SL close failed: {_exc}", icon="\u26a0\ufe0f")
            oanda_pos = get_open_position(api, INSTRUMENT)
            acct      = get_account_info(api)
        elif _ret >= _tp_frac:
            _cu = -_cur_side * oanda_pos["units"]
            try:
                api.create_order(INSTRUMENT, _cu, suppress=True, ret=True)
                _app["trade_log"].append(dict(
                    time=_now_str, action="TP_CLOSE",
                    price=latest_price, units=_cu,
                    upl_at_close=oanda_pos["unrealized_pl"],
                ))
                st.toast(f"Take Profit hit at {latest_price:.5f}", icon="\U0001f7e2")
            except Exception as _exc:
                st.warning(f"TP close failed: {_exc}", icon="\u26a0\ufe0f")
            oanda_pos = get_open_position(api, INSTRUMENT)
            acct      = get_account_info(api)

    # ── Signal-based trade ─────────────────────────────────────────────────
    if model and signal_result:
        _prob    = signal_result.get("prob") or 0.5
        # Use regime-adjusted effective thresholds for the execution decision
        _exec_long, _exec_short = _effective_thresholds(
            long_threshold, short_threshold, _app.get("frozen_regime")
        )
        _target  = (1 if _prob >= _exec_long else (-1 if _prob <= _exec_short else 0))
        _cur_side = oanda_pos["side"]
        if _target != _cur_side:
            if _cur_side != 0:
                _cu = -_cur_side * oanda_pos["units"]
                try:
                    api.create_order(INSTRUMENT, _cu, suppress=True, ret=True)
                    _app["trade_log"].append(dict(
                        time=_now_str,
                        action="CLOSE_" + ("LONG" if _cur_side == 1 else "SHORT"),
                        price=latest_price, units=_cu,
                        upl_at_close=oanda_pos["unrealized_pl"],
                    ))
                except Exception as _exc:
                    st.warning(f"Auto-close failed: {_exc}", icon="\u26a0\ufe0f")
            if _target != 0:
                _ou = _target * trade_units
                try:
                    api.create_order(INSTRUMENT, _ou, suppress=True, ret=True)
                    _app["trade_log"].append(dict(
                        time=_now_str,
                        action="LONG" if _target == 1 else "SHORT",
                        price=latest_price, units=_ou, upl_at_close=0.0,
                    ))
                except Exception as _exc:
                    st.warning(f"Auto-open failed: {_exc}", icon="\u26a0\ufe0f")
            oanda_pos = get_open_position(api, INSTRUMENT)
            acct      = get_account_info(api)

_balance  = acct.get("balance",       0.0)
_nav      = acct.get("nav",           0.0)
_upl      = acct.get("unrealized_pl", 0.0)
_rpl      = acct.get("realized_pl",   0.0)
_currency = acct.get("currency",     "")
_side     = oanda_pos["side"]
_units    = oanda_pos["units"]
_avg_px   = oanda_pos["avg_price"]
_pos_upl  = oanda_pos["unrealized_pl"]

# ── Track session equity curve (sampled every 15 s with page refresh) ─────────
if _nav:
    _app["equity_history"].append({"time": now_bst.strftime("%Y-%m-%dT%H:%M:%S"), "nav": _nav})
    _app["equity_history"] = _app["equity_history"][-500:]

# ── Strip 1: OANDA Current Position (always visible) ─────────────────────────
_pos_icons = {
    1:  ("\U0001f4c8", "#4CAF50", "#0d2b0d"),
    -1: ("\U0001f4c9", "#F44336", "#2b0d0d"),
    0:  ("\u26aa",     "#888888", "#1a1a2e"),
}
_pico, _pbc, _pbg = _pos_icons.get(_side, _pos_icons[0])
_pdir_lbl = {1: "LONG", -1: "SHORT", 0: "FLAT"}[_side]
if _side != 0 and _avg_px:
    _pos_detail = (
        f"Avg entry: <b style='color:#eee;'>{_avg_px}</b>"
        f" &nbsp;|&nbsp; Units: <b style='color:#eee;'>{_units:,}</b>"
        f" &nbsp;|&nbsp; UPL: <b style='color:#eee;'>{_pos_upl:+.4f} {_currency}</b>"
    )
else:
    _pos_detail = "No open position on OANDA"
st.markdown(
    f'<div style="background:{_pbg};border-left:5px solid {_pbc};padding:8px 16px;'
    f'border-radius:6px;margin:4px 0;display:flex;align-items:center;gap:20px;">'
    f'<span style="font-size:18px;">{_pico}</span>'
    f'<span style="color:{_pbc};font-size:15px;font-weight:700;min-width:60px;">OANDA &nbsp;{_pdir_lbl}</span>'
    f'<span style="color:#aaa;font-size:12px;">{_pos_detail}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

st.divider()
# \u2500\u2500 Position / Balance / Model Prob \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# ── Portfolio summary (single compact row) ────────────────────────────────────
_pf1, _pf2, _pf3, _pf4, _pf5 = st.columns(5)
with _pf1:
    st.metric(f"\U0001f4b0 Balance ({_currency})", f"{_balance:,.2f}")
with _pf2:
    st.metric(
        "\U0001f4ca NAV",
        f"{_nav:,.2f}",
        delta=f"{_nav - _balance:+,.2f}" if abs(_nav - _balance) > 0.01 else None,
    )
with _pf3:
    st.metric("\U0001f4c8 Unrealised P&L", f"{_upl:+,.4f}")
with _pf4:
    st.metric("\U0001f4ca Realised P&L", f"{_rpl:+,.4f}")
with _pf5:
    if not _app["auto_trade"]:
        st.metric("\U0001f9e0 Model Prob", "\u2014", help="Start the bot to enable predictions")
    else:
        _live_sig = _app.get("latest_signal", {})
        _prob = _live_sig.get("prob")
        if _prob is not None:
            _dir = _live_sig.get("direction", "")
            st.metric("\U0001f9e0 Model Prob", f"{_prob:.4f}",
                      delta=f"\u2192 {_dir}")
        else:
            st.metric("\U0001f9e0 Model Prob", "Loading\u2026")

st.divider()

# ── Fragment: Model Prediction Strip (auto-updates every 60s) ─────────────────
@st.fragment(run_every=60)
def _prediction_feed():
    # Only compute + show when bot is ON.
    if not _app["auto_trade"]:
        st.info("Start the bot to enable live model predictions.", icon="\U0001f9e0")
        return
    if not model or not api:
        st.warning("Model or API unavailable.", icon="\u23f3")
        return

    # Wrap everything in try/except — an uncaught exception during a scheduled
    # fragment rerun silently kills Streamlit's auto-refresh timer.
    try:
        _base_long  = float(_app["active_params"]["long_threshold"])
        _base_short = float(_app["active_params"]["short_threshold"])
        # Apply frozen-regime offset so prediction uses the same thresholds as execution
        _lt, _st = _effective_thresholds(_base_long, _base_short, _app.get("frozen_regime"))

        _bars = fetch_recent_bars(api, INSTRUMENT, lookback, window, granularity="M1")
        if _bars.empty:
            st.warning("Waiting for M1 bars\u2026", icon="\u23f3")
            return

        _sig  = predict_signal(model, mean, std, feature_cols, lookback, window, _bars, _lt, _st)
        _prob = _sig.get("prob")
        _dir  = _sig.get("direction", "FLAT")
        _bar_ts = str(_bars.index[-1])[:16]

        # Persist for auto-trade + portfolio metric
        _app["latest_signal"]     = _sig
        _app["last_decision_bar"] = _bar_ts
        _now_ts = datetime.now(BST_TZ).strftime("%H:%M:%S")
        _app["last_pred_updated"] = _now_ts

        # Decision log — one entry per new bar
        if _bar_ts != _app.get("last_decision_bar_logged"):
            _app["last_decision_bar_logged"] = _bar_ts
            _app["decision_log"].append({
                "bar":       _bar_ts,
                "prob":      round(float(_prob), 4) if _prob is not None else None,
                "direction": _dir,
            })
            _app["decision_log"] = _app["decision_log"][-200:]

        _clr_map  = {"LONG": ("#4CAF50", "#0d2b0d"), "SHORT": ("#F44336", "#2b0d0d"), "FLAT": ("#FFC107", "#2b2500")}
        _bc, _bg  = _clr_map.get(_dir, ("#888888", "#1a1a2e"))
        _dir_icon = {"LONG": "\U0001f4c8", "SHORT": "\U0001f4c9", "FLAT": "\u26aa"}.get(_dir, "\u26aa")
        _prob_str = f"{_prob:.4f}" if _prob is not None else "N/A"
        st.markdown(
            f'<div style="background:{_bg};border-left:5px solid {_bc};padding:8px 16px;'
            f'border-radius:6px;margin:4px 0;display:flex;align-items:center;gap:20px;">'
            f'<span style="font-size:18px;">{_dir_icon}</span>'
            f'<span style="color:{_bc};font-size:15px;font-weight:700;min-width:120px;">Model &nbsp;{_dir}</span>'
            f'<span style="color:#aaa;font-size:12px;">'
            f'prob = <b style="color:#eee;">{_prob_str}</b>'
            f' &nbsp;|&nbsp; LONG \u2265 {_lt:.2f} / SHORT \u2264 {_st:.2f}'
            f' &nbsp;|&nbsp; bar = <b style="color:#eee;">{_bar_ts}</b>'
            f' &nbsp;|&nbsp; updated <b style="color:#eee;">{_now_ts}</b>'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception as _pred_err:
        # Show error but DO NOT re-raise — re-raising kills the fragment timer
        st.warning(f"Prediction error (retrying in 60 s): {_pred_err}", icon="\u26a0\ufe0f")

_prediction_feed()

st.divider()

@st.fragment(run_every=1)
def _tick_feed():
    # ── 1. Drain real stream ticks into tick log ───────────────────────────
    _stream_status = is_stream_running()
    _stream_price  = None  # most recent stream tick this second (if any)
    if api:
        new = drain_tick_buffer()
        if new:
            for t in new:
                ts_str = t["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                _app["tick_log"].append(
                    f"TICK: {ts_str} BST  |  Bid: {t['bid']:.5f}"
                    f"  |  Ask: {t['ask']:.5f}  |  Mid: {t['price']:.5f}"
                )
            _app["tick_log"] = _app["tick_log"][-500:]
            _stream_price = new[-1]   # use latest tick for this second

    # ── 2. One price point per second, always timestamped with local NOW ───
    # Using local time for every entry means the chart is ALWAYS time-ordered
    # and never zigzags from OANDA stream timestamps being slightly in the past.
    # Stream tick price takes priority; REST fills in when stream is quiet.
    if api:
        _now_iso = datetime.now(BST_TZ).isoformat()
        if _stream_price:
            _app["price_history"].append({
                "time": _now_iso, "stream": True,
                "bid": _stream_price["bid"], "ask": _stream_price["ask"],
                "price": _stream_price["price"],
            })
        else:
            _rest = get_bid_ask_price(api, INSTRUMENT)
            if _rest:
                _app["price_history"].append({
                    "time": _now_iso, "stream": False,
                    "bid": _rest["bid"], "ask": _rest["ask"], "price": _rest["price"],
                })
        _app["price_history"] = _app["price_history"][-2000:]

    # ── 3. Freeze volatility regime at each new M15 candle boundary ───────
    # "Frozen Regime per Candle": compute once at candle open, hold until next.
    # Uses the last 15 minutes of collected price points (up to 900 entries
    # at 1 tick/s) so temporal resolution matches the model decision frequency.
    _now_bst = datetime.now(BST_TZ)
    _candle_bar = _now_bst.replace(
        minute=(_now_bst.minute // 15) * 15, second=0, microsecond=0
    ).strftime("%Y-%m-%d %H:%M")
    if _candle_bar != _app.get("last_candle_bar") and _app["price_history"]:
        _ph_for_vol = _app["price_history"][-900:]   # last ~15 min of ticks
        _vol_prices = pd.Series([p["price"] for p in _ph_for_vol])
        _new_regime = get_volatility_regime(_vol_prices)
        _new_regime["frozen_at"] = _now_bst.strftime("%H:%M")
        _app["frozen_regime"]   = _new_regime
        _app["last_candle_bar"] = _candle_bar


    if _app["price_history"]:
        _lt = _app["price_history"][-1]
        _lm = float(_lt["price"])
        _lb = float(_lt["bid"])
        _la = float(_lt["ask"])
        _sp = round((_la - _lb) * 10_000, 1)
        _is_stream = _lt.get("stream", False)
        _pm1, _pm2, _pm3 = st.columns(3)
        with _pm1:
            st.metric("\U0001f4b9 Mid Price", f"{_lm:.5f}")
            if _avg_px and _side != 0:
                st.caption(f"Entry: {_avg_px:.5f}  |  \u0394 {(_lm - _avg_px) * _side:+.5f}")
        with _pm2:
            st.metric("\U0001f7e2 Bid", f"{_lb:.5f}")
            _src_lbl = "\U0001f4e1 stream" if _is_stream else "\U0001f310 REST"
            st.caption(f"Spread: {_sp} pts  |  {_src_lbl}")
        with _pm3:
            st.metric("\U0001f7e3 Ask", f"{_la:.5f}")
    else:
        st.info("Connecting to OANDA pricing\u2026", icon="\U0001f4e1")

    # ── Live mid-price chart ───────────────────────────────────────────────
    st.subheader("\U0001f4ca Live Mid Price \u2014 EUR/USD")
    if _app["price_history"]:
        _ph = pd.DataFrame(_app["price_history"])
        _ph["time"] = pd.to_datetime(_ph["time"])
        _ph = _ph.sort_values("time").reset_index(drop=True)   # guarantee time-ordered
        # Live 15-minute interval (last tick in each bucket)
        _live_15 = (
            _ph.set_index("time")[["price"]]
               .resample("15min")
               .last()
               .dropna()
        )
        # Historical context: recent 1 day of completed M15 bars
        _hist_15 = pd.DataFrame()
        if api:
            _hist_15 = _get_recent_m15_history(api, INSTRUMENT, lookback, window)

        # Combine history + live; keep latest for overlaps so live wins current bucket.
        _parts = []
        if not _hist_15.empty:
            _parts.append(_hist_15)
        if not _live_15.empty:
            _parts.append(_live_15)
        if not _parts:
            st.info("Chart will appear once pricing loads…", icon="\U0001f4c8")
            return
        _plot_15 = pd.concat(_parts).sort_index()
        _plot_15 = _plot_15[~_plot_15.index.duplicated(keep="last")]
        _plot_15 = _plot_15.tail(110)

        _last_mid   = float(_plot_15["price"].iloc[-1])
        _y_min      = float(_plot_15["price"].min())
        _y_max      = float(_plot_15["price"].max())
        _pad        = max((_y_max - _y_min) * 0.3, 0.00005)  # at least 0.5 pip padding

        _fig = go.Figure()
        # Continuous price line using 15-minute interval points.
        _fig.add_trace(go.Scatter(
            x=_plot_15.index, y=_plot_15["price"],
            mode="lines",
            name="Mid (15m)",
            line=dict(color="#4DA6FF", width=1.8),
        ))
        _fig.add_hline(
            y=_last_mid,
            line_dash="dot",
            line_color="#FFD700",
            annotation_text=f"  {_last_mid:.5f}",
            annotation_position="right",
        )
        for _t in _app["trade_log"]:
            if not _t.get("price"):
                continue
            _act = _t.get("action", "")
            _clr = "#4CAF50" if "LONG" in _act else ("#F44336" if "SHORT" in _act else "#FFC107")
            _sym = "triangle-up" if "LONG" in _act else "triangle-down"
            _fig.add_trace(go.Scatter(
                x=[_t["time"]], y=[_t["price"]],
                mode="markers",
                marker=dict(symbol=_sym, size=14, color=_clr,
                            line=dict(width=1.5, color="white")),
                showlegend=False, name=_act,
            ))
        _fig.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=0, r=70, t=10, b=0),
            xaxis_title="Time (BST)",
            yaxis=dict(
                title="EUR/USD Mid",
                range=[_y_min - _pad, _y_max + _pad],
                tickformat=".5f",
            ),
            legend=dict(orientation="h", y=1.05, x=0),
            showlegend=True,
        )
        st.plotly_chart(_fig, use_container_width=True)
    else:
        st.info("Chart will appear once pricing loads\u2026", icon="\U0001f4c8")

    # ── Live tick log (scrollable) + Volatility Regime ────────────────────
    st.subheader("\U0001f4e1 Live Tick Feed")
    _c1, _c2 = st.columns([4, 3])
    with _c1:
        if _app["tick_log"]:
            _tick_html = escape("\n".join(reversed(_app["tick_log"][-200:]))).replace("\n", "<br>")
            st.markdown(
                f'<div class="tick-feed-box">'
                f'{_tick_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "Waiting for live ticks from OANDA stream\u2026",
                icon="\U0001f4e1",
            )
    with _c2:
        # ── Volatility Regime Detection (Frozen per M15 Candle) ───────────
        # Read the frozen regime stored by the M15 boundary check above.
        # Fall back to a live estimate while not yet frozen (early in session).
        _fr = _app.get("frozen_regime")
        if _fr is None:
            # No candle boundary crossed yet — compute live estimate as fallback
            _price_s = (
                pd.Series([p["price"] for p in _app["price_history"]])
                if _app["price_history"] else pd.Series([], dtype=float)
            )
            _vr = get_volatility_regime(_price_s)
            _freeze_note = "⏳ Awaiting first M15 boundary…"
            _is_frozen   = False
        else:
            _vr          = _fr
            _freeze_note = f"🔒 Frozen at {_fr.get('frozen_at', '—')}"
            _is_frozen   = True

        # Compute the threshold adjustment text
        _base_long_disp  = float(_app["active_params"]["long_threshold"]  if _app["auto_trade"] else _app["slider_params"]["long_threshold"])
        _base_short_disp = float(_app["active_params"]["short_threshold"] if _app["auto_trade"] else _app["slider_params"]["short_threshold"])
        _eff_l, _eff_s   = _effective_thresholds(_base_long_disp, _base_short_disp, _vr if _is_frozen else None)
        _adj_delta_long  = _eff_l - _base_long_disp
        _thr_note = (
            f"Long: {_base_long_disp:.2f} → <b>{_eff_l:.2f}</b> "
            f"({'↑' if _adj_delta_long > 0 else '↓' if _adj_delta_long < 0 else '='}"
            f"{abs(_adj_delta_long):.2f})<br>"
            f"Short: {_base_short_disp:.2f} → <b>{_eff_s:.2f}</b>"
        )

        if _vr["regime"] == "UNKNOWN":
            _vol_body = (
                '<span class="vol-regime-label" style="color:#888888;">⬜ Unknown</span>'
                '<span class="vol-regime-stat">Collecting price data…<br>'
                'Regime will appear once enough ticks arrive.</span>'
            )
        else:
            _pct_bar_fill = f"{_vr['percentile']:.0f}%"
            _vol_pips     = f"{_vr['vol'] * 10_000:.2f}" if _vr["vol"] else "—"
            _vol_body = (
                f'<span class="vol-regime-label" style="color:{_vr["color"]};">'
                f'{_vr["label"]}</span>'
                # percentile gauge bar
                f'<div style="width:100%;background:rgba(127,127,127,0.18);'
                f'border-radius:6px;height:10px;margin:4px 0;">'
                f'<div style="width:{_pct_bar_fill};height:10px;border-radius:6px;'
                f'background:{_vr["color"]};transition:width 0.5s;"></div></div>'
                f'<span class="vol-regime-stat">'
                f'Percentile: <b>{_vr["percentile"]:.0f}</b> / 100<br>'
                f'Rolling vol: <b>{_vol_pips}</b> pips (σ)<br>'
                f'{_thr_note}</span>'
                f'<span style="font-size:11px;opacity:0.55;color:var(--text-color);'
                f'margin-top:4px;">{_freeze_note}</span>'
            )

        st.markdown(
            f'<div class="vol-regime-box">'
            f'<div style="font-size:13px;font-weight:600;opacity:0.6;'
            f'margin-bottom:4px;color:var(--text-color);">⚡ Volatility Regime</div>'
            f'{_vol_body}'
            f'</div>',
            unsafe_allow_html=True,
        )


_tick_feed()

# ── Fragment: OANDA Trade History (auto-refreshes every 15s) ─────────────────
@st.fragment(run_every=15)
def _trade_history_feed():
    st.subheader("\U0001f4cb Trade History \u2014 OANDA Account (EUR/USD)")
    if not api:
        st.warning("OANDA not connected.")
        return
    _trades = get_recent_trades(api, INSTRUMENT, count=30)
    if _trades:
        _tdf = pd.DataFrame(_trades)
        # Reorder columns for readability
        _cols = ["id", "state", "side", "units", "open_price", "close_price",
                 "open_time", "close_time", "realised_pl", "unrealised_pl"]
        _cols = [c for c in _cols if c in _tdf.columns]
        st.dataframe(_tdf[_cols], use_container_width=True, height=360)
    else:
        st.info("No trades found for EUR/USD on this OANDA account.")

_trade_history_feed()

render_footer()
