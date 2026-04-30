from zoneinfo import ZoneInfo

import streamlit as st

BST_TZ = ZoneInfo("Europe/London")
GITHUB_URL = "https://github.com/Mohit-Bhoir/Multimodal-Trihybrid-Algorithmic-Forex-Trading-Model-"


def inject_page_chrome() -> None:
    """Inject shared page chrome that works in both light and dark themes."""
    st.markdown(
        """
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
.fca-banner {
    margin: 0 0 14px 0;
    padding: 10px 14px;
    border-radius: 14px;
    border: 1px solid rgba(224, 166, 51, 0.28);
    border-left: 4px solid #f2b44a;
    background: linear-gradient(90deg, rgba(242, 180, 74, 0.14), rgba(242, 180, 74, 0.07));
    color: var(--text-color);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    width: 100%;
    box-sizing: border-box;
}
.fca-banner__title {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #d58d1f;
    margin-bottom: 2px;
}
.fca-banner__text {
    font-size: 0.93rem;
    line-height: 1.45;
    color: var(--text-color);
}
.app-footer {
    margin-top: 28px;
    padding: 16px 18px;
    border-radius: 14px;
    border: 1px solid rgba(127, 127, 127, 0.22);
    background: var(--secondary-background-color);
    color: var(--text-color);
    text-align: center;
    line-height: 1.55;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.06);
    width: 100%;
    box-sizing: border-box;
}
.app-footer__title {
    color: var(--text-color);
    font-weight: 700;
    font-size: 0.98rem;
}
.app-footer__meta {
    font-size: 0.9rem;
    color: var(--text-color);
    opacity: 0.75;
}
.app-footer a {
    color: var(--primary-color);
    font-weight: 600;
    text-decoration: none;
}
.app-footer a:hover {
    opacity: 0.88;
    text-decoration: underline;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        """
<div class="fca-banner">
  <div class="fca-banner__title">FCA Disclaimer</div>
  <div class="fca-banner__text">
    Educational demo only. Nothing on this app is investment advice, a personal recommendation, or a financial promotion.<br>
        CFDs and FX are high risk; use only in line with FCA rules and your own legal obligations.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        f"""
<div class="app-footer">
  <div class="app-footer__title">EUR/USD LSTM Algorithmic Trading System · Demo Build by Mohit Bhoir</div>
  <div class="app-footer__meta">Data Sources: OANDA v20 Pricing + Historical EUR/USD Backtest Dataset</div>
  <div>View the code on <a href="{GITHUB_URL}" target="_blank">GitHub</a></div>
</div>
""",
        unsafe_allow_html=True,
    )
