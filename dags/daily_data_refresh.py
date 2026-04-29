"""
daily_data_refresh.py
=====================
Airflow DAG — runs Mon–Fri at 22:00 UTC (after London market close).

Data layout
-----------
  data/                        ← model-building pipeline (fetch_data.py → preprocess.py → train.py)
  data/backtest/               ← this DAG only; consumed by the Streamlit frontend
    forex_data_backtest.csv    ← single append-only file; date range selectable in the UI

Single task
-----------
fetch_new_bars — pull today's M15 bars from OANDA via count-based API,
                 append to data/backtest/forex_data_backtest.csv, dedup, sort.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Resolve project root (two levels above dags/) ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── File paths (separate from model-pipeline data/) ──────────────────────────
# Single append-only CSV — no train/test split needed here.
BACKTEST_CSV = ROOT / "data" / "backtest" / "forex_data_backtest.csv"
OANDA_CFG    = ROOT / "src" / "oanda.cfg"

# ── How many M15 bars to fetch per run (1 trading day = 96 bars; fetch 2× for safety) ──
BARS_PER_RUN = 200


# ══════════════════════════════════════════════════════════════════════════════
# Single task — fetch new bars and append to BACKTEST_CSV
# ══════════════════════════════════════════════════════════════════════════════
def fetch_new_bars():
    """
    Fetch the latest BARS_PER_RUN M15 candles from OANDA using the
    count-based endpoint (avoids the 400 'Time is in the future' bug
    that affects tpqoa.get_history()).

    New rows are appended to BACKTEST_CSV, deduplicated on the time
    index, and sorted ascending. The Streamlit backtest page reads this
    single file directly — no train/test split needed here.
    """
    import tpqoa  # imported here so Airflow workers can skip it if not installed

    api = tpqoa.tpqoa(str(OANDA_CFG))

    # ── Fetch via count-based candles endpoint ────────────────────────────────
    def _fetch_candles(price_type: str) -> pd.Series:
        resp = api.ctx.instrument.candles(
            "EUR_USD",
            count=BARS_PER_RUN,
            granularity="M15",
            price=price_type,
        )
        records = {}
        for c in resp.body["candles"]:
            if not c.complete:
                continue
            ts = pd.Timestamp(c.time).tz_localize(None)  # UTC naive, matches existing data
            candle = getattr(c, price_type.lower())
            records[ts] = float(candle.c)
        return pd.Series(records, name=price_type)

    mid = _fetch_candles("M")
    bid = _fetch_candles("B")
    ask = _fetch_candles("A")

    new_df = pd.DataFrame({
        "price":  mid,
        "spread": ask - bid,
    }).dropna()
    new_df.index.name = "time"

    if new_df.empty:
        print("No complete candles returned — nothing to append.")
        return

    # ── Load existing CSV and append ─────────────────────────────────────────
    BACKTEST_CSV.parent.mkdir(parents=True, exist_ok=True)

    if BACKTEST_CSV.exists():
        existing = pd.read_csv(BACKTEST_CSV, parse_dates=["time"], index_col="time")
    else:
        existing = pd.DataFrame(columns=["price", "spread"])
        existing.index.name = "time"

    combined = (
        pd.concat([existing, new_df])
        .sort_index()
    )
    combined = combined[~combined.index.duplicated(keep="last")]

    combined.to_csv(BACKTEST_CSV)
    added = len(combined) - len(existing)
    print(
        f"fetch_new_bars: fetched {len(new_df)} bars, "
        f"added {added} new rows → {len(combined)} total in {BACKTEST_CSV}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# DAG definition
# ══════════════════════════════════════════════════════════════════════════════
default_args = {
    "owner":            "dissertation",
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry":   False,
}

with DAG(
    dag_id="daily_data_refresh",
    description="Fetch fresh EUR/USD M15 bars from OANDA and append to backtest CSV",
    schedule_interval="0 22 * * 1-5",   # 22:00 UTC, Mon–Fri only
    start_date=datetime(2026, 4, 29),
    catchup=False,
    default_args=default_args,
    tags=["forex", "data", "oanda"],
) as dag:

    PythonOperator(
        task_id="fetch_new_bars",
        python_callable=fetch_new_bars,
    )
