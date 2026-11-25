# EUR/USD LSTM Algorithmic Trading & Backtesting System

An end-to-end algorithmic FX trading prototype and interactive Streamlit web application built on **EUR/USD M15** candles, combining a **2-layer LSTM** directional classifier, a mark-to-market historical backtesting engine, live **OANDA v20 Practice API** execution/streaming, and an **Apache Airflow (Astro CLI)** MLOps retraining pipeline.

**Live Demo:** [https://forex-algotrading.streamlit.app/](https://forex-algotrading.streamlit.app/)

> **Project Scope Note:** This repository is part of an ongoing MSc dissertation exploring multimodal FX modelling. The **deployed web application currently runs Phase 1 — the unimodal price-action LSTM model** on 7 technical features. Experimental modules (`src/DNNModel.py`, `src/news.py`) are maintained separately and are **not** fused into the live trading signal yet.

---

## Architecture Overview

| Layer | Implementation |
|---|---|
| **Web App (Frontend)** | Streamlit multi-page app (`frontend/app.py`, `1_Live_Trading.py`, `2_Backtest.py`) with Plotly charts |
| **Model** | 2-layer TensorFlow/Keras LSTM binary directional classifier (`models/lstm_model.h5`, 50-bar lookback, 7 technical features) |
| **Features (7)** | `returns`, `sma` (20 vs 150), `boll` (20-bar z-score), `min`, `max`, `mom` (3-bar), `vol` (20-bar) |
| **Volatility Regime** | M15 20-bar rolling log-return std percentile (200-bar lookback) classified as Low / Normal / High, symmetrically adjusting live signal thresholds |
| **Backtest Engine** | Batched LSTM inference (`frontend/utils/trading_engine.py`) with mark-to-market equity, per-bar bid-ask spread on all closes, SL/TP exits, and out-of-sample guardrails (`2026-01-09` onward) |
| **Broker Integration** | OANDA v20 Practice (Demo) API via `tpqoa` (live tick streaming + REST fallback) |
| **Local MLOps Stack** | Apache Airflow (Astronomer / Astro CLI) DAGs in `dags/` for daily candle ingestion (`daily_data_refresh.py`) and weekly grid-search + 4-window walk-forward retraining (`retrain_lstm.py`), plus MLflow tracking (`src/train_lstm.py`, `src/evaluate.py`) |

---

## Repository Layout

```text
├── frontend/
│   ├── app.py                     # Streamlit entry point / system overview
│   ├── pages/
│   │   ├── 1_Live_Trading.py      # OANDA live streaming, M15 chart, regime & demo bot
│   │   └── 2_Backtest.py          # Interactive historical backtester + JSON export
│   └── utils/
│       ├── trading_engine.py      # Model loading, feature gen, OANDA helpers, BacktestEngine
│       └── ui.py                  # Shared styling, FCA disclaimer, footer
├── dags/
│   ├── daily_data_refresh.py      # Airflow DAG: weekday OANDA M15 candle append
│   └── retrain_lstm.py            # Airflow DAG: weekly grid search + walk-forward quality gate
├── src/
│   ├── train_lstm.py              # Standalone LSTM training + MLflow logging
│   ├── evaluate.py                # Out-of-sample evaluation & iterative backtest
│   ├── fetch_data.py              # Historical OANDA data download
│   ├── preprocess.py              # Train/test temporal split (last 4 months held out)
│   ├── DNNModel.py                # Experimental DNN architecture (not deployed)
│   └── news.py                    # Experimental news sentiment prototype (not deployed)
├── models/
│   ├── lstm_model.h5              # Trained LSTM weights
│   └── lstm_feature_stats.pkl     # Training-split normalization mean/std & metadata
└── data/
    ├── processed/train/           # Training split (2024-04-24 -> 2026-01-08)
    ├── processed/test/            # Held-out test split (2026-01-08 -> 2026-05-08)
    └── backtest/                  # Combined backtest CSV (2024-04-24 -> 2026-05-08)
```

---

## Running Locally

### 1. Streamlit Web App
```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

For OANDA live features, provide either `src/oanda.cfg` locally or `.streamlit/secrets.toml`:
```toml
[oanda]
account_id = "YOUR_PRACTICE_ACCOUNT_ID"
access_token = "YOUR_PRACTICE_ACCESS_TOKEN"
account_type = "practice"

# Optional: lock live Bot ON/OFF and Close All Positions buttons behind a PIN
[admin]
pin = "YOUR_PIN"
```

### 2. Local Airflow MLOps Stack (Optional)
```bash
astro dev start
```
Airflow UI runs at `http://localhost:8080` and schedules `daily_data_refresh` and `retrain_lstm` against the mounted local `data/` and `models/` directories.

---

## Disclaimer

Educational research and demonstration prototype only. Nothing in this repository or web application constitutes investment advice, a personal recommendation, or a financial promotion. CFDs and FX are high-risk instruments.
