"""
retrain_lstm.py
===============
Airflow DAG — weekly LSTM retraining pipeline (Astronomer / Astro CLI).
Schedule: Sunday 02:00 UTC (before Monday London market open).

Dependency isolation
--------------------
  tensorflow==2.21.0 requires protobuf>=6.31.1.
  Airflow''s opentelemetry-proto (baked into Astro runtime 12) requires protobuf<5.
  These cannot coexist in the same pip environment.

  Solution: the two TF-heavy tasks (tune_and_train, iterative_backtest) run
  inside an isolated PythonVirtualenvOperator subprocess.  TF is installed
  ONLY inside that subprocess venv and never touches Airflow''s own packages.
  The venv is cached between runs (venv_cache_path) so it is only built once.

Pipeline
--------
  fetch_training_data  ->  preprocess_splits  ->  tune_and_train (venv)
                                                        |
                                            iterative_backtest (venv)
                                                        |
                                            archive_best_model

Model archive layout
--------------------
  models/
    lstm_model.h5              <- live model   (overwritten on successful retrain)
    lstm_feature_stats.pkl     <- live stats   (overwritten on successful retrain)
    archive/
      YYYY-MM-DD/
        lstm_model.h5
        lstm_feature_stats.pkl
        metrics.json           <- walk-forward results + quality gate outcome
        best_params.json       <- winning hyperparameter configuration

Quality gate (all must pass for promotion)
------------------------------------------
    aggregate Sharpe   >=  0.3
    worst window DD    >= -25 %
    total closed trades >= 20
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator

# -- Project root ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# -- File paths (passed as plain strings to venv tasks) -------------------------
BACKTEST_CSV  = ROOT / "data" / "backtest" / "forex_data_backtest.csv"
TRAIN_CSV     = ROOT / "data" / "processed" / "train" / "forex_data_processed_train.csv"
TEST_CSV      = ROOT / "data" / "processed" / "test"  / "forex_data_processed_test.csv"
STAGING_DIR   = ROOT / "data"   / "retrain_staging"
LIVE_MODEL    = ROOT / "models" / "lstm_model.h5"
LIVE_STATS    = ROOT / "models" / "lstm_feature_stats.pkl"
ARCHIVE_ROOT  = ROOT / "models" / "archive"
OANDA_CFG     = ROOT / "src"    / "oanda.cfg"
VENV_CACHE    = ROOT / "include" / ".venv_cache"

# -- Venv requirements for TF tasks --------------------------------------------
_TF_REQUIREMENTS = [
    "tensorflow==2.21.0",
    "scikit-learn",
    "numpy<2",
    "pandas",
]

# -- Constants shared across tasks ---------------------------------------------
LOOKBACK          = 50
WINDOW            = 20
VALIDATION_MONTHS = 4
TEST_MONTHS       = 4
PATIENCE          = 10
SEED              = 100
WF_WINDOWS        = 4
FEATURE_COLS      = ["returns", "sma", "boll", "min", "max", "mom", "vol"]
BARS_PER_FETCH    = 5_000
OANDA_STALE_DAYS  = 7
MIN_SHARPE        =  0.3
MAX_DRAWDOWN_PCT  = -25.0
MIN_TRADE_COUNT   =  20
PARAM_GRID = {
    "units_1":       [32, 64],
    "units_2":       [16, 32],
    "dropout":       [0.1, 0.2],
    "batch_size":    [32, 64],
    "epochs":        [10],
    "learning_rate": [0.001, 0.0005],
}


# ==============================================================================
# Task 1 -- fetch_training_data  (plain PythonOperator -- no TF needed)
# ==============================================================================

def fetch_training_data(**context):
    """
    Ensure the backtest CSV is up-to-date.
    Skips OANDA if the file was updated within OANDA_STALE_DAYS days.
    Pushes XCom: source_rows (int).
    """
    import pandas as pd
    from datetime import datetime as _dt

    backtest_csv = Path(context["params"]["backtest_csv"])
    oanda_cfg    = Path(context["params"]["oanda_cfg"])
    bars         = int(context["params"]["bars_per_fetch"])
    stale_days   = int(context["params"]["oanda_stale_days"])

    if backtest_csv.exists():
        age_days = (_dt.now() - _dt.fromtimestamp(backtest_csv.stat().st_mtime)).days
        if age_days <= stale_days:
            df = pd.read_csv(backtest_csv, parse_dates=["time"], index_col="time")
            print(
                f"[fetch_training_data] CSV fresh ({age_days}d old) "
                f"-- {len(df)} rows, skipping OANDA."
            )
            context["ti"].xcom_push(key="source_rows", value=len(df))
            return

    print("[fetch_training_data] CSV stale or missing -- fetching from OANDA ...")
    import tpqoa

    api = tpqoa.tpqoa(str(oanda_cfg))

    def _fetch(price_type):
        _attr = {"M": "mid", "B": "bid", "A": "ask"}[price_type]
        resp  = api.ctx.instrument.candles(
            "EUR_USD", count=bars, granularity="M15", price=price_type,
        )
        rec = {}
        for c in resp.body["candles"]:
            if not c.complete:
                continue
            ts = pd.Timestamp(c.time).tz_localize(None)
            rec[ts] = float(getattr(c, _attr).c)
        return pd.Series(rec, name=price_type)

    mid, bid, ask = _fetch("M"), _fetch("B"), _fetch("A")
    new_df = pd.DataFrame({"price": mid, "spread": ask - bid}).dropna()
    new_df.index.name = "time"

    if new_df.empty:
        raise ValueError("OANDA returned no complete candles.")

    backtest_csv.parent.mkdir(parents=True, exist_ok=True)
    if backtest_csv.exists():
        existing = pd.read_csv(backtest_csv, parse_dates=["time"], index_col="time")
        combined = pd.concat([existing, new_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new_df.sort_index()

    combined.to_csv(backtest_csv)
    print(f"[fetch_training_data] Saved {len(combined)} rows -> {backtest_csv}")
    context["ti"].xcom_push(key="source_rows", value=len(combined))


# ==============================================================================
# Task 2 -- preprocess_splits  (plain PythonOperator -- pandas only)
# ==============================================================================

def preprocess_splits(**context):
    """
    Split backtest CSV -> train / test partitions.
    Pushes XCom: train_rows, test_rows.
    """
    import pandas as pd

    backtest_csv = Path(context["params"]["backtest_csv"])
    train_csv    = Path(context["params"]["train_csv"])
    test_csv     = Path(context["params"]["test_csv"])
    test_months  = int(context["params"]["test_months"])

    if not backtest_csv.exists():
        raise FileNotFoundError(f"Training source CSV not found: {backtest_csv}")

    df = pd.read_csv(backtest_csv, parse_dates=["time"], index_col="time")

    if len(df) < 2_000:
        raise ValueError(f"Insufficient data: {len(df)} rows (need >= 2,000).")

    test_start = df.index[-1] - pd.DateOffset(months=test_months)
    train_df   = df[df.index < test_start].copy()
    test_df    = df[df.index >= test_start].copy()

    if len(train_df) < 1_000:
        raise ValueError(f"Train split too small: {len(train_df)} rows.")
    if len(test_df)  < 200:
        raise ValueError(f"Test split too small: {len(test_df)} rows.")

    train_csv.parent.mkdir(parents=True, exist_ok=True)
    test_csv.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_csv)
    test_df.to_csv(test_csv)

    print(
        f"[preprocess_splits] Train: {len(train_df)} rows  "
        f"({train_df.index[0].date()} -> {train_df.index[-1].date()})\n"
        f"[preprocess_splits] Test:  {len(test_df)} rows  "
        f"({test_df.index[0].date()} -> {test_df.index[-1].date()})"
    )
    context["ti"].xcom_push(key="train_rows", value=len(train_df))
    context["ti"].xcom_push(key="test_rows",  value=len(test_df))


# ==============================================================================
# Task 3 -- tune_and_train  (PythonVirtualenvOperator -- TF isolated)
# ==============================================================================
# FULLY SELF-CONTAINED: no references to DAG-file module-level symbols.
# All imports inside function body.  Receives config via op_kwargs.
# Returns JSON-serialisable dict -> stored in XCom as return_value.
# ------------------------------------------------------------------------------

def tune_and_train_venv(
    train_csv,
    staging_dir,
    lookback,
    window,
    validation_months,
    patience,
    seed,
    param_grid,
    feature_cols,
):
    """
    Hyperparameter grid search + final model training inside isolated TF venv.
    Returns dict: {staging_model, staging_stats, best_config, val_metrics}.
    """
    import pickle
    import random
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from pathlib import Path
    from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    FEATURE_COLS = feature_cols
    staging_path  = Path(staging_dir)
    staging_path.mkdir(parents=True, exist_ok=True)
    staging_model = str(staging_path / "lstm_model.h5")
    staging_stats = str(staging_path / "lstm_feature_stats.pkl")

    # -- Helpers ---------------------------------------------------------------
    def set_seeds(s):
        random.seed(s)
        np.random.seed(s)
        tf.random.set_seed(s)

    def generate_features(data, w=20):
        data = data.copy()
        data["returns"]   = np.log(data["price"] / data["price"].shift(1))
        data["direction"] = np.where(data["returns"] > 0, 1, 0)
        data["sma"]       = data["price"].rolling(w).mean() - data["price"].rolling(150).mean()
        data["boll"]      = (
            (data["price"] - data["price"].rolling(w).mean())
            / data["price"].rolling(w).std()
        )
        data["min"]  = data["price"].rolling(w).min() / data["price"] - 1
        data["max"]  = data["price"].rolling(w).max() / data["price"] - 1
        data["mom"]  = data["returns"].rolling(3).mean()
        data["vol"]  = data["returns"].rolling(w).std()
        data.dropna(inplace=True)
        return data

    def compute_class_weights(labels):
        counts = np.bincount(labels.astype(int), minlength=2)
        total  = counts.sum()
        return {i: total / (2 * c) if c else 1.0 for i, c in enumerate(counts)}

    def build_model(config, input_shape):
        m = Sequential([
            Input(shape=input_shape),
            LSTM(config["units_1"], return_sequences=True),
            Dropout(config["dropout"]),
            LSTM(config["units_2"], return_sequences=False),
            Dropout(config["dropout"]),
            Dense(1, activation="sigmoid"),
        ])
        m.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=config.get("learning_rate", 0.001)),
            metrics=["accuracy"],
        )
        return m

    def fit_config(X_tr, y_tr, X_vl, y_vl, config, pat, s):
        set_seeds(s)
        tf.keras.backend.clear_session()
        model = build_model(config, (X_tr.shape[1], X_tr.shape[2]))
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_vl, y_vl),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            shuffle=False,
            verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=pat,
                                     restore_best_weights=True)],
            class_weight=compute_class_weights(y_tr),
        )
        val_loss, val_acc = model.evaluate(X_vl, y_vl, verbose=0)
        best_ep = int(np.argmin(history.history["val_loss"])) + 1
        return model, float(val_loss), float(val_acc), best_ep

    # -- Load + feature engineering --------------------------------------------
    data = pd.read_csv(train_csv, parse_dates=["time"], index_col="time")
    data = generate_features(data, w=window)

    val_start  = data.index[-1] - pd.DateOffset(months=validation_months)
    train_mask = data.index < val_start
    mean = data.loc[train_mask, FEATURE_COLS].mean()
    std  = data.loc[train_mask, FEATURE_COLS].std().replace(0, 1)

    with open(staging_stats, "wb") as fh:
        pickle.dump({
            "mean": mean, "std": std,
            "feature_cols": FEATURE_COLS,
            "lookback": lookback, "window": window,
        }, fh)

    data[FEATURE_COLS] = (data[FEATURE_COLS] - mean) / std
    features   = data[FEATURE_COLS].values
    targets    = data["direction"].values
    dates      = data.index

    X_all, y_all, seq_dates = [], [], []
    for i in range(lookback, len(features)):
        X_all.append(features[i - lookback:i])
        y_all.append(targets[i])
        seq_dates.append(dates[i])

    X_all     = np.array(X_all, dtype=np.float32)
    y_all     = np.array(y_all, dtype=np.float32)
    seq_dates = np.array(seq_dates)

    train_idx = seq_dates < val_start
    val_idx   = seq_dates >= val_start
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val,   y_val   = X_all[val_idx],   y_all[val_idx]

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Train or val split is empty -- reduce validation_months.")
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        raise ValueError("Both splits must contain both target classes.")

    print(f"[tune_and_train] X_train={X_train.shape}  X_val={X_val.shape}")

    # -- Hyperparameter search -------------------------------------------------
    search_space = list(ParameterGrid(param_grid))
    n_splits     = min(5, len(X_train) - 1)
    if n_splits < 2:
        raise ValueError("Not enough samples for TimeSeriesCV.")

    splitter    = TimeSeriesSplit(n_splits=n_splits)
    best_result = None
    total       = len(search_space)

    for idx, config in enumerate(search_space, 1):
        print(f"[tune_and_train] Config {idx}/{total}: {config}")
        fold_losses, fold_accs, fold_eps = [], [], []

        for tr_idx, vl_idx in splitter.split(X_train):
            Xft, yft = X_train[tr_idx], y_train[tr_idx]
            Xfv, yfv = X_train[vl_idx], y_train[vl_idx]
            if len(np.unique(yft)) < 2 or len(np.unique(yfv)) < 2:
                continue
            _, vl, va, ep = fit_config(Xft, yft, Xfv, yfv, config, patience, seed)
            fold_losses.append(vl)
            fold_accs.append(va)
            fold_eps.append(ep)

        if not fold_accs:
            continue

        result = {
            **config,
            "cv_accuracy":   float(np.mean(fold_accs)),
            "cv_loss":       float(np.mean(fold_losses)),
            "cv_best_epoch": int(round(np.mean(fold_eps))),
        }
        print(f"  => CV acc={result['cv_accuracy']:.4f}  loss={result['cv_loss']:.4f}")

        if best_result is None:
            best_result = result
        elif (
            result["cv_accuracy"] > best_result["cv_accuracy"]
            or (
                abs(result["cv_accuracy"] - best_result["cv_accuracy"]) < 1e-6
                and result["cv_loss"] < best_result["cv_loss"]
            )
        ):
            best_result = result

    if best_result is None:
        raise ValueError("Hyperparameter search failed: no valid CV folds.")

    print(f"[tune_and_train] Best config: {best_result}")

    # -- Final retrain on full training pool -----------------------------------
    final_config = {
        k: v for k, v in best_result.items()
        if k not in {"cv_accuracy", "cv_loss", "cv_best_epoch"}
    }
    final_config["epochs"] = max(
        int(final_config.get("epochs", 10)),
        int(best_result.get("cv_best_epoch", 10)),
    )

    final_model, val_loss, val_acc, best_ep = fit_config(
        X_train, y_train, X_val, y_val, final_config, patience, seed
    )
    final_model.save(staging_model)

    print(
        f"[tune_and_train] Saved -> {staging_model}\n"
        f"[tune_and_train] val_acc={val_acc:.4f}  loss={val_loss:.4f}  best_epoch={best_ep}"
    )

    return {
        "staging_model": staging_model,
        "staging_stats": staging_stats,
        "best_config":   final_config,
        "val_metrics": {
            "val_accuracy": round(float(val_acc), 4),
            "val_loss":     round(float(val_loss), 4),
            "best_epoch":   int(best_ep),
        },
    }


# ==============================================================================
# Task 4 -- iterative_backtest  (PythonVirtualenvOperator -- TF isolated)
# FULLY SELF-CONTAINED -- no module-level references.
# ==============================================================================

def iterative_backtest_venv(
    train_result,
    train_csv,
    test_csv,
    wf_windows,
    min_sharpe,
    max_drawdown_pct,
    min_trade_count,
):
    """
    Walk-forward (iterative) backtest on the held-out test set.
    Returns dict: {passes, metrics, window_metrics}.
    """
    import pickle
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from pathlib import Path

    staging_model = train_result["staging_model"]
    staging_stats = train_result["staging_stats"]

    model = tf.keras.models.load_model(staging_model, compile=False)
    with open(staging_stats, "rb") as fh:
        stats = pickle.load(fh)

    mean         = stats["mean"]
    std          = stats["std"].replace(0, 1)
    feature_cols = list(stats["feature_cols"])
    lookback     = int(stats["lookback"])
    window       = int(stats["window"])

    # -- Feature engineering (inline) -----------------------------------------
    def gen_features(df, w):
        df = df.copy()
        df["returns"]   = np.log(df["price"] / df["price"].shift(1))
        df["direction"] = np.where(df["returns"] > 0, 1, 0)
        df["sma"]       = df["price"].rolling(w).mean() - df["price"].rolling(150).mean()
        df["boll"]      = (
            (df["price"] - df["price"].rolling(w).mean())
            / df["price"].rolling(w).std()
        )
        df["min"]  = df["price"].rolling(w).min() / df["price"] - 1
        df["max"]  = df["price"].rolling(w).max() / df["price"] - 1
        df["mom"]  = df["returns"].rolling(3).mean()
        df["vol"]  = df["returns"].rolling(w).std()
        df.dropna(inplace=True)
        return df

    # -- Single window simulation ----------------------------------------------
    def run_window(raw_data):
        df = gen_features(raw_data[["price"]].copy(), window)
        min_bars = lookback + 200
        if len(df) < min_bars:
            return {"error": f"Need >={min_bars} bars, got {len(df)}"}

        df_norm = df.copy()
        df_norm[feature_cols] = (df[feature_cols] - mean) / std

        n         = len(df)
        sequences = np.stack([
            df_norm[feature_cols].iloc[i - lookback:i].to_numpy(dtype=np.float32)
            for i in range(lookback, n)
        ])
        all_probs = model.predict(sequences, verbose=0, batch_size=512).reshape(-1)

        nav = 100_000.0; position = 0; entry_px = None
        equity = []; trades = []
        SL, TP, LT, ST = 0.005, 0.010, 0.55, 0.45

        for idx, i in enumerate(range(lookback, n)):
            price = float(df["price"].iloc[i])

            if position != 0 and entry_px is not None:
                ret = (price - entry_px) / entry_px * position
                if ret <= -SL:
                    pnl = nav * (-SL); nav += pnl
                    trades.append({"pnl": pnl, "action": "SL_CLOSE"})
                    position = 0; entry_px = None
                    equity.append(nav); continue
                elif ret >= TP:
                    pnl = nav * TP; nav += pnl
                    trades.append({"pnl": pnl, "action": "TP_CLOSE"})
                    position = 0; entry_px = None
                    equity.append(nav); continue

            prob    = float(all_probs[idx])
            new_pos = 1 if prob >= LT else (-1 if prob <= ST else 0)

            if new_pos != position:
                if position != 0 and entry_px is not None:
                    pnl = nav * (price - entry_px) / entry_px * position
                    nav += pnl
                    trades.append({"pnl": pnl, "action": "CLOSE"})
                    position = 0; entry_px = None
                if new_pos != 0:
                    position = new_pos; entry_px = price

            equity.append(nav)

        if position != 0 and entry_px is not None:
            last_px = float(df["price"].iloc[-1])
            pnl = nav * (last_px - entry_px) / entry_px * position
            nav += pnl
            trades.append({"pnl": pnl, "action": "END_CLOSE"})
            equity.append(nav)

        if len(equity) < 2:
            return {"error": "No equity data"}

        eq      = np.asarray(equity, dtype=float)
        br      = np.diff(eq) / eq[:-1]
        sharpe  = (float(np.mean(br) / np.std(br)) * (252 * 96) ** 0.5
                   if np.std(br) > 0 else 0.0)
        cum_max = np.maximum.accumulate(eq)
        max_dd  = float(np.min((eq - cum_max) / cum_max * 100))
        tot_ret = (eq[-1] - eq[0]) / eq[0] * 100
        closed  = [t for t in trades if t.get("pnl", 0.0) != 0.0]
        tc      = len(closed)
        wins    = sum(1 for t in closed if t["pnl"] > 0)
        wr      = wins / tc * 100 if tc else 0.0

        return dict(
            sharpe           = round(float(sharpe),   3),
            max_drawdown_pct = round(float(max_dd),   2),
            total_return_pct = round(float(tot_ret),  2),
            trade_count      = tc,
            win_rate_pct     = round(float(wr),       2),
            final_nav        = round(float(eq[-1]),   2),
        )

    # -- Walk-forward ----------------------------------------------------------
    test_df  = pd.read_csv(test_csv,  parse_dates=["time"], index_col="time")
    train_df = pd.read_csv(train_csv, parse_dates=["time"], index_col="time")

    warmup_needed   = 200 + window + lookback
    warmup_tail     = train_df.tail(warmup_needed)
    rows_per_window = max(len(test_df) // wf_windows, warmup_needed)
    window_results  = []

    print(
        f"[iterative_backtest] Test: {len(test_df)} rows  "
        f"({test_df.index[0].date()} -> {test_df.index[-1].date()})"
    )

    for w in range(wf_windows):
        w_start = w * rows_per_window
        w_end   = (w + 1) * rows_per_window if w < wf_windows - 1 else len(test_df)
        w_slice = test_df.iloc[w_start:w_end]

        if len(w_slice) < 100:
            print(f"[iterative_backtest] Window {w+1}: too few rows -- skipping.")
            continue

        result = run_window(pd.concat([warmup_tail, w_slice]))
        if "error" in result:
            print(f"[iterative_backtest] Window {w+1}: {result['error']}")
            continue

        result.update({
            "window": w + 1,
            "start":  str(w_slice.index[0].date()),
            "end":    str(w_slice.index[-1].date()),
        })
        window_results.append(result)
        print(
            f"[iterative_backtest] Window {w+1}"
            f"  ({result['start']} -> {result['end']}):"
            f"  Sharpe={result['sharpe']:.3f}"
            f"  Ret={result['total_return_pct']:.2f}%"
            f"  MaxDD={result['max_drawdown_pct']:.2f}%"
            f"  Trades={result['trade_count']}"
        )

    if not window_results:
        return {
            "passes":         False,
            "metrics":        {"error": "No valid walk-forward windows."},
            "window_metrics": [],
        }

    avg_sharpe   = float(np.mean([r["sharpe"]           for r in window_results]))
    avg_return   = float(np.mean([r["total_return_pct"] for r in window_results]))
    worst_dd     = float(np.min( [r["max_drawdown_pct"] for r in window_results]))
    total_trades = int(  sum(    [r["trade_count"]      for r in window_results]))
    avg_win_rate = float(np.mean([r["win_rate_pct"]     for r in window_results]))

    aggregate = dict(
        avg_sharpe         = round(avg_sharpe,   3),
        avg_return_pct     = round(avg_return,   2),
        worst_drawdown_pct = round(worst_dd,     2),
        total_trades       = total_trades,
        avg_win_rate_pct   = round(avg_win_rate, 2),
        n_windows          = len(window_results),
    )
    passes = (
        avg_sharpe   >= min_sharpe       and
        worst_dd     >= max_drawdown_pct and
        total_trades >= min_trade_count
    )

    print(f"\n[iterative_backtest] Aggregate: {aggregate}")
    print(
        f"[iterative_backtest] Gate -- "
        f"Sharpe {avg_sharpe:.3f}>={min_sharpe}: {avg_sharpe>=min_sharpe}  |  "
        f"MaxDD {worst_dd:.2f}%>={max_drawdown_pct}%: {worst_dd>=max_drawdown_pct}  |  "
        f"Trades {total_trades}>={min_trade_count}: {total_trades>=min_trade_count}"
    )
    print(f"[iterative_backtest] PASSES: {passes}")

    return {"passes": passes, "metrics": aggregate, "window_metrics": window_results}


# ==============================================================================
# Task 5 -- archive_best_model  (plain PythonOperator)
# ==============================================================================

def archive_best_model(**context):
    """
    Pull XCom from venv tasks, apply quality gate, conditionally promote model.
    """
    ti = context["ti"]

    train_result  = ti.xcom_pull(task_ids="tune_and_train")
    bt_result     = ti.xcom_pull(task_ids="iterative_backtest")

    passes         = bt_result.get("passes", False)
    metrics        = bt_result.get("metrics", {})
    window_metrics = bt_result.get("window_metrics", [])
    best_config    = train_result.get("best_config", {})
    val_metrics    = train_result.get("val_metrics", {})
    staging_model  = train_result["staging_model"]
    staging_stats  = train_result["staging_stats"]

    run_date    = context["ds"]
    archive_dir = Path(context["params"]["archive_root"]) / run_date
    archive_dir.mkdir(parents=True, exist_ok=True)

    live_model = Path(context["params"]["live_model"])
    live_stats = Path(context["params"]["live_stats"])

    _min_sharpe       = float(context["params"]["min_sharpe"])
    _max_drawdown_pct = float(context["params"]["max_drawdown_pct"])
    _min_trade_count  = int(context["params"]["min_trade_count"])

    # Write archive regardless of gate outcome
    shutil.copy2(staging_model, archive_dir / "lstm_model.h5")
    shutil.copy2(staging_stats, archive_dir / "lstm_feature_stats.pkl")
    (archive_dir / "metrics.json").write_text(json.dumps({
        "promoted":           passes,
        "backtest_aggregate": metrics,
        "window_metrics":     window_metrics,
        "val_metrics":        val_metrics,
        "quality_gate": {
            "min_sharpe":       _min_sharpe,
            "max_drawdown_pct": _max_drawdown_pct,
            "min_trade_count":  _min_trade_count,
        },
    }, indent=2, default=str))
    (archive_dir / "best_params.json").write_text(json.dumps({
        "model_config": best_config,
        "lookback":     context["params"]["lookback"],
        "window":       context["params"]["window"],
        "feature_cols": ["returns", "sma", "boll", "min", "max", "mom", "vol"],
    }, indent=2, default=str))

    if not passes:
        reasons = []
        if metrics.get("avg_sharpe", 999)       < _min_sharpe:
            reasons.append(f"Sharpe {metrics.get('avg_sharpe')} < {_min_sharpe}")
        if metrics.get("worst_drawdown_pct", 0) < _max_drawdown_pct:
            reasons.append(f"MaxDD {metrics.get('worst_drawdown_pct')}% < {_max_drawdown_pct}%")
        if metrics.get("total_trades", 999)      < _min_trade_count:
            reasons.append(f"Trades {metrics.get('total_trades')} < {_min_trade_count}")
        print("[archive_best_model] GATE FAILED -- live model unchanged.")
        print(f"[archive_best_model] Reasons: {'; '.join(reasons) or 'unknown'}")
        print(f"[archive_best_model] Rejected artefacts -> {archive_dir}")
        return

    # Promote
    live_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(staging_model, live_model)
    shutil.copy2(staging_stats, live_stats)
    print("[archive_best_model] SUCCESS -- new model promoted to live.")
    print(f"  live model  : {live_model}")
    print(f"  live stats  : {live_stats}")
    print(f"  archive dir : {archive_dir}")
    print(f"  metrics     : {metrics}")


# ==============================================================================
# DAG definition
# ==============================================================================

_params = {
    "backtest_csv":    str(BACKTEST_CSV),
    "train_csv":       str(TRAIN_CSV),
    "test_csv":        str(TEST_CSV),
    "staging_dir":     str(STAGING_DIR),
    "live_model":      str(LIVE_MODEL),
    "live_stats":      str(LIVE_STATS),
    "archive_root":    str(ARCHIVE_ROOT),
    "oanda_cfg":       str(OANDA_CFG),
    "lookback":           LOOKBACK,
    "window":             WINDOW,
    "validation_months":  VALIDATION_MONTHS,
    "test_months":        TEST_MONTHS,
    "patience":           PATIENCE,
    "seed":               SEED,
    "wf_windows":         WF_WINDOWS,
    "bars_per_fetch":     BARS_PER_FETCH,
    "oanda_stale_days":   OANDA_STALE_DAYS,
    "min_sharpe":         MIN_SHARPE,
    "max_drawdown_pct":   MAX_DRAWDOWN_PCT,
    "min_trade_count":    MIN_TRADE_COUNT,
}

default_args = {
    "owner":            "dissertation",
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "email_on_failure": False,
    "email_on_retry":   False,
}

with DAG(
    dag_id="retrain_lstm",
    description=(
        "Weekly LSTM retraining: fetch -> preprocess -> "
        "hyperparams+train (venv) -> walk-forward backtest (venv) -> archive"
    ),
    schedule_interval="0 2 * * 0",
    start_date=datetime(2026, 4, 27),
    catchup=False,
    default_args=default_args,
    params=_params,
    tags=["forex", "lstm", "retrain", "mlops"],
) as dag:

    t_fetch = PythonOperator(
        task_id="fetch_training_data",
        python_callable=fetch_training_data,
    )

    t_preprocess = PythonOperator(
        task_id="preprocess_splits",
        python_callable=preprocess_splits,
    )

    # TF isolated in virtualenv ------------------------------------------------
    t_train = PythonVirtualenvOperator(
        task_id="tune_and_train",
        python_callable=tune_and_train_venv,
        requirements=_TF_REQUIREMENTS,
        system_site_packages=False,
        use_dill=True,
        venv_cache_path=str(VENV_CACHE),
        op_kwargs={
            "train_csv":         str(TRAIN_CSV),
            "staging_dir":       str(STAGING_DIR),
            "lookback":          LOOKBACK,
            "window":            WINDOW,
            "validation_months": VALIDATION_MONTHS,
            "patience":          PATIENCE,
            "seed":              SEED,
            "param_grid":        PARAM_GRID,
            "feature_cols":      FEATURE_COLS,
        },
        execution_timeout=timedelta(hours=4),
    )

    t_backtest = PythonVirtualenvOperator(
        task_id="iterative_backtest",
        python_callable=iterative_backtest_venv,
        requirements=_TF_REQUIREMENTS,
        system_site_packages=False,
        use_dill=True,
        venv_cache_path=str(VENV_CACHE),
        op_kwargs={
            "train_result":     "{{ ti.xcom_pull(task_ids='tune_and_train') }}",
            "train_csv":        str(TRAIN_CSV),
            "test_csv":         str(TEST_CSV),
            "wf_windows":       WF_WINDOWS,
            "min_sharpe":       MIN_SHARPE,
            "max_drawdown_pct": MAX_DRAWDOWN_PCT,
            "min_trade_count":  MIN_TRADE_COUNT,
        },
        execution_timeout=timedelta(hours=1),
    )

    t_archive = PythonOperator(
        task_id="archive_best_model",
        python_callable=archive_best_model,
    )

    t_fetch >> t_preprocess >> t_train >> t_backtest >> t_archive
