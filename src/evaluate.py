import os
import pickle
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not all([MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD]):
    raise ValueError("Missing MLflow credentials in .env")

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

params = yaml.safe_load((BASE_DIR / "params.yaml").read_text())["evaluate"]


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def load_pickle(path_value):
    with open(resolve_path(path_value), "rb") as file_handle:
        return pickle.load(file_handle)


def generate_features(data, window=20):
    data = data.copy()
    data["returns"] = np.log(data["price"] / data["price"].shift(1))
    data["direction"] = np.where(data["returns"] > 0, 1, 0)
    data["sma"] = data["price"].rolling(window).mean() - data["price"].rolling(150).mean()
    data["boll"] = (data["price"] - data["price"].rolling(window).mean()) / data["price"].rolling(window).std()
    data["min"] = data["price"].rolling(window).min() / data["price"] - 1
    data["max"] = data["price"].rolling(window).max() / data["price"] - 1
    data["mom"] = data["returns"].rolling(3).mean()
    data["vol"] = data["returns"].rolling(window).std()
    data.dropna(inplace=True)
    return data


def compute_hit_ratio(predictions, actuals):
    hits = np.sign(predictions * actuals)
    correct_predictions = int((hits == 1).sum())
    hit_ratio = (correct_predictions / len(hits) * 100) if len(hits) else 0.0
    return hit_ratio, correct_predictions


def is_within_trading_window(timestamp, start_hour=13, end_hour=17):
    current_time = timestamp.time()
    return current_time >= pd.Timestamp(f"{start_hour:02d}:00:00").time() and current_time <= pd.Timestamp(
        f"{end_hour:02d}:00:00"
    ).time()


def extract_long_probability(model, features):
    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba, so the probability trade filter cannot be applied.")

    probabilities = model.predict_proba(features)
    class_to_index = {label: index for index, label in enumerate(model.classes_)}
    if 1 not in class_to_index:
        raise ValueError("Loaded model does not expose class '1', so long-class probability thresholds cannot be applied.")

    long_probabilities = probabilities[:, class_to_index[1]]
    return pd.Series(long_probabilities, index=features.index, name="pred_proba")


def build_feature_frame(data, lags, mean, std):
    """Build feature frame with lagged features for model input."""
    data = generate_features(data)
    feature_cols = []
    
    for col in ["returns", "sma", "boll", "min", "max", "mom", "vol"]:
        if col in data.columns:
            for lag in range(1, lags + 1):
                lag_col = f"{col}_lag_{lag}"
                data[lag_col] = data[col].shift(lag)
                feature_cols.append(lag_col)
    
    # Standardize features
    data[feature_cols] = (data[feature_cols] - mean) / std
    data.dropna(inplace=True)
    
    return data, feature_cols


class IterativeBase:

    def __init__(self, symbol, start, end, amount, use_spread=True, data=None, verbose=True):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_balance = amount
        self.current_balance = amount
        self.units = 0
        self.use_spread = use_spread
        self.trades = 0
        self.position = 0
        self.verbose = verbose
        self.data = self.get_data(data)

    def get_data(self, data=None):
        if data is None:
            raw = pd.read_csv(resolve_path(params["test_path"]), parse_dates=["time"], index_col="time")
        else:
            raw = data.copy()

        raw = raw.sort_index()

        if self.start is not None:
            raw = raw.loc[raw.index >= pd.Timestamp(self.start)]
        if self.end is not None:
            raw = raw.loc[raw.index <= pd.Timestamp(self.end)]

        if "spread" not in raw.columns:
            raw["spread"] = 0.0
        if "returns" not in raw.columns:
            raw["returns"] = np.log(raw["price"] / raw["price"].shift(1))

        return raw.dropna().copy()

    def reset(self):
        self.current_balance = self.initial_balance
        self.units = 0
        self.trades = 0
        self.position = 0

    def get_values(self, bar):
        timestamp = self.data.index[bar]
        date = str(timestamp.date())
        time = str(timestamp.time())
        price = round(self.data["price"].iloc[bar], 5)
        spread = round(self.data["spread"].iloc[bar], 5)
        return date, time, price, spread

    def get_current_balance(self, bar):
        date, time, _, _ = self.get_values(bar)
        if self.verbose:
            print(f"Date: {date} {time} | Current Balance: {self.current_balance:.2f}")

    def buy_instrument(self, bar, units=None, amount=None):
        date, time, price, spread = self.get_values(bar)
        if self.use_spread:
            price += spread / 2
        if amount is not None:
            units = int(amount / price)
        self.current_balance -= units * price
        self.units += units
        self.trades += 1
        if self.verbose:
            print(
                f"Date: {date} {time} | Action: BUY | Units: {units} | Price: {price:.5f} | Current Balance: {self.current_balance:.2f}"
            )

    def sell_instrument(self, bar, units=None, amount=None):
        date, time, price, spread = self.get_values(bar)
        if self.use_spread:
            price -= spread / 2
        if amount is not None:
            units = int(amount / price)
        self.current_balance += units * price
        self.units -= units
        self.trades += 1
        if self.verbose:
            print(
                f"Date: {date} {time} | Action: SELL | Units: {units} | Price: {price:.5f} | Current Balance: {self.current_balance:.2f}"
            )

    def current_nav(self, bar):
        _, _, price, _ = self.get_values(bar)
        return self.current_balance + (self.units * price)

    def performance_snapshot(self, bar, log_perf=False):
        nav = self.current_nav(bar)
        performance = (nav - self.initial_balance) / self.initial_balance * 100
        if log_perf:
            mlflow.log_metric("net_performance", performance)
            mlflow.log_metric("ending_balance", nav)
            mlflow.log_metric("trades_executed", self.trades)
        return {
            "ending_balance": nav,
            "net_performance_pct": performance,
            "trades_executed": self.trades,
        }

    def close_pos(self, bar, log_perf=False):
        date, time, price, spread = self.get_values(bar)
        print(75 * "-")
        print(f"Date: {date} {time} | Closing Position of {self.units} units at {price:.5f}")
        self.current_balance += self.units * price
        self.current_balance -= (abs(self.units) * spread / 2) * self.use_spread
        self.units = 0
        self.position = 0
        self.trades += 1
        summary = self.performance_snapshot(bar, log_perf=log_perf)
        self.get_current_balance(bar)
        print(f"Net Performance: {summary['net_performance_pct']:.2f}%")
        print(f"Number of trades executed: {self.trades}")
        print(75 * "-")
        return summary


class IterativeBacktest(IterativeBase):

    def go_long(self, bar, units=None, amount=None):
        if self.position == -1:
            self.buy_instrument(bar, units=-self.units)
        if units is not None:
            self.buy_instrument(bar, units=units)
        elif amount is not None:
            if amount == "all":
                amount = self.current_balance
            self.buy_instrument(bar, amount=amount)

    def go_short(self, bar, units=None, amount=None):
        if self.position == 1:
            self.sell_instrument(bar, units=self.units)
        if units is not None:
            self.sell_instrument(bar, units=units)
        elif amount is not None:
            if amount == "all":
                amount = self.current_balance
            self.sell_instrument(bar, amount=amount)

    def run_prediction_strategy(
        self,
        predictions,
        actuals=None,
        probabilities=None,
        short_probability_threshold=0.48,
        long_probability_threshold=0.52,
        trading_start_hour=13,
        trading_end_hour=17,
        log_perf=False,
    ):
        predictions = predictions.loc[self.data.index].astype(int)
        if probabilities is not None:
            probabilities = probabilities.loc[self.data.index]
        self.reset()

        if self.verbose:
            print("\n" + "=" * 75)
            print("STARTING ITERATIVE BACKTEST")
            print(f"Bars evaluated: {len(predictions)}")
            print("=" * 75)

        position_history = []
        trade_history = []
        probability_history = []
        trading_window_history = []
        probability_filter_history = []
        decision_reason_history = []

        for index, prediction in predictions.items():
            bar = self.data.index.get_loc(index)
            trade_executed = 0
            action = "HOLD"
            decision_probability = float(probabilities.loc[index]) if probabilities is not None else np.nan
            within_trading_window = is_within_trading_window(
                index,
                start_hour=trading_start_hour,
                end_hour=trading_end_hour,
            )
            short_signal = pd.notna(decision_probability) and decision_probability < short_probability_threshold
            long_signal = pd.notna(decision_probability) and decision_probability > long_probability_threshold
            passes_probability_filter = short_signal or long_signal
            decision_reason = "No trade signal"

            if not within_trading_window:
                action = "SKIP"
                decision_reason = f"Outside trading window {trading_start_hour:02d}:00-{trading_end_hour:02d}:00 UTC"
            elif probabilities is not None and not passes_probability_filter:
                action = "HOLD"
                decision_reason = (
                    f"Long probability {decision_probability:.2%} between "
                    f"{short_probability_threshold:.2%} and {long_probability_threshold:.2%}"
                )
            elif long_signal and self.position <= 0:
                action = "GO LONG"
                decision_reason = f"Long probability {decision_probability:.2%} > {long_probability_threshold:.2%}"
                self.go_long(bar, amount="all")
                self.position = 1
                trade_executed = 1
            elif short_signal and self.position >= 0:
                action = "GO SHORT"
                decision_reason = f"Long probability {decision_probability:.2%} < {short_probability_threshold:.2%}"
                self.go_short(bar, amount="all")
                self.position = -1
                trade_executed = 1
            elif long_signal:
                action = "STAY LONG"
                decision_reason = "Already long"
            elif short_signal:
                action = "STAY SHORT"
                decision_reason = "Already short"
            elif prediction == 0:
                action = "NEUTRAL"
                decision_reason = "Predicted neutral class"

            if self.verbose:
                print(
                    f"Date: {index.strftime('%Y-%m-%d %H:%M:%S')} | Model Decision: {prediction} | Probability: {decision_probability:.2%} | Action: {action} | Position: {self.position} | Reason: {decision_reason}"
                )

            position_history.append(self.position)
            trade_history.append(trade_executed)
            probability_history.append(decision_probability)
            trading_window_history.append(within_trading_window)
            probability_filter_history.append(passes_probability_filter)
            decision_reason_history.append(decision_reason)

        last_bar = self.data.index.get_loc(predictions.index[-1])
        if self.units != 0:
            summary = self.close_pos(last_bar, log_perf=log_perf)
        else:
            summary = self.performance_snapshot(last_bar, log_perf=log_perf)

        if actuals is not None:
            hit_ratio, correct_predictions = compute_hit_ratio(predictions, actuals.loc[predictions.index])
            summary["hit_ratio"] = hit_ratio
            summary["correct_predictions"] = correct_predictions
            summary["total_predictions"] = len(predictions)

        results = self.data.loc[predictions.index].copy()
        results["pred"] = predictions
        results["position"] = position_history
        results["trade_executed"] = trade_history
        results["pred_proba"] = probability_history
        results["within_trading_window"] = trading_window_history
        results["passed_probability_filter"] = probability_filter_history
        results["decision_reason"] = decision_reason_history

        return summary, results

    def test_logreg_strategy(self, lags=5, model_path=None, test_days=None, log_perf=False, feature_stats_path=None):
        if model_path is None:
            raise ValueError("model_path is required for unseen data evaluation. This method does not fit on evaluation data.")

        stats_path = feature_stats_path or params["model_params"]
        feature_stats = load_pickle(stats_path)
        feature_frame, feature_cols = build_feature_frame(
            self.data,
            lags,
            mean=feature_stats["mean"],
            std=feature_stats["std"],
        )

        model = load_pickle(model_path)
        feature_frame["pred"] = model.predict(feature_frame[feature_cols])
        feature_frame["pred_proba"] = extract_long_probability(model, feature_frame[feature_cols])
        summary, strategy_results = self.run_prediction_strategy(
            feature_frame["pred"],
            actuals=feature_frame["direction"],
            probabilities=feature_frame["pred_proba"],
            log_perf=log_perf,
        )

        merged_results = feature_frame.join(
            strategy_results[["position", "trade_executed"]],
            how="left",
        )

        print(f"\nHit Ratio: {summary['hit_ratio']:.2f}%")
        print(f"Total predictions: {summary['total_predictions']}")
        print(f"Correct predictions: {summary['correct_predictions']}")

        return model, merged_results, summary


def evaluate(
    test_path,
    model_path,
    stats_path,
    initial_balance=100000,
    use_spread=True,
    log_perf=True,
    verbose=True,
):
    # -----------------------
    # LOAD STATS + MODEL
    # -----------------------
    stats = load_pickle(stats_path)
    mean = stats["mean"]
    std = stats["std"]
    feature_cols = stats["feature_cols"]
    lookback = stats["lookback"]
    window = stats["window"]

    model = tf.keras.models.load_model(resolve_path(model_path))

    # -----------------------
    # LOAD & PREPARE TEST DATA
    # -----------------------
    test_data = pd.read_csv(resolve_path(test_path), parse_dates=["time"], index_col="time")
    df = generate_features(test_data, window=window)

    # Standardise with training stats
    df[feature_cols] = (df[feature_cols] - mean) / std

    # -----------------------
    # CREATE SEQUENCES
    # -----------------------
    features = df[feature_cols].values
    X_seq, seq_indices = [], []
    for i in range(lookback, len(features)):
        X_seq.append(features[i - lookback:i])
        seq_indices.append(df.index[i])

    X_seq = np.array(X_seq, dtype=np.float32)
    df = df.loc[seq_indices].copy()

    # -----------------------
    # PREDICTIONS
    # -----------------------
    probs = model.predict(X_seq, verbose=0).flatten()
    df["pred_proba"] = probs
    df["pred"] = (probs >= 0.5).astype(int)

    y_test = df["direction"]

    # -----------------------
    # METRICS
    # -----------------------
    accuracy = accuracy_score(y_test, df["pred"])
    hit_ratio, correct_predictions = compute_hit_ratio(df["pred"], df["direction"])

    # -----------------------
    # BACKTEST
    # -----------------------
    backtester = IterativeBacktest(
        symbol="EVALUATION",
        start=df.index.min(),
        end=df.index.max(),
        amount=initial_balance,
        use_spread=use_spread,
        data=df,
        verbose=verbose,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Unseen Test Data Prediction (LSTM)")

    with mlflow.start_run():

        backtest_summary, backtest_results = backtester.run_prediction_strategy(
            df["pred"],
            actuals=df["direction"],
            probabilities=df["pred_proba"],
            log_perf=log_perf,
        )

        # -----------------------
        # LOGGING
        # -----------------------
        mlflow.log_param("lookback", lookback)
        mlflow.log_param("window", window)
        mlflow.log_param("features", str(feature_cols))
        mlflow.log_param("initial_balance", initial_balance)
        mlflow.log_param("use_spread", use_spread)
        mlflow.log_param("trading_window_utc", "12:00-17:00")
        mlflow.log_param("short_probability_threshold", 0.47)
        mlflow.log_param("long_probability_threshold", 0.53)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("hit_ratio", hit_ratio)
        mlflow.log_metric("correct_predictions", correct_predictions)
        mlflow.log_metric("bars_in_trading_window", int(backtest_results["within_trading_window"].sum()))
        mlflow.log_metric("signals_passing_probability_filter", int(backtest_results["passed_probability_filter"].sum()))

        mlflow.tensorflow.log_model(model, artifact_path="model")

    # -----------------------
    # PRINT RESULTS
    # -----------------------
    print("Classification Report:")
    print(classification_report(y_test, df["pred"]))

    print(f"Accuracy: {accuracy:.4f}")
    print(df["pred"].value_counts())
    print(f"Hit Ratio: {hit_ratio:.2f}%")
    print(f"Net Performance: {backtest_summary['net_performance_pct']:.2f}%")
    print(f"Ending Balance: {backtest_summary['ending_balance']:.2f}")
    print(f"Trades Executed: {backtest_summary['trades_executed']}")

    # -----------------------
    # FINAL RESULTS
    # -----------------------
    evaluation_results = df.join(
        backtest_results[
            [
                "position",
                "trade_executed",
                "within_trading_window",
                "passed_probability_filter",
                "decision_reason",
            ]
        ],
        how="left",
    )

    summary = {
        "accuracy": accuracy,
        "hit_ratio": hit_ratio,
        "correct_predictions": correct_predictions,
        "net_performance_pct": backtest_summary["net_performance_pct"],
        "ending_balance": backtest_summary["ending_balance"],
        "trades_executed": backtest_summary["trades_executed"],
    }

    return summary, evaluation_results


if __name__ == "__main__":
    evaluate(
        params["test_path"],
        params["model_path"],
        params["stats_path"],
    )


