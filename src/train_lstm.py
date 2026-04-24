import os
import pickle
import random
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from dotenv import load_dotenv
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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

params = yaml.safe_load((BASE_DIR / "params.yaml").read_text())["train_lstm"]

FEATURE_COLS = ["returns", "sma", "boll", "min", "max", "mom", "vol"]


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def expand_param_grid(param_grid):
    if isinstance(param_grid, dict):
        return list(ParameterGrid(param_grid))
    if isinstance(param_grid, list):
        combinations = []
        for grid in param_grid:
            combinations.extend(list(ParameterGrid(grid)))
        return combinations
    raise ValueError("param_grid must be a dictionary or a list of dictionaries.")


def compute_class_weights(labels):
    counts = np.bincount(labels.astype(int), minlength=2)
    total = counts.sum()
    weights = {}
    for class_index, count in enumerate(counts):
        weights[class_index] = total / (2 * count) if count else 1.0
    return weights


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


def prepare_datasets(train_path, stats_path, validation_months, lookback, window):
    data = pd.read_csv(resolve_path(train_path), parse_dates=["time"], index_col="time")
    data = generate_features(data, window=window)

    val_start_date = data.index[-1] - pd.DateOffset(months=validation_months)

    train_mask = data.index < val_start_date
    mean = data.loc[train_mask, FEATURE_COLS].mean()
    std = data.loc[train_mask, FEATURE_COLS].std().replace(0, 1)

    stats_file = resolve_path(stats_path)
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "wb") as f:
        pickle.dump({
            "mean": mean,
            "std": std,
            "feature_cols": FEATURE_COLS,
            "lookback": lookback,
            "window": window,
        }, f)

    data[FEATURE_COLS] = (data[FEATURE_COLS] - mean) / std

    features = data[FEATURE_COLS].values
    targets = data["direction"].values
    dates = data.index

    X_all, y_all, seq_dates = [], [], []
    for i in range(lookback, len(features)):
        X_all.append(features[i - lookback:i])
        y_all.append(targets[i])
        seq_dates.append(dates[i])

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)
    seq_dates = np.array(seq_dates)

    train_idx = seq_dates < val_start_date
    val_idx = seq_dates >= val_start_date

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Training or validation split is empty. Adjust validation_months.")

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        raise ValueError("Both splits must contain both target classes.")

    return X_train, y_train, X_val, y_val


def build_lstm_model(config, input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(config["units_1"], return_sequences=True))
    model.add(Dropout(config["dropout"]))
    model.add(LSTM(config["units_2"], return_sequences=False))
    model.add(Dropout(config["dropout"]))
    model.add(Dense(1, activation="sigmoid"))
    optimizer = Adam(learning_rate=config.get("learning_rate", 0.001))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def fit_configuration(X_train, y_train, X_val, y_val, config, patience, seed):
    set_seeds(seed)
    tf.keras.backend.clear_session()

    model = build_lstm_model(config, input_shape=(X_train.shape[1], X_train.shape[2]))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    ]
    class_weights = compute_class_weights(y_train)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    return model, float(val_loss), float(val_accuracy), best_epoch


def hyperparameter_tuning(X_train, y_train, param_grid, patience, seed):
    search_space = expand_param_grid(param_grid)
    max_splits = min(5, len(X_train) - 1)
    if max_splits < 2:
        raise ValueError("Not enough training samples for time-series cross-validation.")
    splitter = TimeSeriesSplit(n_splits=max_splits)

    best_result = None
    tuning_results = []
    total = len(search_space)

    for idx, config in enumerate(search_space, 1):
        print(f"\n[Config {idx}/{total}] {config}")
        fold_losses = []
        fold_accuracies = []
        fold_epochs = []

        for train_idx, val_idx in splitter.split(X_train):
            X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
            X_fold_val, y_fold_val = X_train[val_idx], y_train[val_idx]

            if len(np.unique(y_fold_train)) < 2 or len(np.unique(y_fold_val)) < 2:
                continue

            _, val_loss, val_accuracy, best_epoch = fit_configuration(
                X_fold_train, y_fold_train,
                X_fold_val, y_fold_val,
                config, patience, seed,
            )
            fold_losses.append(val_loss)
            fold_accuracies.append(val_accuracy)
            fold_epochs.append(best_epoch)
            print(f"  Fold {len(fold_accuracies)}/{max_splits}: val_acc={val_accuracy:.4f}, val_loss={val_loss:.4f}, best_epoch={best_epoch}")

        if not fold_accuracies:
            continue

        result = {
            **config,
            "cv_accuracy": float(np.mean(fold_accuracies)),
            "cv_loss": float(np.mean(fold_losses)),
            "cv_best_epoch": int(round(np.mean(fold_epochs))),
        }
        tuning_results.append(result)
        print(f"  => CV accuracy: {result['cv_accuracy']:.4f}, CV loss: {result['cv_loss']:.4f}")

        if best_result is None:
            best_result = result
            continue

        better_accuracy = result["cv_accuracy"] > best_result["cv_accuracy"]
        same_accuracy = np.isclose(result["cv_accuracy"], best_result["cv_accuracy"])
        lower_loss = result["cv_loss"] < best_result["cv_loss"]
        if better_accuracy or (same_accuracy and lower_loss):
            best_result = result

    if best_result is None:
        raise ValueError("Hyperparameter search failed: no valid folds could be evaluated.")

    return best_result, pd.DataFrame(tuning_results)


def build_tensorflow_signature(X, predictions):
    input_schema = Schema([
        TensorSpec(np.dtype(np.float32), (-1, X.shape[1], X.shape[2]))
    ])
    if predictions.ndim <= 1:
        output_shape = (-1,)
    else:
        output_shape = (-1, *predictions.shape[1:])
    output_schema = Schema([
        TensorSpec(np.dtype(predictions.dtype), output_shape)
    ])
    return ModelSignature(inputs=input_schema, outputs=output_schema)


def train(train_path, model_path, stats_path, param_grid, validation_months,
          lookback, window, patience, seed, registered_model_name):

    X_train, y_train, X_val, y_val = prepare_datasets(
        train_path, stats_path, validation_months, lookback, window
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("LSTM Strategy for FX Trading")

    best_config, tuning_results = hyperparameter_tuning(
        X_train, y_train, param_grid, patience, seed
    )

    final_config = {
        key: value
        for key, value in best_config.items()
        if key not in {"cv_accuracy", "cv_loss", "cv_best_epoch"}
    }
    final_config["epochs"] = max(final_config["epochs"], best_config["cv_best_epoch"])

    with mlflow.start_run():
        final_model, _, _, _ = fit_configuration(
            X_train, y_train, X_val, y_val, final_config, patience, seed
        )

        probabilities = final_model.predict(X_val, verbose=0).reshape(-1)
        predictions = (probabilities >= 0.5).astype(int)
        test_loss, test_accuracy = final_model.evaluate(X_val, y_val, verbose=0)
        precision = precision_score(y_val, predictions, zero_division=0)
        recall = recall_score(y_val, predictions, zero_division=0)
        f1 = f1_score(y_val, predictions, zero_division=0)

        mlflow.log_param("lookback", lookback)
        mlflow.log_param("window", window)
        mlflow.log_param("validation_months", validation_months)
        mlflow.log_param("seed", seed)
        mlflow.log_param("features", str(FEATURE_COLS))
        mlflow.log_params(final_config)

        mlflow.log_metric("accuracy", float(test_accuracy))
        mlflow.log_metric("loss", float(test_loss))
        mlflow.log_metric("precision", float(precision))
        mlflow.log_metric("recall", float(recall))
        mlflow.log_metric("f1_score", float(f1))
        mlflow.log_metric("cv_accuracy", float(best_config["cv_accuracy"]))
        mlflow.log_metric("cv_loss", float(best_config["cv_loss"]))

        if not tuning_results.empty:
            mlflow.log_text(tuning_results.to_csv(index=False), "tuning_results.csv")

        input_example = X_train[:5]
        signature = build_tensorflow_signature(
            X_train,
            final_model.predict(input_example, verbose=0),
        )
        mlflow.tensorflow.log_model(
            model=final_model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

        local_model_path = resolve_path(model_path)
        local_model_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.save(local_model_path)

        print(f"Features used: {FEATURE_COLS}")
        print(f"Accuracy: {accuracy_score(y_val, predictions):.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Loss: {test_loss:.4f}")
        print(f"Best configuration: {final_config}")
        print(f"Model saved to {local_model_path}")
        print(f"Stats saved to {resolve_path(stats_path)}")


if __name__ == "__main__":
    train(
        train_path=params["train_path"],
        model_path=params["model_path"],
        stats_path=params["stats_path"],
        param_grid=params["param_grid"],
        validation_months=params["validation_months"],
        lookback=params["lookback"],
        window=params["window"],
        patience=params["patience"],
        seed=params["seed"],
        registered_model_name=params["registered_model_name"],
    )
