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
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1

import warnings
warnings.filterwarnings("ignore")


try:
    from DNNModel import create_model, set_seeds
except ImportError:
    from src.DNNModel import create_model, set_seeds

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

params = yaml.safe_load((BASE_DIR / "params.yaml").read_text())["train_dnn"]


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


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
    counts = np.bincount(labels, minlength=2)
    total = counts.sum()
    weights = {}
    for class_index, count in enumerate(counts):
        weights[class_index] = total / (2 * count) if count else 1.0
    return weights


def prepare_datasets(train_path, stats_path, test_months, n_lags):
    data = pd.read_csv(resolve_path(train_path), parse_dates=["time"], index_col="time")
    data["returns"] = np.log(data["price"] / data["price"].shift(1))
    data["direction"] = np.sign(data["returns"])

    feature_cols = []
    for lag in range(1, n_lags + 1):
        column_name = f"returns_lag_{lag}"
        data[column_name] = data["returns"].shift(lag)
        feature_cols.append(column_name)

    data = data.dropna().copy()

    mean = data[feature_cols].mean()
    std = data[feature_cols].std().replace(0, 1)

    stats_file = resolve_path(stats_path)
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "wb") as file_handle:
        pickle.dump({"mean": mean, "std": std}, file_handle)

    data.loc[:, feature_cols] = (data[feature_cols] - mean) / std

    test_start_date = data.index[-1] - pd.DateOffset(months=test_months)

    features = data[feature_cols].copy()
    targets = (data["direction"] > 0).astype(int)

    x_train = features[features.index < test_start_date].copy()
    y_train = targets[targets.index < test_start_date].copy()
    x_test = features[features.index >= test_start_date].copy()
    y_test = targets[targets.index >= test_start_date].copy()

    if x_train.empty or x_test.empty:
        raise ValueError("Training or holdout split is empty. Adjust test_months or inspect the input data.")

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Both training and holdout splits must contain both target classes.")

    return x_train, y_train, x_test, y_test, feature_cols


def build_compiled_model(config, input_dim):
    optimizer = Adam(learning_rate=config["learning_rate"])
    regularizer = l1(config.get("reg_strength", 0.0005))
    return create_model(
        hl=config["hl"],
        hu=config["hu"],
        dropout=config.get("dropout", False),
        rate=config.get("rate", 0.3),
        regularize=config.get("regularize", False),
        reg=regularizer,
        optimizer=optimizer,
        input_dim=input_dim,
    )


def build_tensorflow_signature(features, predictions):
    input_schema = Schema([
        TensorSpec(np.dtype(np.float32), (-1, features.shape[1]))
    ])
    if predictions.ndim <= 1:
        output_shape = (-1,)
    else:
        output_shape = (-1, *predictions.shape[1:])
    output_schema = Schema([
        TensorSpec(np.dtype(predictions.dtype), output_shape)
    ])
    return ModelSignature(inputs=input_schema, outputs=output_schema)


def fit_configuration(x_train, y_train, x_val, y_val, config, patience, seed):
    set_seeds(seed)
    tf.keras.backend.clear_session()

    model = build_compiled_model(config, input_dim=x_train.shape[1])
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]
    class_weights = compute_class_weights(y_train.to_numpy())
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    return model, float(val_loss), float(val_accuracy), best_epoch


def hyperparameter_tuning(x_train, y_train, param_grid, patience, seed):
    search_space = expand_param_grid(param_grid)
    max_splits = min(5, len(x_train) - 1)
    if max_splits < 2:
        raise ValueError("Not enough training rows for time-series cross-validation.")
    splitter = TimeSeriesSplit(n_splits=max_splits)
    best_result = None
    tuning_results = []

    for config in search_space:
        fold_losses = []
        fold_accuracies = []
        fold_epochs = []

        for train_idx, val_idx in splitter.split(x_train):
            x_fold_train = x_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            x_fold_val = x_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            if y_fold_train.nunique() < 2 or y_fold_val.nunique() < 2:
                continue

            _, val_loss, val_accuracy, best_epoch = fit_configuration(
                x_fold_train,
                y_fold_train,
                x_fold_val,
                y_fold_val,
                config,
                patience,
                seed,
            )
            fold_losses.append(val_loss)
            fold_accuracies.append(val_accuracy)
            fold_epochs.append(best_epoch)

        if not fold_accuracies:
            continue

        result = {
            **config,
            "cv_accuracy": float(np.mean(fold_accuracies)),
            "cv_loss": float(np.mean(fold_losses)),
            "cv_best_epoch": int(round(np.mean(fold_epochs))),
        }
        tuning_results.append(result)

        if best_result is None:
            best_result = result
            continue

        better_accuracy = result["cv_accuracy"] > best_result["cv_accuracy"]
        same_accuracy = np.isclose(result["cv_accuracy"], best_result["cv_accuracy"])
        lower_loss = result["cv_loss"] < best_result["cv_loss"]
        if better_accuracy or (same_accuracy and lower_loss):
            best_result = result

    if best_result is None:
        raise ValueError("Hyperparameter search failed because no valid folds could be evaluated.")

    return best_result, pd.DataFrame(tuning_results)


def split_train_validation(x_train, y_train, validation_split):
    validation_size = max(1, int(len(x_train) * validation_split))
    if validation_size >= len(x_train):
        raise ValueError("validation_split leaves no rows for model fitting.")

    x_fit = x_train.iloc[:-validation_size]
    y_fit = y_train.iloc[:-validation_size]
    x_val = x_train.iloc[-validation_size:]
    y_val = y_train.iloc[-validation_size:]

    if y_fit.nunique() < 2 or y_val.nunique() < 2:
        raise ValueError("Final train/validation split must contain both target classes.")

    return x_fit, y_fit, x_val, y_val


def train(train_path, model_path, stats_path, param_grid, test_months, n_lags, validation_split, patience, seed, registered_model_name):
    x_train, y_train, x_test, y_test, feature_cols = prepare_datasets(train_path, stats_path, test_months, n_lags)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("DNN Strategy for FX Trading")

    best_config, tuning_results = hyperparameter_tuning(x_train, y_train, param_grid, patience, seed)
    x_fit, y_fit, x_val, y_val = split_train_validation(x_train, y_train, validation_split)

    final_config = {
        key: value
        for key, value in best_config.items()
        if key not in {"cv_accuracy", "cv_loss", "cv_best_epoch"}
    }
    final_config["epochs"] = max(final_config["epochs"], best_config["cv_best_epoch"])

    with mlflow.start_run():
        final_model, _, _, _ = fit_configuration(
            x_fit,
            y_fit,
            x_val,
            y_val,
            final_config,
            patience,
            seed,
        )

        probabilities = final_model.predict(x_test, verbose=0).reshape(-1)
        predictions = (probabilities >= 0.5).astype(int)
        test_loss, test_accuracy = final_model.evaluate(x_test, y_test, verbose=0)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        mlflow.log_param("n_lags", n_lags)
        mlflow.log_param("test_months", test_months)
        mlflow.log_param("validation_split", validation_split)
        mlflow.log_param("seed", seed)
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

        input_example = x_train.head(5).to_numpy(dtype=np.float32)
        signature = build_tensorflow_signature(
            x_train,
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

        print(f"Features used: {feature_cols}")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Loss: {test_loss:.4f}")
        print(f"Best configuration: {final_config}")
        print(f"Model saved to {local_model_path}")


if __name__ == "__main__":
    train(
        train_path=params["train_path"],
        model_path=params["model_path"],
        stats_path=params["model_params"],
        param_grid=params["param_grid"],
        test_months=params["test_months"],
        n_lags=params["n_lags"],
        validation_split=params["validation_split"],
        patience=params["patience"],
        seed=params["seed"],
        registered_model_name=params["registered_model_name"],
    )