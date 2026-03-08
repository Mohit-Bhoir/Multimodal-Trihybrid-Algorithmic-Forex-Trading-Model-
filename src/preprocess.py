import pandas as pd
import yaml
import os
import numpy as np
import datetime

params = yaml.safe_load(open("params.yaml"))["preprocess"]

def preprocess(input_path, X_train_path, X_test_path, y_train_path, y_test_path, test_days, n_lags):
    df = pd.read_csv(input_path, parse_dates=["time"], index_col="time")

    df["returns"] = np.log(df["price"] / df["price"].shift(1))
    df["direction"] = np.sign(df["returns"])

    cols = []
    for lag in range(1, n_lags + 1):
        column_name = f"returns_lag_{lag}"
        df[column_name] = df["returns"].shift(lag)
        cols.append(column_name)

    df.dropna(inplace=True)

    mean = df[cols].mean()
    std = df[cols].std().replace(0, 1)
    df.loc[:, cols] = (df[cols] - mean) / std

    test_start_date = df.index[-1] - datetime.timedelta(days=test_days)

    train_data = df[df.index < test_start_date].copy()
    test_data = df[df.index >= test_start_date].copy()

    X_train = train_data[cols]
    y_train = train_data["direction"]
    X_test = test_data[cols]
    y_test = test_data["direction"]

    print(f"\nTraining samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")

    os.makedirs(os.path.dirname(X_train_path), exist_ok=True)
    X_train.to_csv(X_train_path)

    os.makedirs(os.path.dirname(X_test_path), exist_ok=True)
    X_test.to_csv(X_test_path)

    os.makedirs(os.path.dirname(y_train_path), exist_ok=True)
    y_train.to_csv(y_train_path)

    os.makedirs(os.path.dirname(y_test_path), exist_ok=True)
    y_test.to_csv(y_test_path)

if __name__ == "__main__":
    preprocess(
        params["input_path"],
        params["X_train_path"],
        params["X_test_path"],
        params["y_train_path"],
        params["y_test_path"],
        params["test_days"],
        params["n_lags"],
    )