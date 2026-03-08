import pandas as pd
import yaml
import os
import numpy as np
import datetime

params = yaml.safe_load(open("params.yaml"))["preprocess"]

def preprocess(input_path, train_path, test_path, test_days):
    df = pd.read_csv(input_path, parse_dates=["time"], index_col="time")

    # df["returns"] = np.log(df["price"] / df["price"].shift(1))
    # df["direction"] = np.sign(df["returns"])


    test_start_date = df.index[-1] - datetime.timedelta(days=test_days)

    
    train_data = df[df.index < test_start_date].copy()
    test_data = df[df.index >= test_start_date].copy()

    print(f"\nTraining samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_data.to_csv(train_path)

    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    test_data.to_csv(test_path)

    

if __name__ == "__main__":
    preprocess(
        params["input_path"],
        params["train_path"],
        params["test_path"],
        params["test_days"],
    )