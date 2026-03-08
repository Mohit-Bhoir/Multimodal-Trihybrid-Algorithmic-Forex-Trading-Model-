import pandas as pd
import sys
from tomlkit import datetime
import yaml
import os 
import numpy as np
import datetime


#Load params from params.yaml

params = yaml.safe_load(open("params.yaml"))['preprocess']

def preprocess(input_path, output_train, output_test, test_days, n_lags):
    #Read the data
    df = pd.read_csv(input_path,parse_dates=['time'], index_col='time')
    
    #Preprocess the data
    
    #Generate log-returns 
    df['returns'] = np.log(df['price'] / df['price'].shift(1))    
    
    #Generate direction
    df['direction'] = np.sign(df['returns'])
    
    #Prepare features 
    cols = []
    for lag in range(1, n_lags + 1):
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        cols.append(f'returns_lag_{lag}')
    df.dropna(inplace=True)
    
    
    #Normalise the features
    mean = df[cols].mean()
    std = df[cols].std().replace(0, 1)
    df.loc[:, cols] = (df[cols] - mean) / std
    
    #Split data into train and test sets
    test_start_date = df.index[-1] - datetime.timedelta(days=test_days)
    train_data = df[df.index < test_start_date].copy()
    test_data = df[df.index >= test_start_date].copy()
    
    print(f"\nTraining samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")
    
    #Save the preprocessed data
    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    train_data.to_csv(output_train)
    
    os.makedirs(os.path.dirname(output_test), exist_ok=True)
    test_data.to_csv(output_test)
    
if __name__ == "__main__":
    preprocess(params['input_path'], params['output_train'], params['output_test'], params['test_days'], params['n_lags'])