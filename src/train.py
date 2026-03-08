import pandas as pd
import sys
from tomlkit import datetime
import yaml
import os 
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle
from mlflow.models import infer_signature
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/Mohit-Bhoir/Multimodal-Trihybrid-Algorithmic-Forex-Trading-Model-.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "Mohit-Bhoir"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "8407fd9e577fa22b1f5f3df39f7a37252fd1ded3"


def hyperparameter_tuning(X_train, y_train,param_grid):
    # Define the model
    rf = RandomForestClassifier()

    # Perform grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Return the best hyperparameters
    return grid_search.best_params_

#Load params from params.yaml
params = yaml.safe_load(open("params.yaml"))['train']

def train(train_data_path, model, param_grid):
    X = pd.read_csv(train_data_path,parse_dates=['time'], index_col='time')
    y = X.pop('direction')
    
    
    

