import pandas as pd
import sys
from tomlkit import datetime
import yaml
import os 
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
from mlflow.models import infer_signature
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv
import mlflow
from mlflow.models import infer_signature
import warnings
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

def hyperparameter_tuning(X_train, y_train,param_grid,model_path):
    # Define the model
    rf = RandomForestClassifier()

    # Perform grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Return the best hyperparameters
    return grid_search

#Load params from params.yaml
params = yaml.safe_load(open("params.yaml"))['train']

def train(X_train_path,y_train_path,X_test_path,y_test_path, model_path, param_grid):
    X_train = pd.read_csv(X_train_path, index_col="time", parse_dates=["time"])
    X_test = pd.read_csv(X_test_path, index_col="time", parse_dates=["time"])

    y_train = pd.read_csv(y_train_path, index_col="time", parse_dates=["time"])["direction"]
    y_test = pd.read_csv(y_test_path, index_col="time", parse_dates=["time"])["direction"]
    
    
    signature = infer_signature(X_train,y_train)
    
    # Perform hyperparameter tuning
    grid_search = hyperparameter_tuning(X_train, y_train, param_grid, model_path)
    best_model = grid_search.best_estimator_

    # Train the model with the best hyperparameters
    best_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    #Start mlflow run 
    mlflow.set_experiment("Random Forest Strategy for FX Trading")
    with mlflow.start_run():
        
        #Perform hyperparameter tuning again to log the best params
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid, model_path)
        
        #Get the best model
        best_model = grid_search.best_estimator_
        
        #Predict and evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        
        
        #Log additional metrics & params
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_params(grid_search.best_params_)

        tracking_url_type_store = urlparse(MLFLOW_TRACKING_URI).scheme
        
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="RandomForestFXTradingModel")
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
            
        
        #Create the directory to save the model if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok= True)

        filename = model_path
        pickle.dump(best_model,open(filename,'wb'))
    
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train(params['X_train_path'],params['y_train_path'],params['X_test_path'],params['y_test_path'], params['model_path'], params['param_grid'])
