import pandas as pd
import sys
import yaml
import os 
import numpy as np
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import pickle
from mlflow.models import infer_signature
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv
import mlflow
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
    logistic_regression = LogisticRegression(random_state=42)

    # Perform grid search
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(estimator=logistic_regression, param_grid=param_grid, cv=tscv , n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Return the best hyperparameters
    return grid_search

#Load params from params.yaml
params = yaml.safe_load(open("params.yaml"))['train']

def train(train_path, model_path, param_grid, test_months, n_lags,model_params):
    # Load the data
    data = pd.read_csv(train_path, parse_dates=['time'], index_col='time')

    #Generate columns / tech indicators here
    data['returns'] = np.log(data['price'] / data['price'].shift(1))
    data['direction'] = np.sign(data['returns'])
    
    #Split data with recent 12 months as test set here
    test_start_date = data.index[-1] - pd.DateOffset(months=test_months)

    cols = []
    for lag in range(1, n_lags + 1):
        data[f'returns_lag_{lag}'] = data['returns'].shift(lag)
        cols.append(f'returns_lag_{lag}')
    data.dropna(inplace=True)   
    
    mean = data[cols].mean()
    std = data[cols].std()


    # Save mean and std to models/feature_stats.pkl for later use
    stats_path = os.path.join(BASE_DIR, 'models', 'feature_stats.pkl')
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)

    data[cols] = (data[cols] - mean) / std
     
    X = data[cols].copy()
    y = data['direction']

    X_train = X[X.index < test_start_date].copy()
    y_train = y[y.index < test_start_date].copy()
    X_test = X[X.index >= test_start_date].copy()
    y_test = y[y.index >= test_start_date].copy()
    
    
    
    signature = infer_signature(X_train, y_train)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    #Start mlflow run 
    mlflow.set_experiment("Logistic Regression Strategy for FX Trading")
    with mlflow.start_run():
        
        #Perform hyperparameter tuning again to log the best params
        grid_search = hyperparameter_tuning(X_train[cols], y_train, param_grid, model_path)
        
        #Get the best model
        best_model = grid_search.best_estimator_
        
        #Predict and evaluate the model
        y_pred = best_model.predict(X_test[cols])
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        
        confusion_matrix_result = confusion_matrix(y_test, y_pred)
        classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
        classification_report_text = classification_report(y_test, y_pred)

        print("Confusion Matrix:")
        print(confusion_matrix_result)
        print("\nClassification Report:")
        print(classification_report_text)

        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metrics({
            "precision_macro": float(classification_report_dict["macro avg"]["precision"]),
            "recall_macro": float(classification_report_dict["macro avg"]["recall"]),
            "f1_macro": float(classification_report_dict["macro avg"]["f1-score"]),
        })

        mlflow.log_dict(
            {
                "confusion_matrix": confusion_matrix_result.tolist(),
                "classification_report": classification_report_dict,
            },
            "evaluation_metrics.json",
        )

        mlflow.log_text(classification_report_text, "classification_report.txt")
        
        #Log additional metrics & params
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_params(grid_search.best_params_)

        tracking_url_type_store = urlparse(MLFLOW_TRACKING_URI).scheme

        signature = infer_signature(X_train, best_model.predict(X_train))

        mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5),
        registered_model_name="LogisticRegressionFXTradingModel"
         )
        
        #Create the directory to save the model if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok= True)

        filename = model_path
        pickle.dump(best_model,open(filename,'wb'))
        
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train(params['train_path'], params['model_path'], params['param_grid'],params['test_months'], params['n_lags'],params['model_params'])
