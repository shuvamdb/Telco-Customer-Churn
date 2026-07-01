import mlflow
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

def train_model(df:pd.DataFrame, target_col:str):
    '''
    Training a XGBoost Model and logs with MLFlow
    
    Args
    df (pd.dataFrame) : Feature dataset
    target_col (str) : Name of the target column
    
    '''
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(test_size=0.2, random_state=42)
    
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    '''
    MLflow is a tool that tracks, stores, and manages ML experiments 
    so we know what we trained, how good it was, and how to reproduce it later.
    
    '''
    
    with mlflow.start_run():
        # Train model
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)
        
        # Log params, metrics and model
        mlflow.log_param("n_estimators",300)
        mlflow.log_metric("accuracy",acc)
        mlflow.log_metric("recall",rec)
        mlflow.xgboost.log_model(model, "model")
        
        # Log dataset so that it shows in MLFlow UI
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")
        
        print(f"Model trained. Accuracy : {acc:.4f} , Recall : {rec:.4f}")
    