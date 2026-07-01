'''
Runs sequentially | Load -> Validate -> Preprocess -> Feature Engineering

'''
import os
import sys
import argparse
import json
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib
import time

# Allows import from src/directory structure

# __file__ : the py file name
# os.path.dirname : It removes the filename and gives only the folder.
# os.path.join , ".." : join() combines paths.".." points to one folder back so path reaches project root level
# sys.path : sys.path is a list of folders Python searches when importing.
# abspath means "absolute path". It converts a relative path into a full path.

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

# Local modules
from src.data.load_data import load_data                            # Data loading with error handling
from src.data.preprocess import preprocess_data                     # Data Cleaning
from src.utils.validate_data import validate_telco_data             # Data Quality checks
from src.features.build_features import build_features              # Feature Engineering for model performance

def main(args):
    '''
    MAIN TRAINING PIPELINE THAT ORCHESTRATES THE ENTIRE ML PIPELINE WORKFLOW
    
    '''
    # MLFlow Setup = Essential for Experiment Tracking
    
    #Configure MLFlow to use local file based tracking (not in server)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    mlruns_path = args.mlflow_uri or f"file:///{project_root.replace(os.sep, '/')}/mlruns"                 # Local file based path
    # mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)                                            # Creates experiment if doesn't exist
    
    with mlflow.start_run():
        mlflow.log_param("model",'xgboost')                # Model Type for comparison
        mlflow.log_param("threshold",args.threshold)       # Classification threshold
        mlflow.log_param("test_size",args.test_size)       # Train/test split ratio
        
        # =======STAGE 1 - DATA LOADING AND VALIDATION==========
        print("🔄 Loading data...")
        df = load_data(args.input)
        print(f"✅ Data loaded : {df.shape[0]} rows, {df.shape[1]} columns")
        
        # ======CRITICAL - Data Quality Validation===============
        print("🔄 Validating data quality using Great Expectations...")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))
        
        if not is_valid:
            mlflow.log_text(json.dump(failed, indent=2), artifact_file="failed_expecatons.json")      # Convert the Python object into a JSON formatted string, indent=2 makes it readable
            raise ValueError(f"❌ Data Quality check failed. Issues : {failed}")
        else:
            print("✅ Data Quality check passed. Logged into MLFlow")
            
        # ========STAGE 2 - DATA PREPROCESSING=====================
        print("🔄 Preprocessing data...")
        df = preprocess_data(df)                          # Basic cleaning - Handles missing value, fix datatypes
        
        # Saving preprocessed data for reproducibility and debugging
        processed_path = os.path.join(project_root, "data", "preprocessed", "WA_Fn-UseC_-Telco-Customer-Churn_preprocessed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)                                 # Savig the preprocessed csv
        print(f"✅ Preprocessed dataset saved to : {processed_path} | Shape : {df.shape}")
        
        # =========STAGE 3 - FEATURE ENGINEERING=====================
        print("🔄 Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"❌ Target column '{target}' not found in data")
                             
        # Applying feature engineering transformations (Binary + One-Hot Encoding)
        df_enc = build_features(df, target_col = target )
        
        # Convert boolean columns to integer
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"✅ Feature Engineering completed : {df_enc.shape[1]} features")
        
        # Save reusable pipeline files (artifacts)
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Get feature columns(excluding target)
        feature_cols = list(df_enc.drop(columns=[target]).columns)
        
        # Saving the feature column order for reproducibility in local (development serving) -- creates json and opens in write mode
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)
            
        # Log to MLFlow for production serving
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")
        
        preprocessing_artifact = {
            "feature_columns" : feature_cols,
            "target" : target
        }
        
        # artifacts/preprocessing.pkl → for your application to use later (serving/prediction)-- 
        #                               so later when you receive new customer data, your system can load this file 
        #                               and apply the same feature order/preprocessing before making predictions.
        
        # mlruns/.../preprocessing.pkl is saved by MLflow only for experiment tracking, so you can remember 
        # which preprocessing setup was used for a particular training run and compare different experiments.
        
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"✅ Saved {len(feature_cols)} feature columns for serving consistency")
        
        # =======================STAGE 4 - Train/Test Split========================
        
        print("Splitting data...")
        X = df_enc.drop(columns=[target])
        y = df_enc[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,y,
            test_size=args.test_size,            # default : 20% test split
            stratify=y,                          # maintain class imbalance
            random_state=42                      # for reproducible split
        )
        print(f"✅ Train : {X_train.shape[0]} samples | Test : {X_test.shape[0]} samples")
        
        # Handle Class Imbalance
        # This tells XGboost to put more weight on the minority class (churners)
        scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()
        print(f"Class Imbalance ratio : {scale_pos_weight:.2f}")
        
        # ====================STAGE 5 - Model Training with optimized hyperparamters=============
        print("🔄 Training XGBoost model...")
        model = XGBClassifier(
            n_estimators=651,                                     # Number of trees (optimized)
            learning_rate= 0.058,                                 # 
            max_depth= 3,                                         # Maximum tree depth (optimized)
            subsample= 0.99,                                      # Sample ratio of training instances
            colsample_bytree= 0.50,                               # Sample ratio of features for each tree
            random_state= 42,
            n_jobs= -1,
            eval_metric= "logloss",
            scale_pos_weight = scale_pos_weight
        )
        
        # Train model and track training time
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)               # Track training time
        print(f"Model trained in : {train_time:.2f} seconds")
        
        # ==================STAGE 6 - Model Evaluation=========================
        print("🔄 Evaluating model performance...")
        
        # Generate prediction and track time
        t1 = time.time()
        proba = model.predict_proba(X_test)[:,1]                  # predict probablity of churn class 1
        
        y_pred = (proba>=args.threshold).astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)
        
        # Log evaluation metrics to mlflow - these metrics are essential for comparison and monitoring
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        mlflow.log_metric("precision", precision)                # out of predicted churners, how many actually churned?
        mlflow.log_metric("recall", recall)                      # out of actual churner, how many did we catch?
        mlflow.log_metric("f1", f1)                              # harmonic mean of precision and recall
        mlflow.log_metric("roc_auc", roc_auc)                    # area under roc curve
        
        print("Model Performance : ")
        print(f"Precision : {precision:.3f} | Recall : {recall:.3f}")
        print(f"F1 : {f1:.3f} | ROC_AUC : {roc_auc:.3f}")
        
        # =================STAGE 7 - Model Logging=====================
        print("🔄 Saving model to mlflow...")
        
        '''mlflow.sklearn.log_model(
            model, artifact_path="model"                         # this create 'model/' folder path inside mlflow un artifacts
            )   '''
        model_path = os.path.join(artifacts_dir,"model.pkl")

        joblib.dump(model, model_path)

        mlflow.log_artifact(model_path, artifact_path="model")    
               
        print("✅ Model saved to MLflow for serving pipeline")
        
        # Final perforamce summary
        print("\n Performance Summary : ")
        print(f"    Training time : {train_time:.2f} s")
        print(f"    Inference time : {pred_time:.4f} s")
        print(f"    Samples per second : {len(X_test)/pred_time:.0f}")
        
        print("\n Detailed Classification Report : ")
        print(classification_report(y_test, y_pred, digits = 3))       
       
if __name__=="__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True, help="Path to csv (e.g. - data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--threshold", type=float, default = 0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--mlflow_uri", type=str, default=None,
                   help="override MLflow tracking URI, else uses project_root/mlruns")
    args = p.parse_args()
    main(args)
    
    '''
    python scripts/run_pipeline.py \                                            
    --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
    --target Churn
    
    python scripts/run_pipeline.py --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv  
    '''