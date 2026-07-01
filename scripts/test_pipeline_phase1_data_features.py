# Test pipeline phase 1
import os
import pandas as pd

# Make sure python can find the src package
import sys
sys.path.append(os.path.abspath("src"))

from data.load_data import load_data
from data.preprocess import preprocess_data
from features.build_features import build_features

# CONFIGURATION
DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COL = "Churn"

def testing_pipeline1():
    print("=== Testing Phase 1 : Load -> Preprocess -> Build Features ===")
    
    # 1) Loading data
    print("1) Loading data...")
    df = load_data(DATA_PATH)
    print(f"✅ Data loaded. Shape: {df.shape}")
    print(df.head(3))
    
    # 2) Preprocess data
    print("2) Preprocessing data...")
    df_preprocessed = preprocess_data(df, target_col=TARGET_COL)
    print(f"✅ Data preprocessed. Shape: {df_preprocessed.shape}")
    print(df_preprocessed.head(3))
    
    # 3) Build features
    print("3) Building features...")
    df_features = build_features(df_preprocessed, target_col=TARGET_COL)
    print(f"✅ Data after feature engineering - Shape : {df_features.shape}")
    print(df_features.head(3))
    
    print("=== Testing Phase 1 : Completed ===")
    
if __name__=="__main__":
    testing_pipeline1()
    
    
    
    
        
    
    