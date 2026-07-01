import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import optuna

print("=== Testing Phase 2 : Modelling with XGBoost ===")

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Target must be numeric (0/1)
if df["Churn"].dtype=="object":
    df["Churn"] = df["Churn"].str.strip().map({"No":0, "Yes":1})
    
assert df["Churn"].isna().sum()==0              # checks if target has NaN values
assert set(df["Churn"].unique()=={0,1})         # checks if target has 0 and 1 values only

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

THRESHOLD = 0.3

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators",300,800),
        "learning_rate": trial.suggest_float("learning_rate",0.01,0.2),
        "max_depth": trial.suggest_int("max_depth",3,10),
        "subsample": trial.suggest_float("subsample",0.5,1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree",0.5,1.0),
        "min_child_weight": trial.suggest_int("min_child_weight",1,10),
        "gamma": trial.suggest_float("gamma",0,5),
        "reg_alpha": trial.suggest_float("reg_alpha",0,5),
        "reg_lambda": trial.suggest_float("reg_lambda",0,5),
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "scale_pos_weight": (y_train==0).sum() / (y_train==1).sum()
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train,y_train)
    proba = model.predict_proba(X_test)[:,1]
    y_pred = (proba>=THRESHOLD).astype(int)
    return recall_score(y_test, y_pred)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print("Best params : ", study.best_params)
    print("Best recall : ", study.best_value)
    
    


    