# inference.py is the bridge between trained model and API.

import os
import mlflow
import pandas as pd
import joblib

# =========================
# PROJECT PATH
# =========================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)
# __file__ gives the location of this file returns src/serving/inference.py
# ../.. goes 2 folder up i.e. E:\Projects\Telco-Customer-Churn-MLE
# Therefore, PROJECT_ROOT = E:\Projects\Telco-Customer-Churn-MLE


# =========================
# LOAD MLFLOW MODEL
# =========================

'''MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "911aacb584b545bb850df55e1dc5244a",
    "artifacts",
    "model",
    "model.pkl"
)'''
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "src",
    "serving",
    "model","911aacb584b545bb850df55e1dc5244a",
    "artifacts",
    "model",
    "model.pkl"
)
model = joblib.load(
    MODEL_PATH
)


'''model = joblib.load(
    MODEL_PATH
)'''

# MODEL_PATH = Creates a string of the path pointing to where model already exists. (created after run_pipeline.py)

'''MODEL_URI = "file:///" + MODEL_PATH.replace("\\", "/")
# replaces \\ to mlflow readable format '/'

# Loads model.pkl
# Mlfow looks for model file inside artifacts
model = mlflow.pyfunc.load_model(
    MODEL_URI
)'''

print(type(model))
print("Model loaded successfully")


# =========================
# LOAD PREPROCESSING ARTIFACTS
# =========================

# Preprocessing artifacts - contain information about how training transformed the data.
# preprocessing.pkl consist transformed state (for eg - Male:1, female:0....)
# feature_columns.txt contains the final column order

''' ARTIFACT_PATH = os.path.join(
    PROJECT_ROOT,
    "mlruns",
    "860777467256719846",
    "911aacb584b545bb850df55e1dc5244a",
    "artifacts"
)'''
ARTIFACT_PATH = os.path.join(
    PROJECT_ROOT,
    "src",
    "serving",
    "model","911aacb584b545bb850df55e1dc5244a","artifacts"
)


# load preprocessing dictionary || preprocessing_artifact=E:\Projects\Telco-Customer-Churn-MLE\mlruns\...\artifacts\preprocessing.pkl

preprocessing_artifact = joblib.load(
    os.path.join(
        ARTIFACT_PATH,
        "preprocessing.pkl"
    )
)

# load feature order - opens feature_columns.txt and reads it. f is a temp variable
with open(
    os.path.join(
        ARTIFACT_PATH,
        "feature_columns.txt"
    )
) as f:

    FEATURE_COLS = [
        line.strip()
        for line in f      # go through every line
    ]

print("Preprocessing loaded")
print("Feature count:", len(FEATURE_COLS))

# this func make sure that prediction-time data looks exactly like training-time data

def transform_input(df):

    df = df.copy()

    # remove spaces
    df.columns = df.columns.str.strip()

    # numeric conversion
    numeric_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]


    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            df[col] = df[col].fillna(0)
            
    BINARY_COLS = {
    "gender": {"Female": 0, "Male": 1},           # Demographics
    "Partner": {"No": 0, "Yes": 1},               # Has partner
    "Dependents": {"No": 0, "Yes": 1},            # Has dependents  
    "PhoneService": {"No": 0, "Yes": 1},          # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},      # Billing preference
    }
    
    for col, mapping in BINARY_COLS.items():
        if col in df.columns:
            df[col] = (df[col]
                       .map(mapping)
                       .astype(int)
                       .fillna(0))
    
    categorical_columns = df.select_dtypes(include=["object"]).columns
    
    # apply same preprocessing
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )


    # match training columns -- rearranges the exact same order of features
    df = df.reindex(
        columns=FEATURE_COLS,
        fill_value=0
    )


    return df

def predict(input_dict):
    
    # this function expects input in the form of dictionary ("gender":"Male",
    #                                                        "Partner":"Yes",
    #                                                        "Dependents":"No",
    #                                                        "PhoneService":"Yes")...
    # Converting the dictionary -> dataframe (designed for giving prediction for 1 cutsomer only , hence, dictionary)
    df = pd.DataFrame([input_dict])
    
    # transforms the input data (remove spaces, convert numeric cols, binary and one hot encoding, reindex )
    df_processed = transform_input(df)
    
    # model is MLflow loaded model
    prediction = model.predict(df_processed)

    # prediction[0] : 0 means row (single) --> designed for 1 customer only
    if prediction[0] == 1:
        return "Likely to churn"
    else:
        return "Not likely to churn"
    
'''
# For testing----------

test_customer = {

    "gender":"Male",
    "Partner":"Yes",
    "Dependents":"No",
    "PhoneService":"Yes",
    "MultipleLines":"No",
    "InternetService":"DSL",
    "OnlineSecurity":"No",
    "OnlineBackup":"Yes",
    "DeviceProtection":"No",
    "TechSupport":"No",
    "StreamingTV":"Yes",
    "StreamingMovies":"Yes",
    "Contract":"Month-to-month",
    "PaperlessBilling":"Yes",
    "PaymentMethod":"Electronic check",
    "MonthlyCharges":70,
    "TotalCharges":1000,
    "tenure":24
}


print(
    predict(test_customer)
)

'''