import pandas as pd
import os

# file_path should be a string, -> pd.DataFrame means this functon i.e. load_data will return a pandas DataFrame (a table)

def load_data(file_path : str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File Not Found: {file_path}")
    
    return pd.read_csv(file_path)

'''

if __name__ == "__main__":

    file_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = load_data(file_path)

    print(df.head())
    print(df.shape)
    print(df.info())
    
'''