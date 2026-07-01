import pandas as pd

# df should be a pandas deataframe and Churn is the target variable, rest are features

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    
    # Removing whitespace from the column headers for tidyness
    df.columns = df.columns.str.strip()
    
    # drop ids if present
    for col in ['customerID','customer_id','CustomerID']:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # target to 0/1 if values are Yes/No
    if target_col in df.columns and df[target_col].dtype==object:
        df[target_col] = df[target_col].str.strip().map({"No":0,"Yes":1})
    
    # Total charges often has blanks
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
    # Senior Citizen column should 0/1 irrespective of being blank
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].fillna(0).astype(int)
        
    # fill NA values with 0 for numeric columns | for other dtype colmns, encoders handles it 
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    return df

    