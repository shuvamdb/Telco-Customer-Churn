import pandas as pd

# Applying binary encoding to 2 category features. Helper function work on a single column, hence series

def _map_binary_series(s: pd.Series) -> pd.Series:
    
    '''
    This function implements binary encoding logic that converts
    categorical features with exactly 2 values into 0/1 integers.
    
    '''
    # Get unique values and remove NaN values
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)
    
    # Yes/No mapping
    if valset == {"Yes","No"}:
        return s.map({"No" : 0, "Yes" : 1}).astype(int)
    
    # Gender Mapping
    if valset == {"Male","Female"}:
        return s.map({"Female":0 , "Male":1}).astype(int)
    
    # For any other two category feature
    if len(vals)==2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]:0, sorted_vals[1]:1}
        return s.astype(str).map(mapping).astype(int)
    
    # FOR NON-BINARY FEATURES - return unchanged, will be handled through One-Hot Encoding
    return s

def build_features(df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
    '''
    Main feature engineering function which transfroms raw customer data into ML ready features.
    The transformations must exactly be replicated in the serving pipeline to ensure  prediction accuracy.
    (Transformations before training the model must be same when applying the model on new data)
    
    '''
    
    df = df.copy()
    print(f"Starting feature engineering on {df.shape[1]} columns...")
    
    # Identifying Feature Types -- STEP 1
    # Find categorical features with datatype as 'object' apart from the target variable
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns 
                if c!=target_col]
    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    
    print(f"✅ Found {len(obj_cols)} categorical columns and {len(numeric_cols)} numeric columns")
    
    # Splitting categories by cardinality--STEP 2
    # Binary features (having 2 unique values) gets binary-encoding
    # Multi-valued features (having > 2 unique values) gets one-hot encoding.
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique()==2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique()>2]
    
    print(f"Binary features : {len(binary_cols)} | Multi-category features : {len(multi_cols)}")
    if binary_cols:
        print(f"Binary Columns : {binary_cols}")
    if multi_cols:
        print(f"Multi-Category Columns : {multi_cols}")
    
    # Applying Binary Encoding--STEP 3
    # Convert 2 categpry features to 0/1 using deteministic mappings
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))
        print(f"{c} : {original_dtype} -> Binary(0/1)")
        
    # Convert boolean columns--STEP 4
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"✅ Converted {len(bool_cols)} boolean columns to int : {bool_cols}")
        
    # One-Hot Encoding to multi valued categorical columns--STEP 5
    if multi_cols:
        print(f"Applying One-Hot Encoding on {len(multi_cols)} multi-catgeory columns...")
        original_shape = df.shape
        
        # Applying One Hot Encoding with drop_first=True
        df = pd.get_dummies(df, columns=multi_cols, drop_first='True')
        
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)  
        # e.g.= If 4 columns existed (incl 2 multi category cols, one has 3 values another has 4), then 2 + 2 + 3=7 columns after encoding,
        # then 2acc to formula 7 (new cols) - 4 (old cols) + 2 (len) = 5, i.e. 5 new features
        
        print(f"✅ Created {new_features} new features from {len(multi_cols)} multi-category columns")
        
    # Data-Type Cleanup--STEP 6
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            
            # Then fill any NaN value with 0 and convert to int
            df[c] = df[c].fillna(0).astype(int)
    
    print(f"✅ FEATURE ENGINEERING COMPLETE : {df.shape[1]} final features")
    return df
        
    
        
          
    
    
    
    
        