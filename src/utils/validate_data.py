# We want to make sure that the data that will come in, is qualitative...male/female, Yes/No, column exists or not...
# We expect these values to be in the range

import great_expectations as gx
from great_expectations.core.batch import Batch
from great_expectations.validator.validator import Validator
from great_expectations.execution_engine import PandasExecutionEngine
from typing import Tuple, List
import pandas as pd


def validate_telco_data(df) -> Tuple[bool, List[str]]:

    print("🔄 Starting validation of Telco data using Great Expectations")
    
    # so that blank values doesn't cause error
    df["tenure"] = pd.to_numeric(
        df["tenure"],
        errors="coerce"
    )

    df["MonthlyCharges"] = pd.to_numeric(
        df["MonthlyCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )
    #=====================================================================================================
    
    # In great expectation 1** version, dataset is not supported and is broken into multiple steps
    # ge_df = gx.dataset.PandasDataset(df)
    
    # Create Great Expectations context / environment. It manages data sources, expectation suites, etc
    context = gx.get_context()

    # Create execution engine, validates pandas df
    execution_engine = PandasExecutionEngine()

    # Create batch
    batch = Batch(
        data=df
    )

    # Create validator (chckes whether columns exists)
    ge_df = Validator(
        execution_engine=execution_engine,
        batches=[batch],
        expectation_suite=None,
        data_context=context
    )
    # Essential column validations
    ge_df.expect_column_to_exist('customerID')
    ge_df.expect_column_values_to_not_be_null('customerID')
    
    # Core columns
    ge_df.expect_column_to_exist('gender')
    ge_df.expect_column_to_exist('Partner')
    ge_df.expect_column_to_exist('Dependents')
    
    # Service feature which are critical for churn analysis
    ge_df.expect_column_to_exist('PhoneService')
    ge_df.expect_column_to_exist('InternetService')
    ge_df.expect_column_to_exist('Contract')
    
    # Financial features (Churn predictors)
    ge_df.expect_column_to_exist('tenure')
    ge_df.expect_column_to_exist('MonthlyCharges')
    ge_df.expect_column_to_exist('TotalCharges')
    
    print("🔄 Validating business logic contraints...")
    
    # Gender must be one of the expected values 
    ge_df.expect_column_values_to_be_in_set("gender", ["Male","Female"])
    
    # Yes/No fields must have valid values
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes","No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes","No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes","No"])
    
    # COntract types must be valid
    ge_df.expect_column_values_to_be_in_set("Contract", ["Month-to-month","One year","Two year"])
    
    # internet service types
    ge_df.expect_column_values_to_be_in_set("InternetService", ["DSL","Fiber optic","No"])
    
 #  NUMERIC RANGE VALIDATIONS
    
    print("🔄 Validating numeric ranges and business constraint...")
    
    # Tenure must be non-negative
    ge_df.expect_column_values_to_be_between("tenure", min_value=0)
    
    # Monthly charges & Total Charges must be positive (No Free Usage--Business constraint)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)
    
    print("🔄 Validating Statistical Properties...")
    
    # Tenure should be reasonable
    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    
    # Monthly charges should be between reasonable range (below $200--Business constraint)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0,max_value=200)
    
    # No missing values in the numeric critical features
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")
    
    print("🔄 Validating DATA CONSISTENCY CHECKS...")
    
    # Total charges should be >= Monthly charges except for new customers
    
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95  # Allow 5% exception
    )
    
    print("🔄 Running complete validation suite...")
    
    results = ge_df.validate()
    
    # Extract failed expectations for detailed error reporting
    failed_expectations=[]
    
    for r in results['results']:
        if not r['success']:
            expectation_type = r['expectation_config']['expectation_type']
            failed_expectations.append(expectation_type)
            
    # Total Validation Summary
    total_checks = len(results['results'])
    
    # Count passed checks. This counts how many expectations returned True (if True then 1, else 0 )
    passed_checks = sum(1 for r in results['results'] if r['success'])
    
    # Count failed checks
    failed_checks = total_checks - passed_checks
    
    if results['success']:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED:  {failed_checks}/{total_checks} checks failed")
        print(f" Failed expectations: {failed_expectations}")
    
    return results['success'], failed_expectations
            
    
    

