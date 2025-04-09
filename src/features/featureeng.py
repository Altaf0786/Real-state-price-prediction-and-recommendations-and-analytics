from pathlib import Path
import numpy as np
import os
import pandas as pd
import sys 
import logging
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from helpercode.missingAnalyser  import MissingValuesAnalysisTemplate , FullMissingnoAnalysis
from helpercode.missinghandler import (
    MissingValueHandler,
    SimpleImputationStrategy,
    KNNImputationStrategy,
    MICEImputationStrategy,
    MissingIndicatorStrategy,
    DeleteMissingValuesStrategy,
)
from helpercode.end_of_distribution_feasible import (
    load_data,
    NormalDistributionImputation,
    QuartileImputation,
    impute_and_analyze
)
from helpercode.mean_median_fesibale import load_data, impute_and_analyze
from helpercode.univariate_outlier import UnivariateOutlierDetector, ZScoreOutlierDetection, IQROutlierDetection


from helpercode.caa_feasible import  main,filter_columns_by_missing_values, compare_distributions
from helpercode.multi_outlier import OutlierDetection, KNNStrategy, OneClassSVMStrategy, IsolationForestStrategy, LOFStrategy, DBSCANStrategy
from helpercode.feature_encoading  import FeatureEngineeringStrategy,CommonEncoder,CommonEncoder,Feature_Engineering,OneHotEncoding,OrdinalEncoding,CountEncoding
# Setup logging
logger = logging.getLogger("DataIngestorApp")  # Using a specific logger name
logger.setLevel(logging.DEBUG)
# creat root path 
root_path = Path(__file__).parent.parent.parent

# Console Handler (for output to console)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set the level for console output
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Include name in the format
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
file_handler_path=root_path/'src/log/dataingetion.log'
# File Handler (for output to log file)
file_handler = logging.FileHandler(file_handler_path)  # Logs to 'app.log'
file_handler.setLevel(logging.DEBUG)  # Set the level for file output
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Include name in the format
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


def load_data(file_path):
    """Load the data from the specified file path."""
    return pd.read_csv(file_path)

def save(df,file_path):
    """Save the DataFrame to the specified file path."""
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")
    

def main():
    
    root_path = Path(__file__).parent.parent.parent
    
     # load the data
    train_data=root_path/'data'/'interim'/'train.csv'
    test_data=root_path/'data'/'interim'/'test.csv'
    train_df=load_data(train_data)
    test_df=load_data(test_data)
    
    print(train_df.head())
    print(train_df.columns)
    # handle missing value
    missing_value_handler = MissingValueHandler(SimpleImputationStrategy(strategy='mean'))
    df_train_imputed = missing_value_handler.handle_missing_values(train_df, columns=['PRICE', 'PRICE_SQFT','TRANSACT_TYPE', 'TOTAL_FLOOR','TOTAL_LANDMARK_COUNT','BEDROOM_NUM','BALCONY_NUM'])
    test_df_imputed = missing_value_handler.handle_missing_values(test_df, columns=['PRICE', 'PRICE_SQFT','TRANSACT_TYPE', 'TOTAL_FLOOR','TOTAL_LANDMARK_COUNT','BEDROOM_NUM','BALCONY_NUM'])
    

    """Handle outlier  values using a specified strategy for selected columns."""
    
    df_numeric_train = df_train_imputed.select_dtypes(include=[np.number]).dropna()
    df_numeric_test = test_df_imputed.select_dtypes(include=[np.number]).dropna()

    # Initialize the UnivariateOutlierDetector with the Z-Score based Outlier Detection Strategy
    outlier_detector = UnivariateOutlierDetector(ZScoreOutlierDetection(threshold=3))


    # Detect and handle outliers dynamically using the selected method (e.g., "iqr")
    for column in df_numeric_train.columns:
        # Handle outliers for each column and update the original DataFrame
        df_train_imputed[column] = outlier_detector.handle_outliers(df_numeric_train[column], method="zscore")
    for column in df_numeric_test.columns:
        # Handle outliers for each column and update the original DataFrame
       test_df_imputed[column] = outlier_detector.handle_outliers(df_numeric_test[column], method="zscore")    
   # multivariate outlier treatement 
    df_numeric =df_train_imputed.select_dtypes(include=[np.number]).dropna()
    numerical_columns = df_numeric.columns.tolist()
    df_numeric =test_df_imputed.select_dtypes(include=[np.number]).dropna()
    numerical_columns = df_numeric.columns.tolist()
    # Example: Apply One-Class SVM strategy to detect outliers
    svm_strategy = OneClassSVMStrategy()
    outlier_detector_svm = OutlierDetection(strategy=svm_strategy, df=df_train_imputed, numerical_columns=numerical_columns)
    
    # Detect outliers with One-Class SVM
    df_with_outliers_svm = outlier_detector_svm.detect_outliers(nu=0.1)

    # Handle and visualize One-Class SVM method-wise:
    df_train = outlier_detector_svm.handle_outliers(method='remove', outlier_column='Outlier_SVM')
    
    outlier_detector_svm = OutlierDetection(strategy=svm_strategy, df=test_df_imputed, numerical_columns=numerical_columns)
    
    # Detect outliers with One-Class SVM
    df_with_outliers_svm = outlier_detector_svm.detect_outliers(nu=0.1)

    # Handle and visualize One-Class SVM method-wise:
    df_test = outlier_detector_svm.handle_outliers(method='remove', outlier_column='Outlier_SVM')
    
    # Create context with the chosen strategy
    encoding_strategy=OneHotEncoding(features = ['PROP_HEADING','CITY'])
    context =Feature_Engineering(strategy=encoding_strategy)

    df_train_processed = context.apply_transformation(df_train )# Define custom mappings for each feature
    df_test_processed = context.apply_transformation(df_test)# Define custom mappings for each feature
    
    # Define categorical features
    categorical_features = ['AGE', 'FLOOR_NUM', 'FURNISH', 'PREFERENCE', 'AMENITY_LUXURY', 'FEATURES_LUXURY']

    # Define custom mappings for each feature
    mapping_list = [
        {'col': 'AGE', 'mapping': {'relative new property': 0, 'moderate old property': 1, 'old property': 2}},
        {'col': 'FLOOR_NUM', 'mapping': {'Unknown': 0, 'Basement': 1, 'Low-rise': 2, 'Ground Level': 3, 'High-rise': 4, 'Mid-rise': 5, 'Very High-rise': 6}},
        {'col': 'FURNISH', 'mapping': {'Unfurnished': 0, 'Semi-Furnished': 1, 'Fully Furnished': 2, 'Partially Furnished': 3}},
        {'col': 'AMENITY_LUXURY', 'mapping': {'Low': 0, 'Medium': 1, 'High': 2}},
        {'col': 'FEATURES_LUXURY', 'mapping': {'Low': 0, 'High': 1, 'Medium': 2}}
    ]
    ordinal_encoding = OrdinalEncoding(
            features=categorical_features,
            mapping=mapping_list
        )

    # Create context with the encoding strategy
    context = Feature_Engineering(strategy=ordinal_encoding)

    # Ensure the processed directory exists
    processed_dir = root_path / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured the processed directory exists at: {processed_dir}")

    # Apply the transformation
    df_train_processed1 = context.apply_transformation(df_train_processed)
    df_test_processed1 = context.apply_transformation(df_test_processed)

    # Save the processed data
    save(df_train_processed1, processed_dir / 'train_processed.csv')
    save(df_test_processed1, processed_dir / 'test_processed.csv')


   

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred during the execution of the main function.")
        
        