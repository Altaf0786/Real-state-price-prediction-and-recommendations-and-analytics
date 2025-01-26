import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import logging
from pathlib import Path

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    test_data_path = "/content/test_processed.csv"
    model_save_dir = "/content/models"

    # Load Data
    test_data = pd.read_csv(test_data_path)
    logger.info("Test dataset loaded successfully.")

    target_column = "PRICE"
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    # Preprocessing (same as in train.py)
    categorical_cols = [col for col in X_test.columns if X_test[col].dtype == 'object']
    preprocess_pipeline = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X_test.select_dtypes(include=['int64', 'float64']).columns),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        ]
    )

    # Apply preprocessing transformations
    X_test_processed = preprocess_pipeline.fit_transform(X_test)

    # Load the trained model
    model_type = 'rf'  # Ensure this matches the model saved in train.py
    model_load_path = Path(model_save_dir) / f"{model_type}_model.joblib"
    model = joblib.load(model_load_path)
    logger.info(f"Model loaded from {model_load_path}")

    # Make predictions
    test_predictions = model.predict(X_test_processed)

    # Calculate RMSE, MSE, and R² score for test data
    test_mse = mean_squared_error(y_test, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, test_predictions)

    # Log results
    logger.info(f"Test RMSE: {test_rmse}")
    logger.info(f"Test MSE: {test_mse}")
    logger.info(f"Test R²: {test_r2}")

    # Output the results
    print(f"Test RMSE: {test_rmse}")
    print(f"Test MSE: {test_mse}")
    print(f"Test R²: {test_r2}")
