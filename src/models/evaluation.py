import pandas as pd
import joblib
import logging
import mlflow
import dagshub
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json
import sys
import traceback
import yaml

# -------------------- Configuration --------------------
TARGET = "PRICE"
MODEL_FILENAME_DEFAULT = "linear_model.joblib"

# -------------------- Logging Setup --------------------
root_path = Path(__file__).parent.parent.parent
log_file_path = root_path / 'src/log/evaluation.log'

logger = logging.getLogger("model evaluation logger")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# -------------------- Dagshub & MLflow Setup --------------------
try:
    dagshub.init(repo_owner='Altaf0786',
                 repo_name='Real-state-price-prediction-and-recommendations-and-analytics',
                 mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/Altaf0786/Real-state-price-prediction-and-recommendations-and-analytics.mlflow")
    mlflow.set_experiment("DVC Pipeline")
    logger.info("Dagshub and MLflow initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Dagshub or MLflow: {e}")
    sys.exit(1)

# -------------------- Utility Functions --------------------
def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}")
        return df
    except Exception as e:
        logger.exception(f"Failed to load data from {data_path}: {e}")
        return None

def make_X_and_y(data: pd.DataFrame, target_column: str):
    try:
        X = data.drop(columns=[target_column, 'PREFERENCE'])
        y = data[target_column]
        return X, y
    except Exception as e:
        logger.exception(f"Failed to split data into X and y: {e}")
        return None, None

def load_model(model_path: Path):
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.info("Available models:")
        for m in model_path.parent.glob("*.joblib"):
            logger.info(f" - {m.name}")
        sys.exit("Model loading failed. Exiting...")
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.exception(f"Error loading model from {model_path}: {e}")
        return None

def save_model_info(save_json_path, run_id, artifact_path, model_name):
    try:
        info_dict = {
            "run_id": run_id,
            "artifact_path": artifact_path,
            "model_name": model_name
        }
        with open(save_json_path, "w") as f:
            json.dump(info_dict, f, indent=4)
        logger.info(f"Saved model info to {save_json_path}")
    except Exception as e:
        logger.exception(f"Error saving model information: {e}")

def get_model_filename_from_params(params_path: Path, default: str = MODEL_FILENAME_DEFAULT) -> str:
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
            return params.get("evaluation", {}).get("model_name", default)
    except Exception as e:
        logger.warning(f"Couldn't read model name from params.yaml, using default '{default}': {e}")
        return default

# -------------------- Main --------------------
if __name__ == "__main__":
    try:
        # Paths
        params_path = root_path / "params.yaml"
        model_name = get_model_filename_from_params(params_path)
        model_path = root_path / "models" / model_name
        train_data_path = root_path / "data" / "processed" / "train_processed.csv"
        test_data_path = root_path / "data" / "processed" / "test_processed.csv"
        save_json_path = root_path / "run_information.json"

        # Load data
        train_data = load_data(train_data_path)
        test_data = load_data(test_data_path)
        if train_data is None or test_data is None:
            sys.exit("Failed to load training/testing data.")

        X_train, y_train = make_X_and_y(train_data, TARGET)
        X_test, y_test = make_X_and_y(test_data, TARGET)

        # Load model
        model = load_model(model_path)
        if model is None:
            sys.exit("Model loading failed.")

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
        mean_cv_score = -cv_scores.mean()

        # Log to MLflow
        with mlflow.start_run() as run:
            mlflow.set_tag("model", model_name)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "mean_cv_score": mean_cv_score
            })
            mlflow.log_metrics({f"CV fold {i}": -score for i, score in enumerate(cv_scores)})

            train_data_input = mlflow.data.from_pandas(train_data, targets=TARGET)
            test_data_input = mlflow.data.from_pandas(test_data, targets=TARGET)
            mlflow.log_input(dataset=train_data_input, context="training")
            mlflow.log_input(dataset=test_data_input, context="validation")

            model_signature = mlflow.models.infer_signature(
                model_input=X_train.sample(20, random_state=42),
                model_output=model.predict(X_train.sample(20, random_state=42))
            )
            mlflow.sklearn.log_model(model, model_name, signature=model_signature)

            mlflow.log_artifact(model_path)

            # Save metadata
            run_id = run.info.run_id
            artifact_uri = mlflow.get_artifact_uri()
            save_model_info(save_json_path, run_id, artifact_uri, model_name)

        logger.info("Evaluation and logging completed successfully.")

    except Exception as e:
        logger.error("An unexpected error occurred during model evaluation.")
        logger.error(traceback.format_exc())
        sys.exit(1)
