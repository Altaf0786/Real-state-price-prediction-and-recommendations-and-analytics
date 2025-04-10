import mlflow
import dagshub
import json
import logging
import sys
import traceback
from pathlib import Path
from mlflow import MlflowClient

# Helper function to create and configure a logger
def create_logger(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # File Handler (if log file path is provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    return logger


# Create logger for model registration
root_path = Path(__file__).parent.parent.parent
log_file_path = root_path / 'src/log/evaluation.log'
logger = create_logger("register_model", log_file_path)

# Initialize Dagshub and MLflow with error handling
def initialize_dagshub_mlflow():
    try:
        dagshub.init(repo_owner='Altaf0786',
                     repo_name='Real-state-price-prediction-and-recommendations-and-analytics',
                     mlflow=True)
        mlflow.set_tracking_uri("https://dagshub.com/Altaf0786/Real-state-price-prediction-and-recommendations-and-analytics.mlflow")
        mlflow.set_experiment("DVC Pipeline")
        logger.info("Dagshub and MLflow initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Dagshub or MLflow: {e}")
        logger.error(traceback.format_exc())  # Logs the stack trace
        sys.exit(1)


# Load model information from JSON file
def load_model_information(file_path):
    try:
        with open(file_path) as f:
            return json.load(f)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        sys.exit(1)


# Register model and transition to Staging
def register_and_transition_model(run_info):
    try:
        run_id = run_info["run_id"]
        model_name = run_info["model_name"]
        model_registry_path = f"runs:/{run_id}/{model_name}"

        # Register model in the MLflow registry
        model_version = mlflow.register_model(model_uri=model_registry_path, name=model_name)

        # Get the registered model version and name
        registered_model_version = model_version.version
        registered_model_name = model_version.name
        logger.info(f"The latest model version in model registry is {registered_model_version}")

        # Transition model to the 'Staging' stage
        client = MlflowClient()
        client.transition_model_version_stage(
            name=registered_model_name,
            version=registered_model_version,
            stage="Staging"
        )

        logger.info("Model pushed to Staging stage")
    except Exception as e:
        logger.error(f"An error occurred during model registration: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


# Main function to execute the script
def main():
    initialize_dagshub_mlflow()

    # Root path for locating the run information file
    root_path = Path(__file__).parent.parent.parent
    run_info_path = root_path / "run_information.json"
    
    # Load model information
    run_info = load_model_information(run_info_path)
    
    # Register the model and transition to the Staging stage
    register_and_transition_model(run_info)


if __name__ == "__main__":
    main()
