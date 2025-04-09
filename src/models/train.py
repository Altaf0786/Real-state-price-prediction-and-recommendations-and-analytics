import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, StackingRegressor, VotingRegressor
)
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import TransformedTargetRegressor

# Logger setup
root_path = Path(__file__).parent.parent.parent
log_file_path = root_path / 'src/log/train.log'
logger = logging.getLogger("train_model_logger")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Function to load parameters from params.yaml
def load_params(params_file='params.yaml'):
    try:
        with open(params_file, 'r') as file:
            params = yaml.safe_load(file)
        logger.info(f"Loaded parameters from {params_file}")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters from {params_file}", exc_info=True)
        raise

# Abstract base class for model strategy
class ModelBuildingStrategy(ABC):
    def __init__(self, model_type, transformer=None, **kwargs):
        self.model_type = model_type
        self.transformer = transformer
        self.kwargs = kwargs

    @abstractmethod
    def create_model(self) -> object:
        pass

# Concrete factory for model instantiation
class ModelFactory:
    def create_model(self, model_type, **kwargs):
        try:
            if model_type == 'linear':
                return LinearRegression(**kwargs)
            elif model_type == 'ridge':
                return Ridge(**kwargs)
            elif model_type == 'lasso':
                return Lasso(**kwargs)
            elif model_type == 'elasticnet':
                return ElasticNet(**kwargs)
            elif model_type == 'mlpreg':
                return MLPRegressor(**kwargs)
            elif model_type == 'tree':
                return DecisionTreeRegressor(**kwargs)
            elif model_type == 'knnreg':
                return KNeighborsRegressor(**kwargs)
            elif model_type == 'rf':
                kwargs.setdefault("n_estimators", 100)
                kwargs.setdefault("random_state", 42)
                return RandomForestRegressor(**kwargs)
            elif model_type == 'gradreg':
                return GradientBoostingRegressor(**kwargs)
            elif model_type == 'extratree':
                return ExtraTreesRegressor(**kwargs)
            elif model_type == 'adareg':
                return AdaBoostRegressor(**kwargs)
            elif model_type == 'xgboostreg':
                kwargs.setdefault("n_estimators", 100)
                return XGBRegressor(**kwargs)
            elif model_type == 'lgmreg':
                kwargs.setdefault("n_estimators", 100)
                return lgb.LGBMRegressor(**kwargs)
            elif model_type == 'cataboostreg':
                kwargs.setdefault("iterations", 100)
                return CatBoostRegressor(verbose=0, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            logger.error(f"Error creating model for type: {model_type}", exc_info=True)
            raise

# Strategy implementation using the factory
class DynamicModelStrategy(ModelBuildingStrategy):
    def create_model(self) -> object:
        try:
            model_factory = ModelFactory()
            base_model = model_factory.create_model(self.model_type, **self.kwargs)
            if self.transformer:
                return TransformedTargetRegressor(regressor=base_model, transformer=self.transformer)
            return base_model
        except Exception as e:
            logger.error(f"Failed to create model with strategy {self.model_type}", exc_info=True)
            raise

# Model builder class using strategy
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def build_model(self) -> object:
        try:
            return self._strategy.create_model()
        except Exception as e:
            logger.error("Model building failed.", exc_info=True)
            raise

# Main entry point
if __name__ == "__main__":
    try:
        params_file = root_path / 'params.yaml'
        params = load_params(params_file)

        train_data_path = root_path / "data" / "processed" / "train_processed.csv"
        model_save_dir = root_path / "models"
        model_save_dir.mkdir(exist_ok=True)

        try:
            train_data = pd.read_csv(train_data_path)
            logger.info("Train dataset loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load train data from {train_data_path}", exc_info=True)
            raise

        target_column = "PRICE"
        try:
            X_train = train_data.drop(columns=[target_column, 'PREFERENCE'])
            y_train = train_data[target_column]
        except KeyError as e:
            logger.error("Required columns not found in training data.", exc_info=True)
            raise

        model_type = params['model']['type']
        model_kwargs = {k: v for k, v in params['model'].items() if k != 'type'}

        strategy = DynamicModelStrategy(model_type=model_type, **model_kwargs)
        model_builder = ModelBuilder(strategy=strategy)
        model = model_builder.build_model()

        try:
            model.fit(X_train, y_train)
            logger.info("Model training completed successfully.")
        except Exception as e:
            logger.error("Error during model training.", exc_info=True)
            raise

        model_save_path = model_save_dir / f"{model_type}_model.joblib"
        try:
            joblib.dump(model, model_save_path)
            logger.info(f"Model saved at: {model_save_path}")
        except Exception as e:
            logger.error(f"Failed to save model at {model_save_path}", exc_info=True)
            raise

    except Exception as e:
        logger.critical("Training pipeline failed completely.", exc_info=True)
