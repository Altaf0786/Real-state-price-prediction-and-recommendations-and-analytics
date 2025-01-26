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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, StackingRegressor, VotingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load parameters from params.yaml
def load_params(params_file='params.yaml'):
    """
    Load parameters from a YAML file.
    
    Args:
        params_file (str): Path to the YAML file. Default is 'params.yaml'.
    
    Returns:
        dict: Dictionary of parameters.
    """
    try:
        with open(params_file, 'r') as file:
            params = yaml.safe_load(file)
        logger.info(f"Loaded parameters from {params_file}")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters from {params_file}: {e}")
        return {}

# Abstract Base Class for Regressor Strategies
class ModelBuildingStrategy(ABC):
    def __init__(self, model_type, transformer=None, **kwargs):
        self.model_type = model_type
        self.transformer = transformer
        self.kwargs = kwargs

    @abstractmethod
    def create_model(self) -> object:
        pass

# Concrete Strategy Factory
class ModelFactory:
    def create_model(self, model_type, **kwargs):
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
            return RandomForestRegressor(n_estimators=100, random_state=42, **kwargs)
        elif model_type == 'gradreg':
            return GradientBoostingRegressor(**kwargs)
        elif model_type == 'extratree':
            return ExtraTreesRegressor(**kwargs)
        elif model_type == 'adareg':
            return AdaBoostRegressor(**kwargs)
        elif model_type == 'xgboostreg':
            return XGBRegressor(n_estimators=100, **kwargs)
        elif model_type == 'lgmreg':
            return lgb.LGBMRegressor(n_estimators=100, **kwargs)
        elif model_type == 'cataboostreg':
            return CatBoostRegressor(iterations=100, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Dynamic Model Strategy
class DynamicModelStrategy(ModelBuildingStrategy):
    def create_model(self) -> object:
        model_factory = ModelFactory()
        base_model = model_factory.create_model(self.model_type, **self.kwargs)

        if self.transformer:
            return TransformedTargetRegressor(regressor=base_model, transformer=self.transformer)
        else:
            return base_model

# Model Builder Class
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def build_model(self) -> object:
        return self._strategy.create_model()

if __name__ == "__main__":
    # Load parameters from YAML
    params = load_params(params_file='params.yaml')

    # Set paths for data and model saving
    train_data_path = "/content/train_processed.csv"
    model_save_dir = "/content/models"
    Path(model_save_dir).mkdir(exist_ok=True)

    # Load data
    train_data = pd.read_csv(train_data_path)
    logger.info("Train dataset loaded successfully.")

    target_column = "PRICE"
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]

    # Preprocessing pipeline
    categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
    preprocess_pipeline = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X_train.select_dtypes(include=['int64', 'float64']).columns),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy=params['preprocessing']['imputation_strategy'])),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        ]
    )

    # Apply preprocessing transformations
    X_train_processed = preprocess_pipeline.fit_transform(X_train)

    # Get model configuration from params.yaml
    model_type = params['model']['type']
    model_kwargs = {key: value for key, value in params['model'].items() if key != 'type'}

    # Create strategy and model builder
    strategy = DynamicModelStrategy(model_type=model_type, **model_kwargs)
    model_builder = ModelBuilder(strategy=strategy)
    model = model_builder.build_model()

    # Fit the model
    model.fit(X_train_processed, y_train)
    logger.info("Model training completed.")

    # Save the trained model
    model_save_path = Path(model_save_dir) / f"{model_type}_model.joblib"
    joblib.dump(model, model_save_path)
    logger.info(f"Model saved to {model_save_path}")
