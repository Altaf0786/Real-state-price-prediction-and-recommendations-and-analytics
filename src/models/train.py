import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, VotingRegressor, StackingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from abc import ABC, abstractmethod
import logging

from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

# --------------------- Logging Setup ---------------------
root_path = Path(__file__).parent.parent.parent
log_file_path = root_path / 'src/log/train.log'
logger = logging.getLogger("train_model_logger")
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

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

# --------------------- Utility Functions ---------------------
def load_params(params_file='params.yaml'):
    try:
        with open(params_file, 'r') as file:
            params = yaml.safe_load(file)
        logger.info(f"Loaded parameters from {params_file}")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters from {params_file}", exc_info=True)
        raise

def save_model(model, save_dir: Path, model_name: str):
    try:
        save_location = save_dir / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(value=model, filename=save_location)
        logger.info(f"Model saved at {save_location}")
    except Exception as e:
        logger.error(f"Failed to save model at {save_location}", exc_info=True)
        raise

def make_X_and_y(data: pd.DataFrame, target_column: str):
    try:
        X = data.drop(columns=[target_column, 'PREFERENCE'])
        y = data[target_column]
        return X, y
    except KeyError as e:
        logger.error(f"Required columns not found in data. Missing: {e}", exc_info=True)
        raise

# --------------------- Factory ---------------------
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

# --------------------- Strategy Design ---------------------
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def create_model(self) -> object:
        pass

class SimpleModelStrategy(ModelBuildingStrategy):
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.factory = ModelFactory()

    def create_model(self):
        return self.factory.create_model(self.model_type, **self.kwargs)

class TransformedRegressorStrategy(ModelBuildingStrategy):
    def __init__(self, base_model):
        self.base_model = base_model

    def create_model(self):
        transformer = PowerTransformer()
        return TransformedTargetRegressor(regressor=self.base_model, transformer=transformer)

class VotingEnsembleStrategy(ModelBuildingStrategy):
    def __init__(self, model_list):
        self.model_list = model_list
        self.factory = ModelFactory()

    def create_model(self):
        estimators = [(name, self.factory.create_model(name)) for name in self.model_list]
        return VotingRegressor(estimators=estimators)

class StackingEnsembleStrategy(ModelBuildingStrategy):
    def __init__(self, model_list, final_estimator_name):
        self.model_list = model_list
        self.final_estimator_name = final_estimator_name
        self.factory = ModelFactory()

    def create_model(self):
        base_models = [(name, self.factory.create_model(name)) for name in self.model_list]
        final_estimator = self.factory.create_model(self.final_estimator_name)
        return StackingRegressor(estimators=base_models, final_estimator=final_estimator)

class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def build_model(self) -> object:
        return self._strategy.create_model()

# --------------------- Main Train Logic ---------------------
if __name__ == "__main__":
    try:
        params = load_params(root_path / "params.yaml")

        train_data_path = root_path / "data" / "processed" / "train_processed.csv"
        train_data = pd.read_csv(train_data_path)
        logger.info("Train dataset loaded successfully.")

        X_train, y_train = make_X_and_y(train_data, "PRICE")

        model_save_dir = root_path / "models"
        model_save_dir.mkdir(exist_ok=True)

        for model_cfg in params["models"]:
            name = model_cfg["name"]

            logger.info(f"Training model: {name}")

            if name == "linear":
                strategy = SimpleModelStrategy("linear")
                model = ModelBuilder(strategy).build_model()
                model.fit(X_train, y_train)
                save_model(model, model_save_dir, f"{name}_model.joblib")

            elif name == "transformreg":
                experiments = model_cfg.get("experiments", [])
                for idx, exp_cfg in enumerate(experiments):
                    base_model_type = exp_cfg.get("base_model", "rf")
                    factory = ModelFactory()

                    if base_model_type == "voting":
                        strategy = VotingEnsembleStrategy(model_list=exp_cfg["models"])
                        base_model = strategy.create_model()
                    elif base_model_type == "stacking":
                        strategy = StackingEnsembleStrategy(
                            model_list=exp_cfg["models"],
                            final_estimator_name=exp_cfg["final_estimator"]
                        )
                        base_model = strategy.create_model()
                    else:
                        base_model = factory.create_model(base_model_type)

                    strategy = TransformedRegressorStrategy(base_model)
                    model = ModelBuilder(strategy).build_model()
                    model.fit(X_train, y_train)

                    sub_model_name = f"{name}_{base_model_type}_{idx}"
                    save_model(model, model_save_dir, f"{sub_model_name}_model.joblib")
                    logger.info(f"Model '{sub_model_name}' trained and saved.")

            elif name == "voting":
                strategy = VotingEnsembleStrategy(model_list=model_cfg["models"])
                model = ModelBuilder(strategy).build_model()
                model.fit(X_train, y_train)
                save_model(model, model_save_dir, f"{name}_model.joblib")

            elif name == "stacking":
                strategy = StackingEnsembleStrategy(
                    model_list=model_cfg["models"],
                    final_estimator_name=model_cfg["final_estimator"]
                )
                model = ModelBuilder(strategy).build_model()
                model.fit(X_train, y_train)
                save_model(model, model_save_dir, f"{name}_model.joblib")

            else:
                strategy = SimpleModelStrategy(name)
                model = ModelBuilder(strategy).build_model()
                model.fit(X_train, y_train)
                save_model(model, model_save_dir, f"{name}_model.joblib")

            logger.info(f"Model '{name}' training completed.")

    except Exception as e:
        logger.critical("Training pipeline failed completely.", exc_info=True)
