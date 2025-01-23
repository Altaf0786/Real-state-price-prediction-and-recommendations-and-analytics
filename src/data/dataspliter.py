import sys
import os
import logging
from pathlib import Path
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from helpercode.data_spliter import DataSplitter, SimpleTrainTestSplitStrategy
# Setup logging
logger = logging.getLogger("DataSpliter")  # Using a specific logger name
logger.setLevel(logging.DEBUG)
# creat root path 
root_path = Path(__file__).parent.parent.parent

# Console Handler (for output to console)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set the level for console output
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Include name in the format
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
file_handler_path=root_path/'src/log/dataspliteer.log'
# File Handler (for output to log file)
file_handler = logging.FileHandler(file_handler_path)  # Logs to 'app.log'
file_handler.setLevel(logging.DEBUG)  # Set the level for file output
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Include name in the format
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Example usage


def load(file_path):
    try:
        logger.info('data load start')
        return pd.read_csv(file_path)
        logger.info('data load complete')
    except Exception as e:
        logger.error("data load does not complete: %s", e)
def save(dataframes, output_dir):
    """
    Save dataframes to CSV files in the specified directory.

    Args:
        dataframes (dict): A dictionary of {name: dataframe} pairs.
        output_dir (Path): Path to the directory where files will be saved.
    """
    try:
        logger.info('Data save start')
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each dataframe as a CSV file
        for name, df in dataframes.items():
            output_file_path = output_dir / f"{name}.csv"
            df.to_csv(output_file_path, index=False)
            logger.info(f"Saved {name} to {output_file_path}")
        
        logger.info('Data save complete')
    except Exception as e:
        logger.error("Data save failed: %s", e)

if __name__ == "__main__":
    logger.info('data spliter start')
    try:
        root_path = Path(__file__).parent.parent.parent
        input_file_path = root_path / "data" / "cleaned" / "House_cleaned.csv"
        logger.info('data spliter start')
        # Example dataframe (replace with actual data loading)
        df = load(input_file_path)
        # Initialize data splitter with a specific strategy
        data_splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42))
        # Split the data
        X_train, X_test, y_train, y_test = data_splitter.split(df, target_column="PRICE")
        
        # Save the data
        output_dir = root_path / "data" / "processed"
        save({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }, output_dir)

        # Save the data
        save({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}, output_dir)
        
        
        
        
        logger.info('data spliter complete')
    except Exception as e:
        logger.error("data spliter does not complete: %s", e)        