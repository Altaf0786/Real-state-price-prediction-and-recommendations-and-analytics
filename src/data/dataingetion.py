import logging
from pathlib import Path
import sys
import os

import pandas as pd
root_path = Path(__file__).resolve().parent.parent.parent

# Add root path to sys.path if not already present
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
from helpercode.data_ingetion import DataIngestorFactory, ZipDataIngestor, CSVDataIngestor, ExcelDataIngestor, JSONDataIngestor
from sklearn.model_selection import train_test_split
# Ensure the log directory exists using pathlib

# Setup logging
logger = logging.getLogger("DataIngestorApp")  # Using a specific logger name
logger.setLevel(logging.DEBUG)


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



def split_data(data: pd.DataFrame, test_size: float, random_state: int):
    train_data, test_data = train_test_split(data, 
                                             test_size=test_size, 
                                             random_state=random_state)
    
    return train_data, test_data

# Example usage
if __name__ == "__main__":
    try:
        logger.info('dataingetion start')
        # Update this path to point to your file
        
        #file_path = Path("/Users/mdaltafshekh/Downloads/House_data.csv") 
        # Input file path
        file_path = Path("/Users/mdaltafshekh/Desktop/House_data.csv")

        # Output path dynamically created using root_path
        output_dir = root_path / "data" / "raw"

        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        # Get the file extension and create an appropriate data ingestor
        file_extension = file_path.suffix
        data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

        # Ingest the data and save it
        dataframes = data_ingestor.ingest(file_path)
        data_ingestor.save(dataframes, output_dir)
        logger.info('dataingetion complete')
        # Print the first few rows of each DataFrame
        
    except Exception as e:
        logger.error("data ingetion does not complete: %s", e)
