import logging
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from helpercode.data_ingetion import DataIngestorFactory, ZipDataIngestor, CSVDataIngestor, ExcelDataIngestor, JSONDataIngestor
# Ensure the log directory exists using pathlib




  # Replace with the actual module path

# Setup logging
logger = logging.getLogger("DataIngestorApp")  # Using a specific logger name
logger.setLevel(logging.DEBUG)

# Console Handler (for output to console)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set the level for console output
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Include name in the format
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File Handler (for output to log file)
file_handler = logging.FileHandler('/Users/mdaltafshekh/real-state-price-prediction-and-recommendations-and-analytics/src/log/dataingetion.log')  # Logs to 'app.log'
file_handler.setLevel(logging.DEBUG)  # Set the level for file output
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Include name in the format
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Example usage
if __name__ == "__main__":
    try:
        logger.info('dataingetion start')
        # Update this path to point to your file
        file_path = Path("/Users/mdaltafshekh/Downloads/House_data.csv")  # Replace with your file path
        output_dir = Path("/Users/mdaltafshekh/real-state-price-prediction-and-recommendations-and-analytics/data/raw")  # Using the same directory for extraction and output
        
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
