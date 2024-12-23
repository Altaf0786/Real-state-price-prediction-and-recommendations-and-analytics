import logging
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'helpercode')))

from data_ingetion import DataIngestorFactory


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
file_handler = logging.FileHandler('src/log/data_ingetion.log')  # Logs to 'app.log'
file_handler.setLevel(logging.DEBUG)  # Set the level for file output
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Include name in the format
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Example usage
if __name__ == "__main__":
    try:
        # Update this path to point to your file
        file_path = Path("/Users/mdaltafshekh/Downloads/House_data.csv")  # Replace with your file path
        output_dir = Path("data/raw")  # Using the same directory for extraction and output

        # Get the file extension and create an appropriate data ingestor
        file_extension = file_path.suffix
        data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

        # Ingest the data and save it
        dataframes = data_ingestor.ingest(file_path)
        data_ingestor.save(dataframes, output_dir)

        # Print the first few rows of each DataFrame
        for name, df in dataframes.items():
            logger.info(f"DataFrame for {name}:")
            logger.info(f"{df.head()}")

    except Exception as e:
        logger.error("An error occurred: %s", e)
