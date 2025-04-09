from os import pipe
import pandas as pd
import ast
import re
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_data(file_path):
    return pd.read_csv(file_path)

def save_data(df, file_path):
    df.to_csv(file_path, index=False)

# Define the function to transform 'location' and 'MAP_DETAILS' columns
def make_feature(df):
    # Ensure the 'location' column contains dictionaries (if it contains string representations, we convert them)
    df['location'] = df['location'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['LOCALITY_NAME'] = df['location'].apply(lambda x: x.get('LOCALITY_NAME') if isinstance(x, dict) else None)
    df['ADDRESS'] = df['location'].apply(lambda x: x.get('ADDRESS') if isinstance(x, dict) else None)

    df['MAP_DETAILS'] = df['MAP_DETAILS'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['LATITUDE'] = df['MAP_DETAILS'].apply(lambda x: x.get('LATITUDE') if isinstance(x, dict) else None)
    df['LONGITUDE'] = df['MAP_DETAILS'].apply(lambda x: x.get('LONGITUDE') if isinstance(x, dict) else None)
    return df

# Function to create binary columns for amenities


# Function to categorize property type based on 'PROP_HEADING'
def categorize_property(df):
    def categorize(property_name):
        if pd.isna(property_name):  # Handle NaN values
            return 'Unknown'
        property_name = str(property_name).lower()
        if 'independent' in property_name and 'floor' in property_name:
            return 'Independent Floor'
        elif 'house' in property_name:
            return 'House'
        elif 'flat' in property_name or 'bhk' in property_name:
            return 'Apartment'
        elif 'plot' in property_name or 'land' in property_name:
            return 'Plot'
        else:
            return 'Other'

    df['PROP_HEADING'] = df['PROP_HEADING'].apply(categorize)
    return df


# Function to convert PRICE to 'Cr'
def convert_to_cr(df):
    def convert(value):
        range_pattern = r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*(Cr|L)"
        single_value_pattern = r"(\d+\.?\d*)\s*(Cr|L)"

        match = re.match(range_pattern, value)
        if match:
            start, end, unit = match.groups()
            start = float(start)
            end = float(end)
            if unit == "Cr":
                return (start + end) / 2
            elif unit == "L":
                return ((start / 100) + (end / 100)) / 2
        match = re.match(single_value_pattern, value)
        if match:
            number, unit = match.groups()
            number = float(number)
            if unit == "L":
                return number / 100
            elif unit == "Cr":
                return number
        return None

    df['PRICE'] = df['PRICE'].apply(convert)
    return df

# Function to convert AREA to numeric
def convert_to_numeric(df):
    def convert(value):
        range_match = re.match(r"(\d+)-(\d+)\s*sq\.ft\.$", value)
        if range_match:
            low, high = map(int, range_match.groups())
            return (low + high) / 2
        single_match = re.match(r"(\d+)\s*sq\.ft\.$", value)
        if single_match:
            return int(single_match.group(1))
        return None

    df['AREA'] = df['AREA'].apply(convert)
    return df

# Function to categorize the 'AGE' column
def categorize_age(df):
    def categorize(age):
        if age <= 2:
            return 'Relative New Property'
        elif 3 <= age <= 5:
            return 'Moderate Old Property'
        else:
            return 'Old Property'

    df['AGE'] = df['AGE'].apply(categorize)
    return df




def create_binary_columns(df):
    # Define the categories that will be used for binary columns
    categories = ['MetroStation', 'Shopping', 'Connectivity', 'Education', 'Hospital',
                  'Airport', 'RailwayStation', 'OfficeComplex', 'Hotel', 'AmusementPark',
                  'GolfCourse', 'Stadium', 'ReligiousPlace', 'ATM', 'Parking', 'BusDepot', 'Miscellaneous']

    def process_row(row):
        if pd.isna(row):  # Check for NaN values
            return {category: 0 for category in categories}  # Return all 0s if NaN

        try:
            # Extract category names from the row
            row_data = ast.literal_eval(row)
            row_categories = [item['category'] for item in row_data if 'category' in item]

            # Return binary values for each category (1 if category exists, else 0)
            return {category: (1 if category in row_categories else 0) for category in categories}
        except (ValueError, SyntaxError):  # Handle any issues with eval
            return {category: 0 for category in categories}  # Return all 0s if evaluation fails

    # Apply the function to the 'FORMATTED_LANDMARK_DETAILS' column and create binary columns
    binary_columns = df['FORMATTED_LANDMARK_DETAILS'].apply(process_row)

    # Convert the result into a DataFrame and join with the original DataFrame
    binary_df = pd.DataFrame(list(binary_columns))
    return pd.concat([df, binary_df], axis=1)

# Function to handle missing values and fill them
def fill_missing_values(df):
    df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').apply(lambda x: x.str.lower())
    return df

def amenity_luxury(df, column_name, new_column_name):
    """
    Calculates the sum of amenities in a specified column of a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column_name (str): The column containing amenities as comma-separated values.
    - new_column_name (str): The name of the new column to store the sums.

    Returns:
    - pd.DataFrame: The DataFrame with the new column added.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    df[column_name] = df[column_name].fillna('')  # Replace NaN with an empty string
    df[new_column_name] = df[column_name].apply(
        lambda x: sum(map(int, x.split(','))) if isinstance(x, str) and x else 0
    )
    return df

def classify_amenity(value):
    if value <= 200:
        return 'Low'
    elif value <= 500:
        return 'Medium'
    else:
        return 'High'


def FEATURES_LUXURY(df, column_name, new_column_name):
    """
    Create a new column with the sum of feature values in each row for the specified column.
    Non-numeric values are ignored.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column_name (str): The column containing the comma-separated feature values.
    - new_column_name (str): The name of the new column where the sum will be stored.

    Returns:
    - pd.DataFrame: The original DataFrame with an added column containing the sum of features.
    """
    # Replace NaN values with empty strings
    df[column_name] = df[column_name].fillna('')

    # Apply the transformation to sum the feature values
    df[new_column_name] = df[column_name].apply(
        lambda x: sum(
            int(val) for val in x.split(',') if val.isdigit()  # Only sum numeric values
        ) if isinstance(x, str) and x else 0
    )

    return df

def classify_features(value):
    if value <= 200:
        return 'Low'
    elif value <= 500:
        return 'Medium'
    else:
        return 'High'

# Define categorization logic
def categorize_age(df):
    def categorize(age):
        if pd.isna(age):
            return 'Unknown Age'  # Handle missing or invalid ages
        if age <= 2:
            return 'Relative New Property'
        elif 3 <= age <= 5:
            return 'Moderate Old Property'
        else:
            return 'Old Property'

    # Ensure AGE is numeric before applying categorization
    df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')  # Converts non-numeric to NaN
    df['AGE'] = df['AGE'].apply(categorize)
    return df


def categorize_furnish(value):
    if value == 4:
        return 'Fully Furnished'
    elif value == 0:
        return 'Unfurnished'
    elif value == 2:
        return 'Semi-Furnished'
    elif value == 1:
        return 'Partially Furnished'
    else:
        return 'Unknown'


def categorize_floor(floor):
    try:
        # Attempt to convert to integer
        floor = int(floor)
        if floor == 0:
            return 'Ground Level'
        elif 1 <= floor <= 5:
            return 'Low-rise'
        elif 6 <= floor <= 15:
            return 'Mid-rise'
        elif 16 <= floor <= 30:
            return 'High-rise'
        elif floor > 30:
            return 'Very High-rise'
    except ValueError:
        # Handle non-numeric values
        floor = str(floor).upper()
        if floor == 'G':
            return 'Ground Level'
        elif floor == 'B':
            return 'Basement'
        elif floor == 'L':
            return 'Ground Level'
    return 'Unknown'

def preprocess_data(df):
    columns_order = [
        'PROP_HEADING', 'FACING', 'TRANSACT_TYPE', 'AREA', 'CITY',
        'BEDROOM_NUM', 'BALCONY_NUM', 'FURNISH', 'OWNTYPE', 'PREFERENCE', 'AGE',
        'TOTAL_FLOOR', 'FLOOR_NUM', 'TOTAL_LANDMARK_COUNT', 'MetroStation', 'Shopping',
        'Connectivity', 'Education', 'Hospital', 'Airport', 'RailwayStation', 'OfficeComplex',
        'Hotel', 'AmusementPark', 'GolfCourse', 'Stadium', 'ReligiousPlace', 'ATM', 'Parking',
        'BusDepot', 'AMENITY_LUXURY', 'FEATURES_LUXURY', 'amenity_luxurys', 'Miscellaneous','PRICE','LONGITUDE', 'LATITUDE','PRICE_SQFT','RESALE','READY_TO_MOVE'
    ]
    df = (df.pipe(make_feature)
            .pipe(create_binary_columns)
            .pipe(categorize_property)
            .pipe(convert_to_cr)
            .pipe(convert_to_numeric)
            .pipe(categorize_age)
            .pipe(fill_missing_values)
            .pipe(FEATURES_LUXURY, 'AMENITIES', 'amenity_luxurys')
            .assign(AMENITY_LUXURY=lambda x: x['amenity_luxurys'].apply(classify_features))
            .pipe(FEATURES_LUXURY, 'FEATURES', 'FEATURES_LUXURY')
            .assign(FEATURES_LUXURY=lambda x: x['FEATURES_LUXURY'].apply(classify_features))
            .assign(
                    READY_TO_MOVE = df['SECONDARY_TAGS'].apply(lambda x: 1 if 'READY TO MOVE' in x else 0),
                    RESALE = df['SECONDARY_TAGS'].apply(lambda x: 1 if 'RESALE' in x else 0)
                )
                            .assign(
                FURNISH=df['FURNISH'].apply(categorize_furnish),
                FLOOR_NUM=df['FLOOR_NUM'].apply(categorize_floor),
                PRICE_SQFT=lambda x: pd.to_numeric(x['PRICE'], errors='coerce') / pd.to_numeric(x['AREA'], errors='coerce')
            )
            .assign(PREFERENCE=df['PREFERENCE'].map({'S': 'Sale', 'R': 'Rent', 'P': 'Pending'}))
            .drop(columns=['MAP_DETAILS', 'DESCRIPTION', 'FORMATTED_LANDMARK_DETAILS', 'PROPERTY_TYPE', 'SECONDARY_TAGS',
                           'location', 'PROP_ID', 'PRICE_PER_UNIT_AREA', 'FEATURES', 'PROP_NAME', 'BUILDING_NAME',
                           'AMENITIES', 'PROP_NAME', 'LOCALITY_NAME', 'ADDRESS', 'SOCIETY_NAME'])
            .loc[~df['PREFERENCE'].isin(['rent', 'pending'])]
            .loc[~df['PROP_NAME'].isin(['on request'])]
            .assign(CITY=df['CITY'].str.lower().replace({
                'mumbai': 'Mumbai', 'navi mumbai': 'Mumbai', 'central mumbai suburbs': 'Mumbai',
                'mumbai andheri-dahisar': 'Mumbai', 'south mumbai': 'Mumbai', 'mumbai beyond thane': 'Mumbai',
                'mira road and beyond': 'Mumbai', 'mumbai harbour': 'Mumbai', 'mumbai south west': 'Mumbai',
                'thane': 'Mumbai', 'kolkata': 'Kolkata', 'kolkata south': 'Kolkata', 'kolkata east': 'Kolkata',
                'kolkata north': 'Kolkata', 'kolkata west': 'Kolkata', 'kolkata central': 'Kolkata', 'secunderabad': 'hyderabad'
            }))

            .reindex(columns=columns_order, fill_value=None)
            .rename(columns=lambda x: x.upper())
            .drop(columns=['AMENITY_LUXURYS'])
            .assign(PREFERENCE=lambda x: x['PREFERENCE'].replace({'Sale':1})))


    return df

def split_data(data: pd.DataFrame, test_size: float, random_state: int):
    train_data, test_data = train_test_split(data, 
                                             test_size=test_size, 
                                             random_state=random_state)
    
    return train_data, test_data

    
   
    
if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent

    file_path = root_path / "data" / "raw" / "House_data.csv"
    df = load_data(file_path)
    df = preprocess_data(df)

    cleaned_data_filename = root_path / "data" / "cleaned" / "House_cleaned.csv"
    cleaned_data_filename.parent.mkdir(parents=True, exist_ok=True)  # ✅ create cleaned/
    save_data(df, cleaned_data_filename)

    train, test = split_data(df, 0.2, 42)
    train_filename = root_path / "data" / "interim" / "train.csv"
    test_filename = root_path / "data" / "interim" / "test.csv"
    
    train_filename.parent.mkdir(parents=True, exist_ok=True)  # ✅ create interim/

    save_data(train, train_filename)
    save_data(test, test_filename)
