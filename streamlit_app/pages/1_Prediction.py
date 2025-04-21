import streamlit as st
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import json
import logging
import sys
import dagshub

# ------------------- 🔧 LOGGER SETUP -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- 🌐 DAGSHUB & MLFLOW INIT -------------------
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

# ------------------- 🔁 LOAD MODEL FROM REGISTRY (PRODUCTION) -------------------
def load_model_information(file_path="/Users/mdaltafshekh/real-state-price-prediction-and-recommendations-and-analytics/run_information.json"):
    with open(file_path) as f:
        return json.load(f)

try:
    run_info = load_model_information()
    model_name = run_info['model_name']
    client = MlflowClient()
    model_uri = f"models:/{model_name}/Staging"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info(f"Model loaded from MLflow Model Registry: {model_uri}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    sys.exit(1)

# ------------------- 🎯 FEATURE NAMES -------------------
features = ['FACING', 'TRANSACT_TYPE', 'AREA', 'BEDROOM_NUM', 'BALCONY_NUM',
       'OWNTYPE', 'TOTAL_FLOOR', 'TOTAL_LANDMARK_COUNT', 'METROSTATION',
       'SHOPPING', 'CONNECTIVITY', 'EDUCATION', 'HOSPITAL', 'AIRPORT',
       'RAILWAYSTATION', 'OFFICECOMPLEX', 'HOTEL', 'AMUSEMENTPARK',
       'GOLFCOURSE', 'STADIUM', 'RELIGIOUSPLACE', 'ATM', 'PARKING', 'BUSDEPOT',
       'MISCELLANEOUS','LONGITUDE', 'LATITUDE', 'PRICE_SQFT',
       'RESALE', 'READY_TO_MOVE', 'Outlier_SVM', 'PROP_HEADING_1',
       'PROP_HEADING_2', 'PROP_HEADING_3', 'PROP_HEADING_4', 'PROP_HEADING_5',
       'CITY_1', 'CITY_2', 'CITY_3', 'CITY_4', 'AGE', 'FLOOR_NUM', 'FURNISH',
        'AMENITY_LUXURY', 'FEATURES_LUXURY']



row = {feature: 0 for feature in features}

# ------------------- 🔄 MAPPINGS -------------------
city_map = {'hyderabad': 'CITY_1', 'Kolkata': 'CITY_2', 'gurgaon': 'CITY_3', 'Mumbai': 'CITY_4'}
prop_heading_map = {
    'apartment': 'PROP_HEADING_1', 'plot': 'PROP_HEADING_2',
    'independent floor': 'PROP_HEADING_3', 'house': 'PROP_HEADING_4', 'other': 'PROP_HEADING_5'
}
age_mapping = {'relative new property': 0, 'moderate old property': 1, 'old property': 2}
floor_mapping = {
    'Unknown': 0, 'Basement': 1, 'Low-rise': 2, 'Ground Level': 3,
    'High-rise': 4, 'Mid-rise': 5, 'Very High-rise': 6
}
furnish_mapping = {'Unfurnished': 0, 'Semi-Furnished': 1, 'Fully Furnished': 2, 'Partially Furnished': 3}
amenity_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
feature_mapping = {'Low': 0, 'High': 1, 'Medium': 2}

# ------------------- 🏠 STREAMLIT UI -------------------
st.title("🏠 House Price per SqFt Prediction App")

# Property Info
st.subheader("🏢 Property Info")

city = st.selectbox("City", list(city_map.keys()))
row[city_map[city]] = 1

prop = st.selectbox("Property Type", list(prop_heading_map.keys()))
row[prop_heading_map[prop]] = 1

row['FACING'] = st.number_input("Facing (0-7)", min_value=0, max_value=7, value=0)

trans_type = st.selectbox("Transaction Type", ["New", "Resale"])
row['TRANSACT_TYPE'] = 0.0 if trans_type == "New" else 1.0

row['AREA'] = st.number_input("Area (SqFt)", min_value=100.0, max_value=20000.0, value=1000.0)
row['BEDROOM_NUM'] = st.slider("Number of Bedrooms", 0, 10, 2)
row['BALCONY_NUM'] = st.slider("Number of Balconies", 0, 5, 1)

own_type = st.selectbox("Ownership Type", ["Freehold", "Leasehold", "Co-operative", "Power of Attorney"])
row['OWNTYPE'] = ["Freehold", "Leasehold", "Co-operative", "Power of Attorney"].index(own_type)

row['TOTAL_FLOOR'] = st.slider("Total Floors", 1, 50, 10)
row['TOTAL_LANDMARK_COUNT'] = st.slider("Landmark Count", 0, 50, 10)

# Nearby Amenities
st.subheader("🏙️ Nearby Amenities (0 or 1)")
for field in [
    'METROSTATION', 'SHOPPING', 'CONNECTIVITY', 'EDUCATION', 'HOSPITAL', 'AIRPORT',
    'RAILWAYSTATION', 'OFFICECOMPLEX', 'HOTEL', 'RELIGIOUSPLACE', 'ATM',
    'MISCELLANEOUS', 'RESALE', 'READY_TO_MOVE', 'Outlier_SVM'
]:
    row[field] = st.selectbox(f"{field.replace('_', ' ').title()}", [0, 1], key=field)

for field in ['AMUSEMENTPARK', 'GOLFCOURSE', 'STADIUM', 'PARKING', 'BUSDEPOT']:
    row[field] = st.slider(f"{field.replace('_', ' ').title()} Distance (km)", 0.0, 50.0, 5.0)

# Location Info
st.subheader("📍 Location Info")
row['LONGITUDE'] = st.number_input("Longitude", value=77.5946)
row['LATITUDE'] = st.number_input("Latitude", value=12.9716)

# Property Characteristics
st.subheader("🏷️ Property Characteristics")
row['AGE'] = age_mapping[st.selectbox("Age of Property", list(age_mapping.keys()))]
row['FLOOR_NUM'] = floor_mapping[st.selectbox("Floor Type", list(floor_mapping.keys()))]
row['FURNISH'] = furnish_mapping[st.selectbox("Furnishing", list(furnish_mapping.keys()))]
row['AMENITY_LUXURY'] = amenity_mapping[st.selectbox("Amenity Luxury", list(amenity_mapping.keys()))]
row['FEATURES_LUXURY'] = feature_mapping[st.selectbox("Features Luxury", list(feature_mapping.keys()))]

# ------------------- 💰 PREDICT -------------------
if st.button("Predict 💰 Price in cr"):
    try:
        input_df = pd.DataFrame([row])  # Convert dict to DataFrame
        prediction = model.predict(input_df)[0]
        st.session_state.prediction_result = f"🏷️ Predicted Price per SqFt: ₹ {prediction:,.2f}"
    except Exception as e:
        st.session_state.prediction_result = f"Prediction failed: {e}"

# Show prediction only if it exists
if 'prediction_result' in st.session_state:
    st.success(st.session_state.prediction_result)
