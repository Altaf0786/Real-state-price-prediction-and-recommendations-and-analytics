import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------
# Set Paths
# ----------------------------
root_path = Path(__file__).resolve().parent.parent.parent
data_path = root_path / 'data' / 'processed' / 'train_processed.csv'
similarity_path = root_path / 'models' / 'similarity_matrix.npy'

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Property Recommender", layout="wide")
st.title("🏘️ Property Recommender")
st.markdown("""
This module helps you explore similar properties based on selected features. 
Use the filters to tailor results by preference.
""")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(data_path)

@st.cache_resource
def load_similarity_matrix():
    return np.load(similarity_path)

df = load_data()
similarity_matrix = load_similarity_matrix()

# ----------------------------
# Recommendation Function
# ----------------------------
def recommend_properties(index, top_n=5, match_preference=False):
    if index < 0 or index >= len(df):
        st.error("Invalid property index.")
        return pd.DataFrame()

    similarity_scores = list(enumerate(similarity_matrix[index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:]

    if match_preference:
        user_pref = df.iloc[index]['PREFERENCE']
        sorted_scores = [x for x in sorted_scores if df.iloc[x[0]]['PREFERENCE'] == user_pref]

    top_indices = [i for i, _ in sorted_scores[:top_n]]
    return df.iloc[top_indices]

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("🔍 Recommendation Settings")
selected_index = st.sidebar.number_input("Select Property Index", min_value=0, max_value=len(df) - 1, value=0)
top_n = st.sidebar.slider("Top N Recommendations", min_value=1, max_value=10, value=5)
filter_by_pref = st.sidebar.checkbox("Match User Preference", value=True)

# ----------------------------
# Display Results
# ----------------------------
st.subheader("🏡 Selected Property")
st.dataframe(df.iloc[[selected_index]][['AREA', 'BEDROOM_NUM', 'PRICE_SQFT', 'FURNISH', 'PREFERENCE']], use_container_width=True)

st.subheader("🔁 Recommended Properties")
recommendations = recommend_properties(selected_index, top_n=top_n, match_preference=filter_by_pref)

if not recommendations.empty:
    st.dataframe(recommendations[['AREA', 'BEDROOM_NUM', 'PRICE_SQFT', 'FURNISH', 'PREFERENCE']], use_container_width=True)
else:
    st.warning("No matching properties found. Try disabling the preference filter or selecting another property.")
