import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from pathlib import Path
from folium.plugins import MarkerCluster

# Set the root path to load the dataset
root_path = Path(__file__).resolve().parent.parent.parent
data_path = root_path / 'data' / 'processed' / 'train_processed.csv'

# Load the dataset
def load_data():
    df = pd.read_csv(data_path)
    return df

# General statistics function
def general_stats(df):
    st.subheader("General Statistics")
    st.write(df.describe())

# Price distribution function
def price_distribution(df):
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['PRICE'], bins=30, kde=True, ax=ax, color='purple')
    ax.set_title("Price Distribution of Houses")
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# Price vs Area function
def price_by_area(df):
    st.subheader("Price by Area")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='AREA', y='PRICE', ax=ax, color='blue', alpha=0.6)
    ax.set_title("Price vs Area")
    ax.set_xlabel('Area (in sqft)')
    ax.set_ylabel('Price')
    st.pyplot(fig)

# Price by number of bedrooms function
def price_by_bedroom(df):
    st.subheader("Price by Number of Bedrooms")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='BEDROOM_NUM', y='PRICE', ax=ax, palette="Set2")
    ax.set_title("Price Distribution by Bedroom Count")
    ax.set_xlabel('Number of Bedrooms')
    ax.set_ylabel('Price')
    st.pyplot(fig)

# Price by amenities function
def price_by_amenities(df):
    st.subheader("Price by Amenities")
    amenities = ['PREFERENCE', 'FURNISH', 'AMENITY_LUXURY', 'FEATURES_LUXURY']
    
    for amenity in amenities:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x=amenity, y='PRICE', ax=ax, palette="Set3")
        ax.set_title(f"Price by {amenity}")
        ax.set_xlabel(amenity)
        ax.set_ylabel('Price')
        st.pyplot(fig)

# Correlation matrix function
def correlation_matrix(df):
    st.subheader("Correlation Matrix")
    
    # Select only numeric columns for correlation calculation
    df_numeric = df.select_dtypes(include=['number'])
    
    # Calculate correlation matrix
    corr = df_numeric.corr()
    
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix")
    
    # Display the plot
    st.pyplot(fig)


# Feature analysis for luxury amenities
def feature_analysis(df):
    st.subheader("Feature Analysis")
    
    # Analyzing luxury features like AMENITY_LUXURY, FEATURES_LUXURY
    amenities = ['AMENITY_LUXURY', 'FEATURES_LUXURY']
    
    for feature in amenities:
        st.write(f"### Distribution of {feature}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x=feature, ax=ax, palette='Blues')
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        st.pyplot(fig)
        
        st.write(f"### {feature} vs Price")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x=feature, y='PRICE', ax=ax, palette="Set2")
        ax.set_title(f"Price by {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel('Price')
        st.pyplot(fig)

# Top 10 most expensive houses function
def top_10_houses(df):
    st.subheader("Top 10 Most Expensive Houses")
    
    # Sort the DataFrame by 'PRICE' in descending order to get the top 10
    top_10 = df[['AREA', 'BEDROOM_NUM', 'PRICE', 'PREFERENCE', 'FURNISH']].sort_values(by='PRICE', ascending=False).head(10)
    
    st.write(top_10)
    
    # Optional: Visualize the top 10 prices
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='PRICE', y=top_10.index.astype(str), data=top_10, ax=ax, palette="viridis")
    ax.set_title("Top 10 Most Expensive Houses")
    ax.set_xlabel('Price')
    ax.set_ylabel('Index of House')
    st.pyplot(fig)

# Map visualization for the properties
import streamlit.components.v1 as components

# Map visualization for the properties
def plot_map(df):
    st.subheader("Map of Property Locations")
    
    # Create a base map centered around the average latitude and longitude
    map_center = [df['LATITUDE'].mean(), df['LONGITUDE'].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=12)
    
    # Add a marker cluster
    marker_cluster = MarkerCluster().add_to(folium_map)
    
    # Add markers for each property
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            popup=f"Price: {row['PRICE']}<br>Area: {row['AREA']} sqft<br>Bedrooms: {row['BEDROOM_NUM']}",
        ).add_to(marker_cluster)
    
    # Use Streamlit components to display the Folium map
    components.html(folium_map._repr_html_(), height=500)


# User filters function
def user_filters(df):
    st.sidebar.title("Filter Options")
    
    # Price range filter
    min_price, max_price = st.sidebar.slider(
        'Price Range',
        min_value=int(df['PRICE'].min()),
        max_value=int(df['PRICE'].max()),
        value=(int(df['PRICE'].min()), int(df['PRICE'].max()))
    )
    df = df[(df['PRICE'] >= min_price) & (df['PRICE'] <= max_price)]
    
    # Area filter
    min_area, max_area = st.sidebar.slider(
        'Area Range (sqft)',
        min_value=int(df['AREA'].min()),
        max_value=int(df['AREA'].max()),
        value=(int(df['AREA'].min()), int(df['AREA'].max()))
    )
    df = df[(df['AREA'] >= min_area) & (df['AREA'] <= max_area)]
    
    # Number of bedrooms filter
    bedroom_options = df['BEDROOM_NUM'].unique()
    selected_bedrooms = st.sidebar.multiselect(
        'Select Number of Bedrooms',
        options=bedroom_options,
        default=bedroom_options
    )
    df = df[df['BEDROOM_NUM'].isin(selected_bedrooms)]
    
    return df

# Main function to load data and display insights
def main():
   
    df = load_data()
    
    # Apply user filters
    filtered_df = user_filters(df)
    
    # Display the general statistics
    general_stats(filtered_df)
    
    # Display price distribution
    price_distribution(filtered_df)
    
    # Display price vs area plot
    price_by_area(filtered_df)
    
    # Display price by bedroom count
    price_by_bedroom(filtered_df)
    
    # Display price by amenities
    price_by_amenities(filtered_df)
    
    # Display correlation matrix
    correlation_matrix(filtered_df)
    
    # Display feature analysis for luxury amenities
    feature_analysis(filtered_df)
    
    # Display top 10 most expensive houses
    top_10_houses(filtered_df)
    
    # Display map of property locations
    plot_map(filtered_df)

# Run the Streamlit app
if __name__ == "__main__":
    main()

