import streamlit as st
from PIL import Image
import time

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="🏠 Real Estate Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Header with Branding -----------------
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/ios/250/real-estate.png", width=80)
with col2:
    st.title("Real Estate Intelligence Platform")
    st.markdown("#### Powered by AI for smarter property decisions 💡")

st.markdown("---")

# ----------------- Welcome Message -----------------
with st.container():
    st.subheader("👋 Welcome to your smart property assistant!")
    st.write(
        "Navigate through the pages using the sidebar on the left. "
        "Our platform helps you make data-driven decisions in real estate — whether you're buying, selling, or investing."
    )

# ----------------- Navigation Overview -----------------
st.markdown("### 🔍 Key Modules")
feature_cols = st.columns(3)

with feature_cols[0]:
    st.info("🔮 **Prediction**\n\nGet instant price estimates based on key property features.")

with feature_cols[1]:
    st.success("💡 **Recommendations**\n\nFind top listings tailored to your preferences and budget.")

with feature_cols[2]:
    st.warning("📊 **Analytics**\n\nExplore insightful charts and trends by city, price, or property type.")

# ----------------- Animated Metrics -----------------
with st.container():
    st.markdown("### 📈 Platform Snapshot")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Properties Analyzed", value="125,000+")
    with col2:
        st.metric(label="Cities Covered", value="85+")
    with col3:
        st.metric(label="Prediction Accuracy", value="92.5%")

# ----------------- Footer -----------------
st.markdown("---")
st.markdown(
    "<small>💼 Built with ❤️ by your data science team | 📬 Contact: support@realestate-ai.com</small>",
    unsafe_allow_html=True
)
