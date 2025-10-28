"""
African Imports Analytics Platform
Main application router - imports and routes to modular dashboards
"""

import streamlit as st
import pandas as pd
import warnings

# Import page modules
from pages import data_overview, eda_explorer, ml_models, anomaly_detection, risk_optimization

warnings.filterwarnings('ignore')

# ======================================================================
# PAGE CONFIGURATION
# ======================================================================
st.set_page_config(
    page_title="African Imports Analytics Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================
# CUSTOM CSS - DARK MODE COMPATIBLE
# ======================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: var(--text-color);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: rgba(30, 136, 229, 0.1);
        padding: 1rem;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .insight-box h4 {
        color: #1E88E5;
        margin-top: 0;
    }
    .insight-box p, .insight-box ul, .insight-box li {
        color: var(--text-color);
    }
    
    /* Ensure text is readable in both modes */
    [data-testid="stMarkdownContainer"] {
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# ======================================================================
# DATA LOADING
# ======================================================================
DATA_FILE = 'imports-from-african-countries.csv'

@st.cache_data
def load_data():
    """Load and prepare the dataset."""
    try:
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        df['unit'] = df['unit'].fillna('Unknown')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.month_name()
        df['quarter'] = df['date'].dt.quarter
        return df
    except FileNotFoundError:
        st.error(f"❌ Data file '{DATA_FILE}' not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

# Load data once
df = load_data()

# ======================================================================
# SIDEBAR NAVIGATION
# ======================================================================
st.sidebar.markdown("# 🌍 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Dashboard:",
    [
        "📊 Data Overview",
        "🔍 EDA Explorer", 
        "🎯 Interactive Models",
        "🔍 Anomaly Detection",
        "⚠️ Risk & Optimization"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**African Imports Analytics Platform**

Comprehensive analysis of imports from African countries covering:
- 139K+ transactions
- 50+ countries
- 4,000+ commodities
- 2015-2025 timeline
""")

# ======================================================================
# ROUTE TO SELECTED PAGE
# ======================================================================
if page == "📊 Data Overview":
    data_overview.render(df)

elif page == "🔍 EDA Explorer":
    eda_explorer.render(df)

elif page == "🎯 Interactive Models":
    ml_models.render(df)

elif page == "🔍 Anomaly Detection":
    anomaly_detection.render(df)

elif page == "⚠️ Risk & Optimization":
    risk_optimization.render(df)

# ======================================================================
# FOOTER
# ======================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>African Imports Analytics Platform</strong></p>
    <p>Comprehensive analysis of import data from African countries (2015-2025)</p>
    <p style="font-size: 0.9rem;">Data-driven insights for trade policy and procurement strategy</p>
</div>
""", unsafe_allow_html=True)
