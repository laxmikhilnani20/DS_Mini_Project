import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
import plotly.express as px
import plotly.graph_objects as go

# Load and clean the data
@st.cache_data
def load_data():
    df = pd.read_csv('imports-from-african-countries.csv')
    # Fill missing units
    df['unit'] = df['unit'].fillna('Unknown')
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    return df

cleaned_df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Time Series Forecast", "Country & Commodity Growth Trend", "Transaction Anomaly Detector", "Supply Chain Risk Dashboard", "Sector Forecast"])

if page == "Time Series Forecast":
    st.title("Time Series Forecast")
    # Your existing forecast code here
    st.write("Existing forecast functionality")

elif page == "Country & Commodity Growth Trend":
    st.title("Commodity Growth Trend by Country")

    # Get unique sorted lists for selectors
    all_countries = sorted(cleaned_df['country_name'].unique())
    all_commodities = sorted(cleaned_df['commodity'].unique())

    # Create the selectors
    selected_country = st.selectbox("Select a Country", all_countries)
    selected_commodity = st.selectbox("Select a Commodity", all_commodities)

    # Filter the DataFrame based on the user's selection
    filtered_data = cleaned_df[
        (cleaned_df['country_name'] == selected_country) &
        (cleaned_df['commodity'] == selected_commodity)
    ]

    if not filtered_data.empty:
        # Resample by month to get a smooth trend
        monthly_trend = filtered_data.resample('M', on='date')['value_dl'].sum().reset_index()
        
        # Rename for plotting
        monthly_trend.columns = ['Month', 'Total Import Value (USD)']
        
        st.subheader(f"Monthly Import Trend for {selected_commodity} from {selected_country}")
        
        # Display the line chart
        st.line_chart(monthly_trend.set_index('Month'))
        
        # Optional: Show the raw data
        st.write("Raw Data for Selection", filtered_data)
    else:
        st.warning("No data found for this specific country and commodity combination.")

elif page == "Transaction Anomaly Detector":
    st.title("Transaction Anomaly Detector")
    # Implement anomaly detection
    st.write("Anomaly detection feature coming soon")

elif page == "Supply Chain Risk Dashboard":
    st.title("Supply Chain Risk Dashboard")
    # Implement risk dashboard
    st.write("Risk dashboard feature coming soon")

elif page == "Sector Forecast":
    st.title("Sector Forecast")
    # Implement basket forecasting
    st.write("Sector forecast feature coming soon")