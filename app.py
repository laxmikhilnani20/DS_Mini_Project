import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from scipy.stats import zscore
import os

# ----------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------
# Set the layout to wide mode for better dashboarding
st.set_page_config(
    page_title="African Imports Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ----------------------------------------------------------------------
# Caching - Load Data and Models
# ----------------------------------------------------------------------

# Define the file paths (adjust if you saved them elsewhere)
DATA_FILE = 'cleaned_imports_from_african-countries.csv'
RIDGE_MODEL_FILE = 'ridge_model_pipeline.pkl'
# --- NOTE: Your notebook trained an XGBoost model but didn't save it.
# --- I am assuming you will save it as 'xgboost_model_pipeline.pkl'
# --- If not, the app will gracefully handle its absence.
XGB_MODEL_FILE = 'xgboost_model_pipeline.pkl'

# List of the top 15 commodities your notebook trained models for
TOP_15_COMMODITIES = [
    'PETROLEUM OILS, ETC,(NOT CRUDE)', 'OTHER NON-MONETARY UNWROUGHT FORMS OF GOLD',
    'STEAM COAL', 'PHOSPHORIC ACID', 'CRUDE PETROLEUM',
    'LIQUEFIED NATURAL GAS', 'BITUMINOUS COAL', 'GOLD DOR',
    'DI AMMONIUM PHOSPHATE', 'MANGANESE ORE(46% OR MORE BUT BLW.48%)',
    'LUMP CHROME ORE', 'MANGANESE ORE(35% OR MORE BUT BLW.44%)',
    'MANGANESE ORE(30% OR MORE BUT BLW.35%)', 'HARD CHROME ORE',
    'MANGANESE ORE(44% OR MORE BUT BLW.46%)'
]

@st.cache_data
def load_data():
    """Loads and preprocesses the cleaned data from CSV."""
    try:
        df = pd.read_csv(DATA_FILE)
        # CRITICAL: Convert date column to datetime objects for time series
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file '{DATA_FILE}' not found.")
        st.stop() # Stop the app if data can't be loaded
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

@st.cache_resource
def load_models():
    """Loads all pickled models from disk."""
    prophet_models = {}
    
    # 1. Load all Prophet models
    for commodity in TOP_15_COMMODITIES:
        model_file = f'prophet_model_{commodity}.pkl'
        try:
            with open(model_file, 'rb') as f:
                prophet_models[commodity] = pickle.load(f)
        except FileNotFoundError:
            st.warning(f"Could not find model file: {model_file}")
        except Exception as e:
            st.error(f"Error loading {model_file}: {e}")

    # 2. Load Ridge model
    try:
        with open(RIDGE_MODEL_FILE, 'rb') as f:
            ridge_pipeline = pickle.load(f)
    except FileNotFoundError:
        st.error(f"CRITICAL: Ridge model '{RIDGE_MODEL_FILE}' not found. Prediction page will fail.")
        ridge_pipeline = None
    except Exception as e:
        st.error(f"Error loading {RIDGE_MODEL_FILE}: {e}")
        ridge_pipeline = None


    # 3. Load XGBoost model (with check)
    xgb_pipeline = None
    if os.path.exists(XGB_MODEL_FILE):
        try:
            with open(XGB_MODEL_FILE, 'rb') as f:
                xgb_pipeline = pickle.load(f)
        except Exception as e:
            st.warning(f"Error loading {XGB_MODEL_FILE}: {e}. XGBoost predictions will be unavailable.")
    else:
        st.info(f"Note: '{XGB_MODEL_FILE}' not found. Only Ridge predictions will be shown.")

    return prophet_models, ridge_pipeline, xgb_pipeline

# ----------------------------------------------------------------------
# Main App
# ----------------------------------------------------------------------

# Load all data and models once
df = load_data()
prophet_models, ridge_pipeline, xgb_pipeline = load_models()

# --- Sidebar Navigation ---
st.sidebar.title("Imports Dashboard")
page = st.sidebar.radio(
    "Select a Page",
    [
        "Home / EDA",
        "Country-Commodity Growth Trend", # User's requested feature
        "Time Series Forecast (Top 15)",
        "Import Value Prediction",
        "Transaction Anomaly Detector" # Novel approach
    ]
)

# ----------------------------------------------------------------------
# Page 1: Home / EDA
# ----------------------------------------------------------------------
if page == "Home / EDA":
    st.title("🌍 African Imports General Dashboard")
    st.write("A high-level overview of the import data from African countries.")

    # Top-level metrics
    total_value_dl = df['value_dl'].sum()
    total_transactions = len(df)
    num_countries = df['country_name'].nunique()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Import Value (USD)", f"${total_value_dl:,.0f}")
    col2.metric("Total Transactions", f"{total_transactions:,}")
    col3.metric("Number of Countries", num_countries)

    st.markdown("---")

    # EDA Plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Countries by Import Value (USD)")
        top_countries = df.groupby('country_name')['value_dl'].sum().nlargest(10).reset_index()
        fig_countries = px.bar(top_countries, x='country_name', y='value_dl', 
                               title="Top 10 Countries", labels={'country_name': 'Country', 'value_dl': 'Total Value (USD)'})
        st.plotly_chart(fig_countries, use_container_width=True)

    with col2:
        st.subheader("Top 10 Commodities by Import Value (USD)")
        top_commodities = df.groupby('commodity')['value_dl'].sum().nlargest(10).reset_index()
        fig_commodities = px.bar(top_commodities, x='commodity', y='value_dl', 
                                 title="Top 10 Commodities", labels={'commodity': 'Commodity', 'value_dl': 'Total Value (USD)'})
        st.plotly_chart(fig_commodities, use_container_width=True)

    st.subheader("Total Import Value Over Time")
    monthly_total = df.resample('M', on='date')['value_dl'].sum().reset_index()
    fig_time = px.line(monthly_total, x='date', y='value_dl', 
                       title="Total Import Value (USD) per Month", labels={'date': 'Date', 'value_dl': 'Total Value (USD)'})
    st.plotly_chart(fig_time, use_container_width=True)

# ----------------------------------------------------------------------
# Page 2: Country-Commodity Growth Trend (USER'S KEY REQUEST)
# ----------------------------------------------------------------------
elif page == "Country-Commodity Growth Trend":
    st.title("📈 Country-Commodity Growth Trend")
    st.write("Select a country and a commodity to see its specific historical import trend.")

    # Get sorted lists for selectors
    all_countries = sorted(df['country_name'].unique())
    all_commodities = sorted(df['commodity'].unique())

    # Create two columns for selectors
    col1, col2 = st.columns(2)
    with col1:
        selected_country = st.selectbox("Select a Country", all_countries, index=all_countries.index("South Africa"))
    with col2:
        selected_commodity = st.selectbox("Select a Commodity", all_commodities, index=all_commodities.index("STEAM COAL"))

    # Filter data based on selection
    filtered_data = df[
        (df['country_name'] == selected_country) &
        (df['commodity'] == selected_commodity)
    ]

    if not filtered_data.empty:
        # Resample by month to get a smooth trend
        monthly_trend = filtered_data.resample('M', on='date')['value_dl'].sum().reset_index()
        monthly_trend.columns = ['Month', 'Total Import Value (USD)']
        
        st.subheader(f"Monthly Import Trend for {selected_commodity} from {selected_country}")
        
        # Display the line chart
        fig_trend = px.line(monthly_trend, x='Month', y='Total Import Value (USD)',
                            title=f"{selected_commodity} from {selected_country}")
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Optional: Show the raw data
        with st.expander("Show Raw Data for Selection"):
            st.dataframe(filtered_data.sort_values('date', ascending=False))
    else:
        st.warning("No data found for this specific country and commodity combination.")

# ----------------------------------------------------------------------
# Page 3: Time Series Forecast (Top 15)
# ----------------------------------------------------------------------
elif page == "Time Series Forecast (Top 15)":
    st.title("⏳ Time Series Forecast (Top 15 Commodities)")
    st.write("Select one of the top 15 commodities to see its 1-year (365 day) forecast.")

    if not prophet_models:
        st.error("No Prophet models were loaded. Please check model files.")
    else:
        selected_commodity = st.selectbox("Select Commodity to Forecast", list(prophet_models.keys()))
        
        if selected_commodity in prophet_models:
            model = prophet_models[selected_commodity]
            
            # Generate future dates
            future = model.make_future_dataframe(periods=365)
            # Predict
            forecast = model.predict(future)
            
            st.subheader(f"Forecast for {selected_commodity}")
            # Use model's built-in plotting
            from prophet.plot import plot
            fig1 = model.plot(forecast)
            st.pyplot(fig1)
            
            st.subheader(f"Forecast Components for {selected_commodity}")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
            
            with st.expander("Show Forecast Data"):
                st.dataframe(forecast.tail(365))
        else:
            st.error("Selected model not found.")

# ----------------------------------------------------------------------
# Page 4: Import Value Prediction
# ----------------------------------------------------------------------
elif page == "Import Value Prediction":
    st.title("💸 Predict Import Value (USD)")
    st.write("Fill in the details of an import to get a predicted value (USD) from our models.")

    if not ridge_pipeline:
        st.error("Regression models not loaded. This page cannot function.")
    else:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                country_name = st.selectbox("Country", sorted(df['country_name'].unique()))
                commodity = st.selectbox("Commodity", sorted(df['commodity'].unique()))
            with col2:
                unit = st.selectbox("Unit", sorted(df['unit'].unique()))
                value_qt = st.number_input("Import Quantity (e.g., in Kgs)", min_value=0.0, value=1000.0, step=100.0)

            # Date components
            import_date = st.date_input("Import Date", pd.to_datetime("today"))
            month = import_date.month
            year = import_date.year

            submit_button = st.form_submit_button("Predict Import Value")

        if submit_button:
            # Create the input DataFrame exactly as the model pipeline expects
            input_data = pd.DataFrame({
                'value_qt': [value_qt],
                'country_name': [country_name],
                'commodity': [commodity],
                'unit': [unit],
                'month': [month],
                'year': [year]
            })

            st.subheader("Model Predictions")
            
            # Ridge Prediction
            try:
                ridge_pred = ridge_pipeline.predict(input_data)[0]
                st.metric("Ridge Model Prediction", f"${ridge_pred:,.2f}")
            except Exception as e:
                st.error(f"Error with Ridge prediction: {e}")

            # XGBoost Prediction (if loaded)
            if xgb_pipeline:
                try:
                    xgb_pred = xgb_pipeline.predict(input_data)[0]
                    st.metric("XGBoost Model Prediction", f"${xgb_pred:,.2f}")
                except Exception as e:
                    st.error(f"Error with XGBoost prediction: {e}")
            else:
                st.info("XGBoost model was not found or failed to load. Only showing Ridge prediction.")

# ----------------------------------------------------------------------
# Page 5: Transaction Anomaly Detector
# ----------------------------------------------------------------------
elif page == "Transaction Anomaly Detector":
    st.title("🔍 Transaction Anomaly Detector")
    st.write("Find transactions with unusual unit prices (Value in USD / Quantity).")
    st.write("This can help identify data entry errors or potential trade irregularities.")

    # Calculate Unit Price
    df_anom = df.copy()
    # Avoid division by zero, replace inf with nan, then drop nans
    df_anom['unit_price'] = df_anom['value_dl'] / df_anom['value_qt']
    df_anom = df_anom.replace([np.inf, -np.inf], np.nan).dropna(subset=['unit_price'])
    
    # Calculate Z-score grouped by commodity
    # This standardizes the unit price within each commodity group
    df_anom['price_zscore'] = df_anom.groupby('commodity')['unit_price'].transform(
        lambda x: zscore(x, nan_policy='omit')
    )
    
    # User-adjustable threshold
    threshold = st.slider("Select Z-Score Threshold (Higher = Stricter)", 
                          min_value=1.0, max_value=10.0, value=3.0, step=0.5)

    # Filter for anomalies
    anomalies = df_anom[
        (df_anom['price_zscore'].abs() > threshold) &
        (df_anom['unit_price'] > 0) # Exclude zero prices
    ].sort_values('price_zscore', ascending=False)
    
    st.subheader(f"Found {len(anomalies)} transactions with Z-Score > {threshold}")
    
    # Display results
    st.dataframe(anomalies[[
        'date', 'country_name', 'commodity', 'value_qt', 'value_dl', 'unit_PRICE', 'price_zscore'
    ]])
