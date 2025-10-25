import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('imports-from-african-countries.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['unit'] = df['unit'].fillna('Unknown')
    # Assume cleaned_df is df
    return df

cleaned_df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Time Series Forecast",
    "Country & Commodity Growth Trend",
    "Anomaly Detection",
    "Supply Chain Dependency & Risk",
    "Commodity Basket Forecasting",
    "Geographic Heatmap"
])

# Home Page
if page == "Home":
    st.title("Imports from African Countries Dashboard")
    st.write("Explore trade data, forecasts, and insights.")
    st.write("Dataset Overview:")
    st.dataframe(cleaned_df.head())
    st.write(f"Total Records: {len(cleaned_df)}")
    st.write(f"Date Range: {cleaned_df['date'].min()} to {cleaned_df['date'].max()}")

# Time Series Forecast Page
elif page == "Time Series Forecast":
    st.title("Time Series Forecast")
    st.write("Forecasting total imports for top 15 commodities using Prophet.")

    # Get top 15 commodities
    top_commodities = cleaned_df.groupby('commodity')['value_dl'].sum().nlargest(15).index.tolist()

    selected_commodity = st.selectbox("Select Commodity", top_commodities)

    # Aggregate monthly data
    monthly_data = cleaned_df[cleaned_df['commodity'] == selected_commodity].resample('M', on='date')['value_dl'].sum().reset_index()
    monthly_data.columns = ['ds', 'y']

    if not monthly_data.empty:
        # Fit Prophet model
        model = Prophet()
        model.fit(monthly_data)

        # Forecast next 12 months
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_data['ds'], y=monthly_data['y'], mode='lines', name='Historical'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None, mode='lines', line_color='lightblue', name='Lower Bound'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line_color='lightblue', name='Upper Bound'))
        st.plotly_chart(fig)
    else:
        st.warning("No data for selected commodity.")

# Country & Commodity Growth Trend Page
elif page == "Country & Commodity Growth Trend":
    st.title("Commodity Growth Trend by Country")

    all_countries = sorted(cleaned_df['country_name'].unique())
    all_commodities = sorted(cleaned_df['commodity'].unique())

    selected_country = st.selectbox("Select a Country", all_countries)
    selected_commodity = st.selectbox("Select a Commodity", all_commodities)

    filtered_data = cleaned_df[
        (cleaned_df['country_name'] == selected_country) &
        (cleaned_df['commodity'] == selected_commodity)
    ]

    if not filtered_data.empty:
        monthly_trend = filtered_data.resample('M', on='date')['value_dl'].sum().reset_index()
        monthly_trend.columns = ['Month', 'Total Import Value (USD)']

        st.subheader(f"Monthly Import Trend for {selected_commodity} from {selected_country}")
        st.line_chart(monthly_trend.set_index('Month'))

        st.write("Raw Data for Selection", filtered_data)
    else:
        st.warning("No data found for this specific country and commodity combination.")

# Anomaly Detection Page
elif page == "Anomaly Detection":
    st.title("Transaction Anomaly Detector")
    st.write("Detect suspicious transactions based on unit price anomalies.")

    # Calculate unit price
    cleaned_df['unit_price'] = cleaned_df['value_dl'] / cleaned_df['value_qt']
    cleaned_df['unit_price'] = cleaned_df['unit_price'].replace([np.inf, -np.inf], np.nan).fillna(0)

    selected_country = st.selectbox("Select Country for Anomaly Check", sorted(cleaned_df['country_name'].unique()))
    selected_commodity = st.selectbox("Select Commodity for Anomaly Check", sorted(cleaned_df['commodity'].unique()))

    group_data = cleaned_df[
        (cleaned_df['country_name'] == selected_country) &
        (cleaned_df['commodity'] == selected_commodity)
    ]

    if not group_data.empty:
        # Z-score for unit_price
        mean_price = group_data['unit_price'].mean()
        std_price = group_data['unit_price'].std()
        group_data['z_score'] = (group_data['unit_price'] - mean_price) / std_price
        anomalies = group_data[abs(group_data['z_score']) > 3]

        st.write(f"Anomalies (Z-score > 3 or < -3): {len(anomalies)}")
        st.dataframe(anomalies[['date', 'value_dl', 'value_qt', 'unit_price', 'z_score']])

        # Plot unit prices with anomalies
        fig = px.scatter(group_data, x='date', y='unit_price', color=abs(group_data['z_score']) > 3,
                         color_discrete_map={True: 'red', False: 'blue'},
                         title="Unit Prices with Anomalies Highlighted")
        st.plotly_chart(fig)
    else:
        st.warning("No data for selection.")

# Supply Chain Dependency & Risk Page
elif page == "Supply Chain Dependency & Risk":
    st.title("Supply Chain Dependency & Risk Dashboard")

    selected_commodity = st.selectbox("Select Commodity for Risk Analysis", sorted(cleaned_df['commodity'].unique()))

    commodity_data = cleaned_df[cleaned_df['commodity'] == selected_commodity]

    if not commodity_data.empty:
        # Concentration Score: Max % from one country
        country_totals = commodity_data.groupby('country_name')['value_dl'].sum()
        total_value = country_totals.sum()
        concentration = (country_totals.max() / total_value) * 100

        # Volatility Score: Std dev of YoY changes
        monthly_totals = commodity_data.resample('M', on='date')['value_dl'].sum()
        yoy_changes = monthly_totals.pct_change(12).dropna()
        volatility = yoy_changes.std() * 100  # As percentage

        st.write(f"Concentration Score: {concentration:.2f}% (from top country)")
        st.write(f"Volatility Score: {volatility:.2f}%")

        # Risk Matrix (simple scatter)
        fig = px.scatter(x=[concentration], y=[volatility], text=[selected_commodity])
        fig.update_layout(xaxis_title="Dependency (%)", yaxis_title="Volatility (%)",
                          title="Risk Matrix")
        fig.add_hline(y=50, line_dash="dash", annotation_text="High Volatility")
        fig.add_vline(x=50, line_dash="dash", annotation_text="High Dependency")
        st.plotly_chart(fig)

        # Top countries
        st.subheader("Top Countries by Value")
        st.bar_chart(country_totals.nlargest(10))
    else:
        st.warning("No data for selected commodity.")

# Commodity Basket Forecasting Page
elif page == "Commodity Basket Forecasting":
    st.title("Commodity Basket Forecasting")

    # Define baskets (hardcoded for simplicity)
    baskets = {
        "Scrap Metals": ['Aluminium Scrap', 'Brass Scrap', 'Copper Scrap', 'Iron And Steel Scrap'],
        "Oils": ['Petroleum Oils', 'Palm Oil'],
        "Gems": ['Gold', 'Diamonds']
    }

    selected_basket = st.selectbox("Select Basket", list(baskets.keys()))

    basket_commodities = baskets[selected_basket]
    basket_data = cleaned_df[cleaned_df['commodity'].isin(basket_commodities)]

    if not basket_data.empty:
        # Aggregate monthly
        monthly_basket = basket_data.resample('M', on='date')['value_dl'].sum().reset_index()
        monthly_basket.columns = ['ds', 'y']

        # Forecast
        model = Prophet()
        model.fit(monthly_basket)
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_basket['ds'], y=monthly_basket['y'], mode='lines', name='Historical'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        st.plotly_chart(fig)
    else:
        st.warning("No data for basket.")

# Geographic Heatmap Page
elif page == "Geographic Heatmap":
    st.title("Geographic Heatmap of Imports")

    # Aggregate by country
    country_totals = cleaned_df.groupby('country_name')['value_dl'].sum().reset_index()

    # Simple bar chart as proxy (since no map library)
    fig = px.bar(country_totals, x='country_name', y='value_dl', title="Import Values by Country")
    st.plotly_chart(fig)

    st.write("Note: For full map, integrate plotly choropleth with country codes.")
