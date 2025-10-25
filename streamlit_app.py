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
    "Geographic Heatmap",
    "Seasonal Patterns",
    "Multi-Country Comparison",
    "Volatility Risk",
    "Price Sensitivity",
    "Top-N Rankings",
    "Forecast Uncertainty",
    "Correlation Explorer",
    "Custom Query",
    "Sustainability ESG"
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

    if not monthly_data.empty and len(monthly_data) > 1:
        try:
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
        except Exception as e:
            st.error(f"Forecasting failed: {str(e)}")
    else:
        st.warning("Insufficient data for forecasting (need at least 2 data points).")

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

        if len(monthly_basket) > 1:
            try:
                # Forecast
                model = Prophet()
                model.fit(monthly_basket)
                future = model.make_future_dataframe(periods=12, freq='M')
                forecast = model.predict(future)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=monthly_basket['ds'], y=monthly_basket['y'], mode='lines', name='Historical'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Basket forecasting failed: {str(e)}")
        else:
            st.warning("Insufficient data for basket forecasting.")
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

# Seasonal Patterns Page
elif page == "Seasonal Patterns":
    st.title("Seasonal Patterns in Imports")

    selected_commodity = st.selectbox("Select Commodity for Seasonal Analysis", sorted(cleaned_df['commodity'].unique()))

    commodity_data = cleaned_df[cleaned_df['commodity'] == selected_commodity]

    if not commodity_data.empty:
        # Monthly aggregation
        monthly = commodity_data.groupby(commodity_data['date'].dt.month)['value_dl'].sum().reset_index()
        monthly.columns = ['Month', 'Total Value']

        fig = px.bar(monthly, x='Month', y='Total Value', title=f"Monthly Imports for {selected_commodity}")
        st.plotly_chart(fig)

        st.write("Seasonal Insights: Higher values in certain months indicate peak seasons.")
    else:
        st.warning("No data for selected commodity.")

# Multi-Country Comparison Page
elif page == "Multi-Country Comparison":
    st.title("Multi-Country Comparison")

    selected_commodity = st.selectbox("Select Commodity", sorted(cleaned_df['commodity'].unique()))
    selected_countries = st.multiselect("Select Countries", sorted(cleaned_df['country_name'].unique()), default=sorted(cleaned_df['country_name'].unique())[:3])

    if selected_countries:
        filtered = cleaned_df[(cleaned_df['commodity'] == selected_commodity) & (cleaned_df['country_name'].isin(selected_countries))]

        if not filtered.empty:
            yearly = filtered.groupby(['country_name', filtered['date'].dt.year])['value_dl'].sum().reset_index()
            yearly.columns = ['Country', 'Year', 'Value']

            fig = px.line(yearly, x='Year', y='Value', color='Country', title=f"Yearly Imports of {selected_commodity}")
            st.plotly_chart(fig)
        else:
            st.warning("No data for selection.")
    else:
        st.warning("Select at least one country.")

# Volatility Risk Page
elif page == "Volatility Risk":
    st.title("Volatility Risk Analysis")

    selected_commodity = st.selectbox("Select Commodity for Volatility", sorted(cleaned_df['commodity'].unique()))

    commodity_data = cleaned_df[cleaned_df['commodity'] == selected_commodity]

    if not commodity_data.empty:
        monthly = commodity_data.resample('M', on='date')['value_dl'].sum()
        volatility = monthly.pct_change().std() * 100

        st.write(f"Volatility (Std Dev of Monthly % Changes): {volatility:.2f}%")

        fig = px.line(monthly, title=f"Monthly Imports for {selected_commodity}")
        st.plotly_chart(fig)
    else:
        st.warning("No data for selected commodity.")

# Price Sensitivity Page
elif page == "Price Sensitivity":
    st.title("Price Sensitivity Analysis")

    # Simple correlation between quantity and value
    st.write("Correlation between Quantity and Value (Price Sensitivity)")

    corr = cleaned_df[['value_qt', 'value_dl']].corr().iloc[0,1]
    st.write(f"Correlation Coefficient: {corr:.2f}")

    fig = px.scatter(cleaned_df, x='value_qt', y='value_dl', title="Quantity vs Value Scatter")
    st.plotly_chart(fig)

# Top-N Rankings Page
elif page == "Top-N Rankings":
    st.title("Top-N Rankings")

    n = st.slider("Select Top N", 5, 20, 10)

    top_countries = cleaned_df.groupby('country_name')['value_dl'].sum().nlargest(n)
    top_commodities = cleaned_df.groupby('commodity')['value_dl'].sum().nlargest(n)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Countries by Import Value")
        st.bar_chart(top_countries)

    with col2:
        st.subheader("Top Commodities by Import Value")
        st.bar_chart(top_commodities)

# Forecast Uncertainty Page
elif page == "Forecast Uncertainty":
    st.title("Forecast Uncertainty")

    selected_commodity = st.selectbox("Select Commodity for Uncertainty", sorted(cleaned_df['commodity'].unique()))

    monthly_data = cleaned_df[cleaned_df['commodity'] == selected_commodity].resample('M', on='date')['value_dl'].sum().reset_index()
    monthly_data.columns = ['ds', 'y']

    if len(monthly_data) > 1:
        try:
            model = Prophet()
            model.fit(monthly_data)
            future = model.make_future_dataframe(periods=12, freq='M')
            forecast = model.predict(future)

            uncertainty = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=uncertainty['ds'], y=uncertainty['yhat'], mode='lines', name='Forecast'))
            fig.add_trace(go.Scatter(x=uncertainty['ds'], y=uncertainty['yhat_lower'], fill=None, mode='lines', line_color='lightblue', name='Lower'))
            fig.add_trace(go.Scatter(x=uncertainty['ds'], y=uncertainty['yhat_upper'], fill='tonexty', mode='lines', line_color='lightblue', name='Upper'))
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Uncertainty analysis failed: {str(e)}")
    else:
        st.warning("Insufficient data.")

# Correlation Explorer Page
elif page == "Correlation Explorer":
    st.title("Correlation Explorer")

    # Correlation between commodities
    pivot = cleaned_df.pivot_table(values='value_dl', index='date', columns='commodity', aggfunc='sum').fillna(0)
    corr_matrix = pivot.corr()

    fig = px.imshow(corr_matrix, title="Commodity Correlation Matrix")
    st.plotly_chart(fig)

# Custom Query Page
elif page == "Custom Query":
    st.title("Custom Query")

    st.write("Filter data by country, commodity, and date range.")

    country = st.selectbox("Country", ['All'] + sorted(cleaned_df['country_name'].unique()))
    commodity = st.selectbox("Commodity", ['All'] + sorted(cleaned_df['commodity'].unique()))
    start_date = st.date_input("Start Date", cleaned_df['date'].min())
    end_date = st.date_input("End Date", cleaned_df['date'].max())

    filtered = cleaned_df[
        (cleaned_df['date'] >= pd.to_datetime(start_date)) &
        (cleaned_df['date'] <= pd.to_datetime(end_date))
    ]

    if country != 'All':
        filtered = filtered[filtered['country_name'] == country]
    if commodity != 'All':
        filtered = filtered[filtered['commodity'] == commodity]

    st.dataframe(filtered.head(100))
    st.write(f"Total Records: {len(filtered)}")

# Sustainability ESG Page
elif page == "Sustainability ESG":
    st.title("Sustainability & ESG Insights")

    st.write("Basic ESG metrics: Assuming some commodities are sustainable.")

    sustainable_commodities = ['Gold', 'Diamonds', 'Coffee']  # Example

    esg_data = cleaned_df[cleaned_df['commodity'].isin(sustainable_commodities)].groupby('commodity')['value_dl'].sum()

    fig = px.pie(values=esg_data.values, names=esg_data.index, title="Sustainable Commodities Share")
    st.plotly_chart(fig)
