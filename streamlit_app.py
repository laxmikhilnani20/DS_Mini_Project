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
    "Data Dashboard",
    "EDA Dashboard",
    "Models & Novelty Dashboard"
])

# Data Dashboard
if page == "Data Dashboard":
    st.title("📊 Data Dashboard")
    st.write("Overview of Imports from African Countries Dataset")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(cleaned_df))
    with col2:
        st.metric("Unique Countries", cleaned_df['country_name'].nunique())
    with col3:
        st.metric("Unique Commodities", cleaned_df['commodity'].nunique())

    st.subheader("Date Range")
    st.write(f"From {cleaned_df['date'].min()} to {cleaned_df['date'].max()}")

    st.subheader("Sample Data")
    st.dataframe(cleaned_df.head(10))

    st.subheader("Top Commodities by Value")
    top_com = cleaned_df.groupby('commodity')['value_dl'].sum().nlargest(10)
    st.bar_chart(top_com)

# EDA Dashboard
elif page == "EDA Dashboard":
    st.title("🔍 EDA Dashboard")
    st.write("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Seasonal", "Comparison", "Geographic"])

    with tab1:
        st.subheader("Country & Commodity Growth Trend")
        all_countries = sorted(cleaned_df['country_name'].unique())
        selected_country = st.selectbox("Select Country", all_countries, key="eda_country")
        available_commodities = sorted(cleaned_df[cleaned_df['country_name'] == selected_country]['commodity'].unique())
        selected_commodity = st.selectbox("Select Commodity", available_commodities, key="eda_commodity")

        filtered_data = cleaned_df[
            (cleaned_df['country_name'] == selected_country) &
            (cleaned_df['commodity'] == selected_commodity)
        ]

        if not filtered_data.empty:
            monthly_trend = filtered_data.resample('M', on='date')['value_dl'].sum().reset_index()
            monthly_trend.columns = ['Month', 'Total Import Value (USD)']
            st.line_chart(monthly_trend.set_index('Month'))
        else:
            st.warning("No data.")

    with tab2:
        st.subheader("Seasonal Patterns")
        selected_commodity = st.selectbox("Select Commodity", sorted(cleaned_df['commodity'].unique()), key="seasonal")
        commodity_data = cleaned_df[cleaned_df['commodity'] == selected_commodity]

        if not commodity_data.empty:
            monthly = commodity_data.groupby(commodity_data['date'].dt.month)['value_dl'].sum().reset_index()
            monthly.columns = ['Month', 'Total Value']
            fig = px.bar(monthly, x='Month', y='Total Value', title=f"Monthly Imports for {selected_commodity}")
            st.plotly_chart(fig)

    with tab3:
        st.subheader("Multi-Country Comparison")
        selected_commodity = st.selectbox("Select Commodity", sorted(cleaned_df['commodity'].unique()), key="multi")
        selected_countries = st.multiselect("Select Countries", sorted(cleaned_df['country_name'].unique()), default=sorted(cleaned_df['country_name'].unique())[:3])

        if selected_countries:
            filtered = cleaned_df[(cleaned_df['commodity'] == selected_commodity) & (cleaned_df['country_name'].isin(selected_countries))]

            if not filtered.empty:
                yearly = filtered.groupby(['country_name', filtered['date'].dt.year])['value_dl'].sum().reset_index()
                yearly.columns = ['Country', 'Year', 'Value']
                fig = px.line(yearly, x='Year', y='Value', color='Country', title=f"Yearly Imports of {selected_commodity}")
                st.plotly_chart(fig)

    with tab4:
        st.subheader("Geographic Heatmap")
        country_totals = cleaned_df.groupby('country_name')['value_dl'].sum().reset_index()
        fig = px.bar(country_totals, x='country_name', y='value_dl', title="Import Values by Country")
        st.plotly_chart(fig)

# Models & Novelty Dashboard
elif page == "Models & Novelty Dashboard":
    st.title("🤖 Models & Novelty Dashboard")
    st.write("Forecasting, Anomalies, and Advanced Insights")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Forecasting", "Anomalies", "Risk Analysis", "Correlation", "Novelty"])

    with tab1:
        st.subheader("Time Series Forecast")
        top_commodities = cleaned_df.groupby('commodity')['value_dl'].sum().nlargest(15).index.tolist()
        selected_commodity = st.selectbox("Select Commodity", top_commodities, key="forecast")

        monthly_data = cleaned_df[cleaned_df['commodity'] == selected_commodity].resample('M', on='date')['value_dl'].sum().reset_index()
        monthly_data.columns = ['ds', 'y']

        if len(monthly_data) > 1:
            try:
                model = Prophet()
                model.fit(monthly_data)
                future = model.make_future_dataframe(periods=12, freq='M')
                forecast = model.predict(future)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=monthly_data['ds'], y=monthly_data['y'], mode='lines', name='Historical'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None, mode='lines', line_color='lightblue', name='Lower Bound'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line_color='lightblue', name='Upper Bound'))
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Forecasting failed: {str(e)}")
        else:
            st.warning("Insufficient data.")

    with tab2:
        st.subheader("Anomaly Detection")
        all_countries = sorted(cleaned_df['country_name'].unique())
        selected_country = st.selectbox("Select Country", all_countries, key="anomaly_country")
        available_commodities = sorted(cleaned_df[cleaned_df['country_name'] == selected_country]['commodity'].unique())
        selected_commodity = st.selectbox("Select Commodity", available_commodities, key="anomaly_commodity")

        group_data = cleaned_df[
            (cleaned_df['country_name'] == selected_country) &
            (cleaned_df['commodity'] == selected_commodity)
        ]

        if not group_data.empty:
            cleaned_df['unit_price'] = cleaned_df['value_dl'] / cleaned_df['value_qt']
            cleaned_df['unit_price'] = cleaned_df['unit_price'].replace([np.inf, -np.inf], np.nan).fillna(0)

            mean_price = group_data['unit_price'].mean()
            std_price = group_data['unit_price'].std()
            group_data['z_score'] = (group_data['unit_price'] - mean_price) / std_price
            anomalies = group_data[abs(group_data['z_score']) > 3]

            st.write(f"Anomalies: {len(anomalies)}")
            fig = px.scatter(group_data, x='date', y='unit_price', color=abs(group_data['z_score']) > 3,
                             color_discrete_map={True: 'red', False: 'blue'}, title="Anomalies Highlighted")
            st.plotly_chart(fig)

    with tab3:
        st.subheader("Supply Chain Risk")
        selected_commodity = st.selectbox("Select Commodity", sorted(cleaned_df['commodity'].unique()), key="risk")

        commodity_data = cleaned_df[cleaned_df['commodity'] == selected_commodity]

        if not commodity_data.empty:
            country_totals = commodity_data.groupby('country_name')['value_dl'].sum()
            total_value = country_totals.sum()
            concentration = (country_totals.max() / total_value) * 100

            monthly_totals = commodity_data.resample('M', on='date')['value_dl'].sum()
            yoy_changes = monthly_totals.pct_change(12).dropna()
            volatility = yoy_changes.std() * 100

            st.write(f"Dependency: {concentration:.2f}%, Volatility: {volatility:.2f}%")

            fig = px.scatter(x=[concentration], y=[volatility], text=[selected_commodity])
            fig.update_layout(xaxis_title="Dependency (%)", yaxis_title="Volatility (%)", title="Risk Matrix")
            st.plotly_chart(fig)

    with tab4:
        st.subheader("Correlation Explorer")
        top_commodities = cleaned_df.groupby('commodity')['value_dl'].sum().nlargest(20).index.tolist()
        monthly_data = cleaned_df[cleaned_df['commodity'].isin(top_commodities)].groupby(['date', 'commodity'])['value_dl'].sum().reset_index()
        monthly_data['date'] = monthly_data['date'].dt.to_period('M').dt.to_timestamp()

        pivot = monthly_data.pivot(index='date', columns='commodity', values='value_dl').fillna(0)

        if not pivot.empty and pivot.shape[1] > 1:
            corr_matrix = pivot.corr()
            fig = px.imshow(corr_matrix, title="Top 20 Commodities Correlation Matrix")
            st.plotly_chart(fig)

    with tab5:
        st.subheader("Novelty: Basket Forecasting & ESG")
        baskets = {
            "Scrap Metals": ['Aluminium Scrap', 'Brass Scrap', 'Copper Scrap', 'Iron And Steel Scrap'],
            "Oils": ['Petroleum Oils', 'Palm Oil'],
            "Gems": ['Gold', 'Diamonds']
        }

        selected_basket = st.selectbox("Select Basket", list(baskets.keys()))
        basket_commodities = baskets[selected_basket]
        basket_data = cleaned_df[cleaned_df['commodity'].isin(basket_commodities)]

        if not basket_data.empty:
            monthly_basket = basket_data.resample('M', on='date')['value_dl'].sum().reset_index()
            monthly_basket.columns = ['ds', 'y']

            if len(monthly_basket) > 1:
                try:
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

        st.subheader("Sustainability ESG")
        sustainable_commodities = ['Gold', 'Diamonds', 'Coffee']
        esg_data = cleaned_df[cleaned_df['commodity'].isin(sustainable_commodities)].groupby('commodity')['value_dl'].sum()
        fig = px.pie(values=esg_data.values, names=esg_data.index, title="Sustainable Commodities Share")
        st.plotly_chart(fig)
