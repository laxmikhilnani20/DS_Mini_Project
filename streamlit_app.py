import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from scipy.stats import zscore
import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="African Imports Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ----------------------------------------------------------------------
# File Paths
# ----------------------------------------------------------------------
DATA_FILE = 'imports-from-african-countries.csv'

# ----------------------------------------------------------------------
# Caching - Load Data and Models
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    """Loads the CLEANED data from CSV."""
    try:
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        # No need to fillna('unit') as the cleaned file already has this
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file '{DATA_FILE}' not found. Make sure it's in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

# Load data
cleaned_df = load_data()

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Data Dashboard",
    "EDA Dashboard",
    "Advanced Dashboards"
])

# ======================================================================
# Page 1: Data Dashboard
# ======================================================================
if page == "Data Dashboard":
    st.title("📊 Data Dashboard")
    st.write("Overview of Imports from African Countries Dataset")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(cleaned_df):,}")
    with col2:
        st.metric("Unique Countries", cleaned_df['country_name'].nunique())
    with col3:
        st.metric("Unique Commodities", cleaned_df['commodity'].nunique())

    st.subheader("Date Range")
    st.write(f"From {cleaned_df['date'].min().strftime('%Y-%m-%d')} to {cleaned_df['date'].max().strftime('%Y-%m-%d')}")

    st.subheader("Sample Data")
    st.dataframe(cleaned_df.head(10))

    st.subheader("Top Commodities by Value (USD)")
    top_com = cleaned_df.groupby('commodity')['value_dl'].sum().nlargest(10)
    st.bar_chart(top_com)

# ======================================================================
# Page 2: EDA Dashboard
# ======================================================================
elif page == "EDA Dashboard":
    st.title("🔍 EDA Dashboard")
    st.write("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Seasonal", "Comparison", "Geographic Map"])

    with tab1:
        st.subheader("Country & Commodity Growth Trend")
        all_countries = sorted(cleaned_df['country_name'].unique())
        
        # Set default index to avoid errors on first load
        try:
            default_country_index = all_countries.index("South Africa")
        except ValueError:
            default_country_index = 0
            
        selected_country = st.selectbox("Select Country", all_countries, key="eda_country", index=default_country_index)
        
        available_commodities = sorted(cleaned_df[cleaned_df['country_name'] == selected_country]['commodity'].unique())
        
        if available_commodities:
            try:
                default_comm_index = available_commodities.index("STEAM COAL")
            except ValueError:
                default_comm_index = 0
                
            selected_commodity = st.selectbox("Select Commodity", available_commodities, key="eda_commodity", index=default_comm_index)

            filtered_data = cleaned_df[
                (cleaned_df['country_name'] == selected_country) &
                (cleaned_df['commodity'] == selected_commodity)
            ]

            if not filtered_data.empty:
                monthly_trend = filtered_data.resample('M', on='date')['value_dl'].sum().reset_index()
                monthly_trend.columns = ['Month', 'Total Import Value (USD)']
                fig_trend = px.line(monthly_trend, x='Month', y='Total Import Value (USD)',
                                    title=f"{selected_commodity} from {selected_country}")
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.warning("No data found for this specific country and commodity combination.")
        else:
            st.warning(f"No commodities found for {selected_country}.")

    with tab2:
        st.subheader("Seasonal Patterns")
        all_commodities = sorted(cleaned_df['commodity'].unique())
        selected_commodity = st.selectbox("Select Commodity", all_commodities, key="seasonal")
        commodity_data = cleaned_df[cleaned_df['commodity'] == selected_commodity]

        if not commodity_data.empty:
            monthly = commodity_data.groupby(commodity_data['date'].dt.month_name())['value_dl'].sum().reset_index()
            monthly.columns = ['Month', 'Total Value']
            # Sort by month
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            monthly['Month'] = pd.Categorical(monthly['Month'], categories=month_order, ordered=True)
            monthly = monthly.sort_values('Month')
            
            fig = px.bar(monthly, x='Month', y='Total Value', title=f"Monthly Import Value for {selected_commodity}")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Multi-Country Comparison")
        selected_commodity = st.selectbox("Select Commodity", sorted(cleaned_df['commodity'].unique()), key="multi")
        
        default_countries = []
        all_countries = sorted(cleaned_df['country_name'].unique())
        if "South Africa" in all_countries: default_countries.append("South Africa")
        if "Nigeria" in all_countries: default_countries.append("Nigeria")
        if "Egypt" in all_countries: default_countries.append("Egypt")
        if not default_countries: default_countries = all_countries[:1]
        
        selected_countries = st.multiselect("Select Countries", all_countries, default=default_countries)

        if selected_countries:
            filtered = cleaned_df[(cleaned_df['commodity'] == selected_commodity) & (cleaned_df['country_name'].isin(selected_countries))]

            if not filtered.empty:
                yearly = filtered.groupby(['country_name', filtered['date'].dt.year])['value_dl'].sum().reset_index()
                yearly.columns = ['Country', 'Year', 'Value']
                fig = px.line(yearly, x='Year', y='Value', color='Country', title=f"Yearly Imports of {selected_commodity}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data for this combination.")

    with tab4:
        st.subheader("Geographic Import Map")
        st.write("Total Import Value (USD) by Country")
        
        # Group by both name and code for the hover data
        country_totals = cleaned_df.groupby(['country_name', 'alpha_3_code'])['value_dl'].sum().reset_index()

        fig = px.choropleth(
            country_totals,
            locations="alpha_3_code",  # Column with 3-letter country codes
            color="value_dl",          # Column with value to plot
            hover_name="country_name", # Column to show on hover
            color_continuous_scale=px.colors.sequential.Plasma,
            title="Total Import Value (USD) from African Countries"
        )
        fig.update_layout(geo=dict(scope='africa'))
        st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# Page 3: Advanced Dashboards
# ======================================================================
elif page == "Advanced Dashboards":
    st.title("🤖 Advanced Dashboards")
    st.write("Forecasting, Anomalies, and Advanced Insights")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Commodity Forecast", "Anomaly Detection", "Supply Chain Risk", "Correlation Matrix", "Basket Forecasting"])

    with tab1:
        st.subheader("Time Series Forecast")
        st.write("Forecast the next 12 months for top commodities using Prophet.")
        
        top_commodities = cleaned_df.groupby('commodity')['value_dl'].sum().nlargest(15).index.tolist()
        selected_commodity = st.selectbox("Select Commodity", top_commodities, key="forecast")

        monthly_data = cleaned_df[cleaned_df['commodity'] == selected_commodity].resample('M', on='date')['value_dl'].sum().reset_index()
        monthly_data.columns = ['ds', 'y']

        if len(monthly_data) > 1:
            try:
                with st.spinner("Training forecast model..."):
                    model = Prophet()
                    model.fit(monthly_data)
                    future = model.make_future_dataframe(periods=12, freq='M')
                    forecast = model.predict(future)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=monthly_data['ds'], y=monthly_data['y'], mode='lines', name='Historical'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None, mode='lines', line_color='rgba(255,0,0,0.2)', name='Lower Bound'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)', name='Upper Bound'))
                fig.update_layout(title=f"Forecast for {selected_commodity}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Forecasting failed: {str(e)}")
        else:
            st.warning("Insufficient data for forecasting.")

    with tab2:
        st.subheader("Anomaly Detection (Unit Price)")
        st.write("Finds transactions with a 'Unit Price' (Value / Quantity) that is statistically unusual for that specific commodity.")
        
        all_countries = sorted(cleaned_df['country_name'].unique())
        selected_country = st.selectbox("Select Country", all_countries, key="anomaly_country")
        
        available_commodities = sorted(cleaned_df[cleaned_df['country_name'] == selected_country]['commodity'].unique())
        if available_commodities:
            selected_commodity = st.selectbox("Select Commodity", available_commodities, key="anomaly_commodity")
            
            group_data = cleaned_df[
                (cleaned_df['country_name'] == selected_country) &
                (cleaned_df['commodity'] == selected_commodity)
            ].copy()

            if not group_data.empty:
                # Calculate unit price, handle potential division by zero
                group_data['unit_price'] = group_data['value_dl'] / group_data['value_qt']
                # Replace inf with nan, then drop rows where unit_price is nan or 0
                group_data = group_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['unit_price'])
                group_data = group_data[group_data['unit_price'] > 0]

                if not group_data.empty:
                    # Calculate Z-score
                    group_data['z_score'] = zscore(group_data['unit_price'], nan_policy='omit')
                    group_data['is_anomaly'] = (group_data['z_score'].abs() > 3) # Default
                    
                    # Interactive Threshold
                    threshold = st.slider("Select Z-Score Threshold", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
                    anomalies = group_data[group_data['z_score'].abs() > threshold]
                    
                    st.write(f"Found {len(anomalies)} anomalies with Z-Score > {threshold}")
                    
                    fig = px.scatter(group_data, x='date', y='unit_price', 
                                     color=group_data['z_score'].abs() > threshold,
                                     color_discrete_map={True: 'red', False: 'blue'}, 
                                     title="Unit Price Over Time (Anomalies Highlighted)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if not anomalies.empty:
                        with st.expander("View Anomalous Transactions"):
                            st.dataframe(anomalies[['date', 'value_qt', 'value_dl', 'unit_price', 'z_score']])
                else:
                    st.warning("No valid data (with quantity > 0) to calculate unit price.")
            else:
                st.warning("No data for this selection.")
        else:
            st.warning(f"No commodities found for {selected_country}.")

    with tab3:
        st.subheader("Supply Chain Risk Matrix")
        st.write("Compare commodities based on supply concentration (Dependency) and import value volatility.")
        
        all_commodities = sorted(cleaned_df['commodity'].unique())
        top_50_commodities = cleaned_df.groupby('commodity')['value_dl'].sum().nlargest(50).index.tolist()
        
        selected_commodities = st.multiselect("Select Commodities to Compare", all_commodities, default=top_50_commodities[:5])

        if selected_commodities:
            risk_data = []
            for commodity in selected_commodities:
                commodity_data = cleaned_df[cleaned_df['commodity'] == commodity]
                
                if not commodity_data.empty:
                    # 1. Concentration (Dependency)
                    country_totals = commodity_data.groupby('country_name')['value_dl'].sum()
                    total_value = country_totals.sum()
                    if total_value > 0:
                        concentration = (country_totals.max() / total_value) * 100
                    else:
                        concentration = 0
                    
                    # 2. Volatility
                    monthly_totals = commodity_data.resample('M', on='date')['value_dl'].sum()
                    yoy_changes = monthly_totals.pct_change(12).dropna() # Year-over-year % change
                    
                    if not yoy_changes.empty:
                        volatility = yoy_changes.std() * 100 # Volatility as std dev of YoY change
                    else:
                        volatility = 0
                        
                    risk_data.append({
                        "Commodity": commodity,
                        "Dependency (%)": concentration,
                        "Volatility (%)": volatility,
                        "Total Value (USD)": total_value
                    })
            
            if risk_data:
                risk_df = pd.DataFrame(risk_data)
                fig = px.scatter(
                    risk_df, 
                    x="Dependency (%)", 
                    y="Volatility (%)", 
                    text="Commodity",
                    size="Total Value (USD)",
                    hover_name="Commodity",
                    title="Risk Matrix: Dependency vs. Volatility"
                )
                fig.update_traces(textposition='top center')
                fig.update_layout(
                    xaxis_title="Dependency (Max % from one country)",
                    yaxis_title="Volatility (Std. Dev of YoY % Change)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("View Risk Data"):
                    st.dataframe(risk_df)
            else:
                st.warning("Could not calculate risk for selected commodities.")


    with tab4:
        st.subheader("Correlation Explorer")
        st.write("Correlation matrix of the top 20 commodities by import value.")
        
        top_commodities_list = cleaned_df.groupby('commodity')['value_dl'].sum().nlargest(20).index.tolist()
        
        # Resample to monthly, pivot
        monthly_data = cleaned_df[cleaned_df['commodity'].isin(top_commodities_list)].groupby(['date', 'commodity'])['value_dl'].sum().reset_index()
        monthly_data['date'] = monthly_data['date'].dt.to_period('M').dt.to_timestamp() # Ensure consistent monthly index
        pivot = monthly_data.pivot(index='date', columns='commodity', values='value_dl').fillna(0)

        if not pivot.empty and pivot.shape[1] > 1:
            corr_matrix = pivot.corr()
            fig = px.imshow(corr_matrix, text_auto=".2f", title="Top 20 Commodities Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Novelty: Basket Forecasting")
        st.write("Forecast the total value of a custom 'basket' of commodities.")
        
        baskets = {
            "Scrap Metals": ['ALUMINIUM SCRAP', 'BRASS SCRAP', 'COPPER SCRAP', 'IRON AND STEEL SCRAP'],
            "Oils (Non-Crude)": ['PETROLEUM OILS, ETC,(NOT CRUDE)', 'PALM OIL', 'SOYA-BEAN OIL'],
            "Precious": ['OTHER NON-MONETARY UNWROUGHT FORMS OF GOLD', 'GOLD DOR', 'ROUGH DIAMONDS']
        }

        selected_basket = st.selectbox("Select Basket", list(baskets.keys()))
        basket_commodities = baskets[selected_basket]
        st.write(f"**Commodities in this basket:** {', '.join(basket_commodities)}")
        
        basket_data = cleaned_df[cleaned_df['commodity'].isin(basket_commodities)]

        if not basket_data.empty:
            monthly_basket = basket_data.resample('M', on='date')['value_dl'].sum().reset_index()
            monthly_basket.columns = ['ds', 'y']

            if len(monthly_basket) > 1:
                try:
                    with st.spinner("Training basket forecast model..."):
                        model = Prophet()
                        model.fit(monthly_basket)
                        future = model.make_future_dataframe(periods=12, freq='M')
                        forecast = model.predict(future)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=monthly_basket['ds'], y=monthly_basket['y'], mode='lines', name='Historical'))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='red', dash='dash')))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None, mode='lines', line_color='rgba(255,0,0,0.2)', name='Lower Bound'))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)', name='Upper Bound'))
                    fig.update_layout(title=f"Forecast for {selected_basket} Basket")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Basket forecasting failed: {str(e)}")
            else:
                st.warning("Insufficient data for basket forecast.")

# ======================================================================
# Page 4: Advanced Dashboards (continued)
# ======================================================================

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
        ].copy()  # Use copy to avoid issues

        if not group_data.empty:
            group_data['unit_price'] = group_data['value_dl'] / group_data['value_qt']
            group_data['unit_price'] = group_data['unit_price'].replace([np.inf, -np.inf], np.nan).fillna(0)

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
