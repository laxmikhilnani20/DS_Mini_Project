import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore
from datetime import datetime
import warnings
import io

# Try to import Prophet, but don't fail if it's not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("⚠️ Prophet not available. Forecasting features will be disabled.")

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
# CUSTOM CSS
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
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
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
        "🎯 Interactive Models"
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
# DASHBOARD 1: DATA OVERVIEW
# ======================================================================
if page == "📊 Data Overview":
    st.markdown('<p class="main-header">📊 Data Overview Dashboard</p>', unsafe_allow_html=True)
    st.markdown("### Complete Dataset Analysis & Export")
    
    # Smart formatting function for currency
    def format_currency(value):
        if value >= 1e9:
            return f"${value/1e9:.2f}B"
        elif value >= 1e6:
            return f"${value/1e6:.2f}M"
        elif value >= 1e3:
            return f"${value/1e3:.2f}K"
        else:
            return f"${value:.2f}"
    
    # === KEY METRICS ===
    st.markdown("## 📈 Key Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Countries", f"{df['country_name'].nunique()}")
    with col3:
        st.metric("Commodities", f"{df['commodity'].nunique():,}")
    with col4:
        total_value = df['value_dl'].sum()
        st.metric("Total Value", format_currency(total_value))
    with col5:
        date_range = (df['date'].max() - df['date'].min()).days / 365.25
        st.metric("Years Span", f"{date_range:.1f}")
    
    # === DATASET DESCRIPTION ===
    st.markdown("## 📖 Dataset Description")
    
    st.markdown("""
    <div class="insight-box">
    <h4>About This Dataset</h4>
    <p>This dataset contains <b>detailed import transaction records</b> from African countries, covering a wide range of commodities 
    and trade partners. It provides comprehensive insights into:</p>
    <ul>
        <li><b>Temporal Patterns:</b> Import trends from 2015 to 2025, including COVID-19 impact periods</li>
        <li><b>Geographic Distribution:</b> Trade flows across Northern, Sub-Saharan, Eastern, Western, Southern, and Central Africa</li>
        <li><b>Commodity Analysis:</b> 4,000+ product categories from petroleum and minerals to manufactured goods</li>
        <li><b>Economic Insights:</b> Values in multiple currencies (USD, local currency) and various units of measurement</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # === DATA SCHEMA ===
    st.markdown("## 🗂️ Data Schema")
    
    schema_col1, schema_col2 = st.columns(2)
    
    with schema_col1:
        st.markdown("""
        **Key Fields:**
        - `date`: Transaction date
        - `country_name`: Importing country
        - `commodity`: Product description
        - `value_dl`: Import value (USD)
        - `value_rs`: Import value (local currency)
        - `value_qt`: Quantity imported
        - `unit`: Measurement unit
        """)
    
    with schema_col2:
        st.markdown("""
        **Geographic Fields:**
        - `alpha_3_code`: ISO 3-letter country code
        - `region`: Major African region
        - `sub_region`: Detailed sub-region
        - `hs_code`: Harmonized System code
        """)
    
    # === TIME RANGE INFO ===
    st.markdown("## 📅 Temporal Coverage")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Start Date:** {df['date'].min().strftime('%Y-%m-%d')}")
    with col2:
        st.info(f"**End Date:** {df['date'].max().strftime('%Y-%m-%d')}")
    with col3:
        st.info(f"**Total Months:** {len(df.groupby(df['date'].dt.to_period('M')))}")
    
    # === DATA VIEWER WITH FILTERS ===
    st.markdown("## 🔍 Data Explorer")
    
    with st.expander("🎛️ Filter Options", expanded=True):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            selected_countries = st.multiselect(
                "Select Countries",
                options=sorted(df['country_name'].unique()),
                default=[]
            )
        
        with filter_col2:
            selected_years = st.multiselect(
                "Select Years",
                options=sorted(df['year'].unique()),
                default=[]
            )
        
        with filter_col3:
            value_threshold = st.number_input(
                "Min. Value (USD)",
                min_value=0,
                value=0,
                step=1000
            )
    
    # Apply filters
    filtered_df = df.copy()
    if selected_countries:
        filtered_df = filtered_df[filtered_df['country_name'].isin(selected_countries)]
    if selected_years:
        filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
    if value_threshold > 0:
        filtered_df = filtered_df[filtered_df['value_dl'] >= value_threshold]
    
    st.markdown(f"**Showing {len(filtered_df):,} records** (filtered from {len(df):,} total)")
    
    # Display data
    st.dataframe(
        filtered_df[['date', 'country_name', 'commodity', 'value_dl', 'value_qt', 'unit', 'sub_region']].head(1000),
        use_container_width=True,
        height=400
    )
    
    if len(filtered_df) > 1000:
        st.warning("⚠️ Showing first 1,000 records. Use filters or download full data below.")
    
    # === DOWNLOAD SECTION ===
    st.markdown("## 💾 Download Reports")
    
    download_col1, download_col2, download_col3 = st.columns(3)
    
    with download_col1:
        # Full dataset
        csv_full = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_full,
            file_name=f"african_imports_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with download_col2:
        # Summary statistics
        summary = filtered_df.groupby('country_name').agg({
            'value_dl': ['sum', 'mean', 'count'],
            'commodity': 'nunique'
        }).reset_index()
        summary.columns = ['Country', 'Total_Value', 'Avg_Value', 'Transaction_Count', 'Unique_Commodities']
        
        csv_summary = summary.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Summary Report (CSV)",
            data=csv_summary,
            file_name=f"summary_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with download_col3:
        # Commodity rankings
        commodity_summary = filtered_df.groupby('commodity').agg({
            'value_dl': 'sum',
            'value_qt': 'sum',
            'country_name': 'nunique'
        }).reset_index().sort_values('value_dl', ascending=False).head(100)
        commodity_summary.columns = ['Commodity', 'Total_Value_USD', 'Total_Quantity', 'Country_Count']
        
        csv_commodity = commodity_summary.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Top 100 Commodities (CSV)",
            data=csv_commodity,
            file_name=f"top_commodities_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # === QUICK INSIGHTS ===
    st.markdown("## 💡 Quick Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        top_country = df.groupby('country_name')['value_dl'].sum().idxmax()
        top_country_value = df.groupby('country_name')['value_dl'].sum().max()
        st.success(f"**Top Importing Country:** {top_country} ({format_currency(top_country_value)})")
        
        top_commodity = df.groupby('commodity')['value_dl'].sum().idxmax()
        st.success(f"**Top Commodity:** {top_commodity}")
    
    with insight_col2:
        most_diverse = df.groupby('country_name')['commodity'].nunique().idxmax()
        diversity_count = df.groupby('country_name')['commodity'].nunique().max()
        st.info(f"**Most Diverse Imports:** {most_diverse} ({diversity_count:,} commodities)")
        
        avg_transaction = df['value_dl'].mean()
        st.info(f"**Average Transaction Value:** {format_currency(avg_transaction)}")


# ======================================================================
# DASHBOARD 2: EDA EXPLORER
# ======================================================================
elif page == "🔍 EDA Explorer":
    st.markdown('<p class="main-header">🔍 Interactive EDA Explorer</p>', unsafe_allow_html=True)
    st.markdown("### Deep-Dive Analysis: Select Country + Commodity for 8 Comprehensive Visualizations")
    
    # === INTERACTIVE SELECTORS ===
    st.markdown("## 🎛️ Select Your Analysis Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌍 Select Country")
        all_countries_eda = sorted(df['country_name'].unique())
        selected_country_eda = st.selectbox(
            "Choose a country to analyze:",
            options=all_countries_eda,
            index=0,
            key="eda_country_select"
        )
    
    with col2:
        st.markdown("#### 📦 Select Commodity")
        available_commodities_eda = sorted(
            df[df['country_name'] == selected_country_eda]['commodity'].unique()
        )
        
        if available_commodities_eda:
            selected_commodity_eda = st.selectbox(
                "Choose a commodity to analyze:",
                options=available_commodities_eda,
                index=0,
                key="eda_commodity_select"
            )
        else:
            st.error("No commodities available for selected country")
            st.stop()
    
    # Filter data for selected combination
    eda_filtered = df[
        (df['country_name'] == selected_country_eda) &
        (df['commodity'] == selected_commodity_eda)
    ].copy()
    
    if eda_filtered.empty:
        st.warning("⚠️ No data available for this country-commodity combination.")
        st.stop()
    
    # === SUMMARY METRICS ===
    st.markdown("---")
    st.markdown(f"## 📊 Analyzing: **{selected_commodity_eda}** from **{selected_country_eda}**")
    
    # Smart formatting function for currency
    def format_currency(value):
        if value >= 1e9:
            return f"${value/1e9:.2f}B"
        elif value >= 1e6:
            return f"${value/1e6:.2f}M"
        elif value >= 1e3:
            return f"${value/1e3:.2f}K"
        else:
            return f"${value:.2f}"
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    total_value = eda_filtered['value_dl'].sum()
    avg_value = eda_filtered['value_dl'].mean()
    
    with metric_col1:
        st.metric("Total Transactions", f"{len(eda_filtered):,}")
    with metric_col2:
        st.metric("Total Value (USD)", format_currency(total_value))
    with metric_col3:
        st.metric("Avg Transaction", format_currency(avg_value))
    with metric_col4:
        date_span = (eda_filtered['date'].max() - eda_filtered['date'].min()).days
        st.metric("Time Span (Days)", f"{date_span}")
    
    st.markdown("---")
    st.markdown("## 📈 8 Comprehensive Visual Analyses with Data Scientist Insights")
    
    # Create tabs for different EDA aspects
    eda_tabs = st.tabs([
        "📈 Time Series",
        "🌍 Geographic Analysis", 
        "📦 Commodity Insights",
        "📊 Distributions",
        "🔗 Correlations",
        "📉 Trends & Patterns"
    ])
    
    # ===== TAB 1: TIME SERIES =====
    with eda_tabs[0]:
        st.markdown("### 1️⃣ Monthly Import Trend Over Time")
        
        monthly_eda = eda_filtered.resample('M', on='date')['value_dl'].sum().reset_index()
        monthly_eda.columns = ['Date', 'Value']
        
        fig_monthly_eda = px.line(
            monthly_eda,
            x='Date',
            y='Value',
            title=f'Monthly Trend: {selected_commodity_eda} from {selected_country_eda}',
            labels={'Value': 'Import Value (USD)'},
            markers=True
        )
        fig_monthly_eda.update_traces(line_color='#1E88E5', line_width=2.5)
        st.plotly_chart(fig_monthly_eda, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Data Scientist Perspective:</h4>
        <p><b>What this shows:</b> Monthly time series of import values revealing temporal patterns and trends.</p>
        <p><b>Key insights to look for:</b> Trend direction (growing/declining), volatility (supply/demand stability), 
        cyclical patterns (seasonal business cycles), and anomalies (policy changes, market shocks).</p>
        <p><b>Why it matters:</b> Time series analysis is fundamental for forecasting, inventory planning, and detecting market disruptions early.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Year-over-Year comparison
        st.markdown("---")
        st.markdown("### 2️⃣ Year-over-Year Growth Analysis")
        yearly_eda = eda_filtered.groupby('year')['value_dl'].sum().reset_index()
        yearly_eda.columns = ['Year', 'Total Value']
        yearly_eda['YoY %'] = yearly_eda['Total Value'].pct_change() * 100
        
        col1, col2 = st.columns(2)
        with col1:
            fig_yearly_eda = px.bar(
                yearly_eda,
                x='Year',
                y='Total Value',
                title='Yearly Import Value',
                text_auto='.2s',
                color='Total Value',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_yearly_eda, use_container_width=True)
        
        with col2:
            if len(yearly_eda.dropna()) > 0:
                fig_yoy_eda = px.line(
                    yearly_eda.dropna(),
                    x='Year',
                    y='YoY %',
                    title='Year-over-Year Growth Rate (%)',
                    markers=True
                )
                fig_yoy_eda.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_yoy_eda.update_traces(line_color='#FF6B6B', line_width=3)
                st.plotly_chart(fig_yoy_eda, use_container_width=True)
            else:
                st.info("Insufficient years for YoY analysis")
        
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Data Scientist Perspective:</h4>
        <p><b>What this shows:</b> Annual aggregation and growth rates showing long-term trends.</p>
        <p><b>Key insights:</b> Consistent positive YoY% indicates sustained market expansion. Large swings signal instability or external shocks.</p>
        <p><b>Why it matters:</b> YoY analysis removes seasonal noise and helps identify long-term strategic trends for policy/investment decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Seasonal patterns
        st.markdown("---")
        st.markdown("### 3️⃣ Seasonal Patterns & Monthly Breakdown")
        eda_filtered_season = eda_filtered.copy()
        eda_filtered_season['Month'] = eda_filtered_season['date'].dt.month_name()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        seasonal_eda = eda_filtered_season.groupby('Month')['value_dl'].agg(['sum', 'mean']).reindex(month_order).reset_index()
        seasonal_eda.columns = ['Month', 'Total', 'Average']
        
        fig_seasonal_eda = px.bar(
            seasonal_eda,
            x='Month',
            y='Total',
            title='Seasonal Import Patterns by Month',
            labels={'Total': 'Total Import Value (USD)'},
            color='Total',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_seasonal_eda, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Data Scientist Perspective:</h4>
        <p><b>What this shows:</b> Aggregated import values across calendar months revealing seasonal business cycles.</p>
        <p><b>Key insights:</b> Peak months indicate high-demand periods (harvest seasons, holidays). Large variations = strong seasonal effects requiring inventory planning.</p>
        <p><b>Why it matters:</b> Understanding seasonality enables better logistics planning, working capital management, and demand forecasting.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quarterly Analysis
        st.markdown("---")
        st.markdown("### 4️⃣ Quarterly Performance Analysis")
        quarterly_eda = eda_filtered.groupby(['year', 'quarter'])['value_dl'].sum().reset_index()
        quarterly_eda['Year-Quarter'] = quarterly_eda['year'].astype(str) + '-Q' + quarterly_eda['quarter'].astype(str)
        
        fig_quarterly_eda = px.bar(
            quarterly_eda,
            x='Year-Quarter',
            y='value_dl',
            title='Quarterly Import Trends',
            labels={'value_dl': 'Import Value (USD)'},
            color='quarter',
            color_continuous_scale='Teal'
        )
        st.plotly_chart(fig_quarterly_eda, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Data Scientist Perspective:</h4>
        <p><b>What this shows:</b> Quarterly aggregation smooths monthly noise while preserving seasonal patterns.</p>
        <p><b>Key insights:</b> Quarter-on-Quarter trends are more stable than monthly. Same quarters across years show consistency or structural changes.</p>
        <p><b>Why it matters:</b> Quarterly analysis aligns with business reporting cycles and balances short-term volatility with long-term trends.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== TAB 2: GEOGRAPHIC ANALYSIS =====
    with eda_tabs[1]:
        st.markdown("### 🌍 Geographic Distribution")
        
        # Top countries
        st.markdown("#### Top Countries by Import Value")
        total_countries = df['country_name'].nunique()
        st.info(f"📊 Total countries available: **{total_countries}**")
        top_n_countries = st.slider("Select number of countries to display:", min_value=5, max_value=total_countries, value=min(15, total_countries), step=5, key="top_countries_slider")
        
        country_totals = df.groupby('country_name')['value_dl'].sum().sort_values(ascending=False).head(top_n_countries).reset_index()
        country_totals.columns = ['Country', 'Total Value (USD)']
        
        fig_countries = px.bar(
            country_totals,
            y='Country',
            x='Total Value (USD)',
            orientation='h',
            title=f'Top {top_n_countries} Importing Countries',
            text_auto='.2s'
        )
        st.plotly_chart(fig_countries, use_container_width=True)
        
        # Geographic map
        st.markdown("#### Import Value Heatmap")
        country_map = df.groupby(['country_name', 'alpha_3_code'])['value_dl'].sum().reset_index()
        
        fig_map = px.choropleth(
            country_map,
            locations='alpha_3_code',
            color='value_dl',
            hover_name='country_name',
            color_continuous_scale='Viridis',
            title='Geographic Distribution of Import Values',
            labels={'value_dl': 'Total Value (USD)'}
        )
        fig_map.update_geos(scope='africa')
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Regional breakdown
        st.markdown("#### Regional Analysis")
        if 'sub_region' in df.columns:
            regional = df.groupby('sub_region')['value_dl'].sum().sort_values(ascending=False).reset_index()
            regional.columns = ['Sub-Region', 'Total Value (USD)']
            
            fig_regional = px.pie(
                regional,
                values='Total Value (USD)',
                names='Sub-Region',
                title='Import Distribution by Sub-Region'
            )
            st.plotly_chart(fig_regional, use_container_width=True)
    
    # ===== TAB 3: COMMODITY INSIGHTS =====
    with eda_tabs[2]:
        st.markdown("### 📦 Commodity Analysis")
        
        # Top commodities
        st.markdown("#### Top Commodities by Value")
        total_commodities = df['commodity'].nunique()
        st.info(f"📊 Total commodities available: **{total_commodities}**")
        top_n_commodities = st.slider("Select number of commodities to display:", min_value=5, max_value=total_commodities, value=min(20, total_commodities), step=5, key="top_commodities_slider")
        
        commodity_totals = df.groupby('commodity')['value_dl'].sum().sort_values(ascending=False).head(top_n_commodities).reset_index()
        commodity_totals.columns = ['Commodity', 'Total Value (USD)']
        
        fig_commodities = px.bar(
            commodity_totals,
            y='Commodity',
            x='Total Value (USD)',
            orientation='h',
            title=f'Top {top_n_commodities} Commodities',
            text_auto='.2s',
            height=max(400, top_n_commodities * 25)  # Dynamic height based on number of items
        )
        st.plotly_chart(fig_commodities, use_container_width=True)
        
        # Commodity diversity
        st.markdown("#### Commodity Diversity by Country")
        st.info(f"📊 Total countries available: **{total_countries}**")
        top_n_diversity = st.slider("Select number of countries to display:", min_value=5, max_value=total_countries, value=min(15, total_countries), step=5, key="diversity_slider")
        
        diversity = df.groupby('country_name')['commodity'].nunique().sort_values(ascending=False).head(top_n_diversity).reset_index()
        diversity.columns = ['Country', 'Unique Commodities']
        
        fig_diversity = px.bar(
            diversity,
            x='Country',
            y='Unique Commodities',
            title=f'Top {top_n_diversity} Countries by Commodity Diversity',
            text_auto=True
        )
        st.plotly_chart(fig_diversity, use_container_width=True)
        
        st.info("📊 **Insight**: Higher commodity diversity indicates more complex import portfolios and varied industrial needs.")
    
    # ===== TAB 4: DISTRIBUTIONS =====
    with eda_tabs[3]:
        st.markdown("### 5️⃣ Transaction Value Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist_eda = px.histogram(
                eda_filtered,
                x='value_dl',
                nbins=30,
                title='Distribution of Transaction Values',
                labels={'value_dl': 'Transaction Value (USD)'},
                marginal='box'
            )
            st.plotly_chart(fig_hist_eda, use_container_width=True)
        
        with col2:
            fig_box_eda = px.box(
                eda_filtered,
                y='value_dl',
                title='Statistical Summary (Box Plot)',
                labels={'value_dl': 'Transaction Value (USD)'}
            )
            st.plotly_chart(fig_box_eda, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Data Scientist Perspective:</h4>
        <p><b>What this shows:</b> Statistical distribution of individual transaction values revealing data characteristics.</p>
        <p><b>Key insights:</b> Right-skewed = few large transactions dominate. Box plot whiskers show outliers requiring investigation.</p>
        <p><b>Why it matters:</b> Distribution shape informs pricing strategies, risk assessment, and whether to use median or mean for KPIs.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Moving Averages
        st.markdown("---")
        st.markdown("### 6️⃣ Trend Smoothing with Moving Averages")
        
        if len(monthly_eda) >= 3:
            monthly_eda['MA_3'] = monthly_eda['Value'].rolling(window=3, min_periods=1).mean()
            if len(monthly_eda) >= 6:
                monthly_eda['MA_6'] = monthly_eda['Value'].rolling(window=6, min_periods=1).mean()
            
            fig_ma_eda = go.Figure()
            fig_ma_eda.add_trace(go.Scatter(x=monthly_eda['Date'], y=monthly_eda['Value'], 
                                           name='Actual', line=dict(color='lightgray', width=1)))
            fig_ma_eda.add_trace(go.Scatter(x=monthly_eda['Date'], y=monthly_eda['MA_3'], 
                                           name='3-Month MA', line=dict(color='#1E88E5', width=2)))
            if len(monthly_eda) >= 6:
                fig_ma_eda.add_trace(go.Scatter(x=monthly_eda['Date'], y=monthly_eda['MA_6'], 
                                               name='6-Month MA', line=dict(color='#FF6B6B', width=2.5)))
            fig_ma_eda.update_layout(title='Trend Analysis with Moving Averages', 
                                     xaxis_title='Date', yaxis_title='Import Value (USD)')
            st.plotly_chart(fig_ma_eda, use_container_width=True)
        else:
            st.info("Insufficient data points for moving average calculation")
        
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Data Scientist Perspective:</h4>
        <p><b>What this shows:</b> Smoothed trends removing short-term noise to reveal underlying direction.</p>
        <p><b>Key insights:</b> Moving averages filter random fluctuations. Crossovers signal trend reversals.</p>
        <p><b>Why it matters:</b> Essential for distinguishing real market shifts from random variation in strategic planning.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quantity vs Value
        st.markdown("---")
        st.markdown("### 7️⃣ Quantity vs Value Relationship")
        
        if (eda_filtered['value_qt'] > 0).any():
            valid_qty_data = eda_filtered[eda_filtered['value_qt'] > 0].copy()
            valid_qty_data['unit_price'] = valid_qty_data['value_dl'] / valid_qty_data['value_qt']
            
            fig_scatter_eda = px.scatter(
                valid_qty_data.sample(min(500, len(valid_qty_data))),
                x='value_qt',
                y='value_dl',
                title='Quantity vs Value Scatter (Price Analysis)',
                labels={'value_qt': 'Quantity', 'value_dl': 'Value (USD)'},
                color='unit_price',
                color_continuous_scale='Plasma',
                hover_data=['date', 'unit_price']
            )
            st.plotly_chart(fig_scatter_eda, use_container_width=True)
            
            if len(valid_qty_data) > 0:
                price_trend = valid_qty_data.resample('M', on='date')['unit_price'].mean().reset_index()
                fig_price_eda = px.line(
                    price_trend,
                    x='date',
                    y='unit_price',
                    title='Average Unit Price Trend Over Time',
                    labels={'unit_price': 'Unit Price (USD)', 'date': 'Date'},
                    markers=True
                )
                st.plotly_chart(fig_price_eda, use_container_width=True)
        else:
            st.info("No quantity data available for this selection")
        
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Data Scientist Perspective:</h4>
        <p><b>What this shows:</b> Relationship between quantities and values, revealing pricing dynamics.</p>
        <p><b>Key insights:</b> Linear relationship = stable pricing. Decreasing unit price with quantity = bulk discounts.</p>
        <p><b>Why it matters:</b> Critical for procurement negotiations, inflation adjustment, and quality-price optimization.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cumulative Growth
        st.markdown("---")
        st.markdown("### 8️⃣ Cumulative Import Value Growth")
        
        cumulative_eda = monthly_eda.copy()
        cumulative_eda['Cumulative Value'] = cumulative_eda['Value'].cumsum()
        
        fig_cumulative_eda = px.area(
            cumulative_eda,
            x='Date',
            y='Cumulative Value',
            title='Cumulative Import Value Over Time',
            labels={'Cumulative Value': 'Total Cumulative Value (USD)'}
        )
        fig_cumulative_eda.update_traces(fillcolor='rgba(30,136,229,0.3)', line_color='#1E88E5')
        st.plotly_chart(fig_cumulative_eda, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Data Scientist Perspective:</h4>
        <p><b>What this shows:</b> Running total of import values showing cumulative business impact over time.</p>
        <p><b>Key insights:</b> Steepening curve = accelerating imports. Total value exchanged over entire period.</p>
        <p><b>Why it matters:</b> Shows total economic impact and helps identify inflection points in trade relationships.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Box plots by region
        if 'sub_region' in df.columns:
            st.markdown("#### Value Distribution by Sub-Region")
            fig_box = px.box(
                df[df['value_dl'] < df['value_dl'].quantile(0.95)],
                x='sub_region',
                y='value_dl',
                title='Import Value Distribution by Sub-Region',
                labels={'value_dl': 'Value (USD)', 'sub_region': 'Sub-Region'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
    
    # ===== TAB 5: CORRELATIONS =====
    with eda_tabs[4]:
        st.markdown("### 🔗 Correlation Analysis")
        
        st.markdown("#### Top Commodities Correlation Matrix")
        total_commodities_corr = df['commodity'].nunique()
        st.info(f"📊 Total commodities available: **{total_commodities_corr}**")
        top_n_corr = st.slider("Select number of commodities for correlation:", min_value=5, max_value=min(50, total_commodities_corr), value=min(15, total_commodities_corr), step=5, key="correlation_slider")
        st.write(f"Analyzing correlation between import patterns of top {top_n_corr} commodities")
        
        # Get top commodities
        top_commodities = df.groupby('commodity')['value_dl'].sum().nlargest(top_n_corr).index.tolist()
        
        # Create monthly pivot for correlation
        corr_data = df[df['commodity'].isin(top_commodities)].copy()
        corr_data['month_year'] = corr_data['date'].dt.to_period('M').dt.to_timestamp()
        
        pivot_corr = corr_data.pivot_table(
            index='month_year',
            columns='commodity',
            values='value_dl',
            aggfunc='sum'
        ).fillna(0)
        
        if not pivot_corr.empty and pivot_corr.shape[1] > 1:
            corr_matrix = pivot_corr.corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto='.2f',
                title='Commodity Import Correlation Matrix',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.info("📊 **Insight**: Positive correlation (red) suggests commodities imported together; negative (blue) suggests inverse patterns.")
    
    # ===== TAB 6: TRENDS & PATTERNS =====
    with eda_tabs[5]:
        st.markdown("### 📉 Advanced Patterns")
        
        # Multi-country comparison
        st.markdown("#### Multi-Country Trend Comparison")
        
        # Select countries
        all_countries = sorted(df['country_name'].unique())
        selected_compare = st.multiselect(
            "Select countries to compare:",
            options=all_countries,
            default=[all_countries[0], all_countries[1]] if len(all_countries) > 1 else [all_countries[0]]
        )
        
        if selected_compare:
            compare_data = df[df['country_name'].isin(selected_compare)].copy()
            compare_monthly = compare_data.groupby([compare_data['date'].dt.to_period('M').dt.to_timestamp(), 'country_name'])['value_dl'].sum().reset_index()
            compare_monthly.columns = ['Date', 'Country', 'Value']
            
            fig_compare = px.line(
                compare_monthly,
                x='Date',
                y='Value',
                color='Country',
                title='Import Value Comparison Over Time',
                labels={'Value': 'Total Value (USD)'}
            )
            st.plotly_chart(fig_compare, use_container_width=True)
        
        # Concentration analysis
        st.markdown("#### Market Concentration")
        st.write("Analyzing if imports are concentrated in few countries/commodities (80-20 rule)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            country_conc = df.groupby('country_name')['value_dl'].sum().sort_values(ascending=False).reset_index()
            country_conc['cumulative_pct'] = (country_conc['value_dl'].cumsum() / country_conc['value_dl'].sum()) * 100
            country_conc['rank'] = range(1, len(country_conc) + 1)
            
            fig_conc = px.line(
                country_conc,
                x='rank',
                y='cumulative_pct',
                title='Country Concentration (Pareto Chart)',
                labels={'rank': 'Country Rank', 'cumulative_pct': 'Cumulative % of Total Value'}
            )
            fig_conc.add_hline(y=80, line_dash="dash", annotation_text="80%")
            st.plotly_chart(fig_conc, use_container_width=True)
        
        with col2:
            commodity_conc = df.groupby('commodity')['value_dl'].sum().sort_values(ascending=False).reset_index()
            commodity_conc['cumulative_pct'] = (commodity_conc['value_dl'].cumsum() / commodity_conc['value_dl'].sum()) * 100
            commodity_conc['rank'] = range(1, len(commodity_conc) + 1)
            
            fig_comm_conc = px.line(
                commodity_conc,
                x='rank',
                y='cumulative_pct',
                title='Commodity Concentration (Pareto Chart)',
                labels={'rank': 'Commodity Rank', 'cumulative_pct': 'Cumulative % of Total Value'}
            )
            fig_comm_conc.add_hline(y=80, line_dash="dash", annotation_text="80%")
            st.plotly_chart(fig_comm_conc, use_container_width=True)

# ======================================================================
# DASHBOARD 3: INTERACTIVE MACHINE LEARNING MODELS
# ======================================================================
elif page == "🎯 Interactive Models":
    st.markdown('<p class="main-header">🎯 Machine Learning Models</p>', unsafe_allow_html=True)
    st.markdown("### Regression, Classification & Clustering Models for Trade Analysis")
    
    # Import ML libraries
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
        silhouette_score
    )
    from sklearn.decomposition import PCA
    import xgboost as xgb
    
    # === SELECTION PANEL ===
    st.markdown("## 🎛️ Data Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌍 Select Country (Optional)")
        all_countries_ml = ['All Countries'] + sorted(df['country_name'].unique().tolist())
        selected_country_ml = st.selectbox(
            "Choose a country or analyze all:",
            options=all_countries_ml,
            index=0,
            key="ml_country"
        )
    
    with col2:
        st.markdown("#### 📦 Select Commodity (Optional)")
        if selected_country_ml == 'All Countries':
            available_commodities_ml = ['All Commodities'] + sorted(df['commodity'].unique().tolist())
        else:
            available_commodities_ml = ['All Commodities'] + sorted(
                df[df['country_name'] == selected_country_ml]['commodity'].unique().tolist()
            )
        
        selected_commodity_ml = st.selectbox(
            "Choose a commodity or analyze all:",
            options=available_commodities_ml,
            index=0,
            key="ml_commodity"
        )
    
    # Filter data based on selection
    ml_data = df.copy()
    if selected_country_ml != 'All Countries':
        ml_data = ml_data[ml_data['country_name'] == selected_country_ml]
    if selected_commodity_ml != 'All Commodities':
        ml_data = ml_data[ml_data['commodity'] == selected_commodity_ml]
    
    if ml_data.empty:
        st.warning("⚠️ No data available for this selection.")
        st.stop()
    
    # Display dataset info
    st.markdown("---")
    st.markdown(f"## 📊 Selected Dataset: **{len(ml_data):,}** transactions")
    
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.metric("Countries", ml_data['country_name'].nunique())
    with info_col2:
        st.metric("Commodities", ml_data['commodity'].nunique())
    with info_col3:
        st.metric("Total Value", f"${ml_data['value_dl'].sum()/1e9:.2f}B")
    with info_col4:
        st.metric("Date Range", f"{ml_data['year'].min()}-{ml_data['year'].max()}")
    
    st.markdown("---")
    
    # === ML MODEL TABS ===
    ml_tabs = st.tabs([
        "📈 Regression Models",
        "🎯 Classification Models", 
        "🔍 Clustering Models"
    ])
    
    # ========================================================================
    # TAB 1: REGRESSION MODELS
    # ========================================================================
    with ml_tabs[0]:
        st.markdown("### 📈 Regression: Predict Import Values")
        
        st.markdown("""
        <div class="insight-box">
        <h4>🎓 What is Regression?</h4>
        <p>Regression models predict <b>continuous numerical values</b>. Here, we predict future import values based on historical patterns.</p>
        <p><b>Use Cases:</b> Demand forecasting, budget planning, price prediction</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Prepare regression data - monthly aggregation
        regression_data = ml_data.groupby(ml_data['date'].dt.to_period('M')).agg({
            'value_dl': 'sum',
            'value_qt': 'sum'
        }).reset_index()
        regression_data['date'] = regression_data['date'].dt.to_timestamp()
        regression_data = regression_data.sort_values('date')
        
        # Create features
        regression_data['month'] = regression_data['date'].dt.month
        regression_data['year'] = regression_data['date'].dt.year
        regression_data['quarter'] = regression_data['date'].dt.quarter
        regression_data['days_since_start'] = (regression_data['date'] - regression_data['date'].min()).dt.days
        
        # Lag features
        regression_data['value_lag1'] = regression_data['value_dl'].shift(1)
        regression_data['value_lag3'] = regression_data['value_dl'].shift(3)
        regression_data['value_rolling_mean_3'] = regression_data['value_dl'].rolling(window=3, min_periods=1).mean()
        
        regression_data = regression_data.dropna()
        
        if len(regression_data) < 10:
            st.warning("⚠️ Insufficient data for regression modeling (need at least 10 monthly records)")
        else:
            # Features and target
            feature_cols = ['month', 'quarter', 'days_since_start', 'value_lag1', 'value_lag3', 'value_rolling_mean_3']
            X = regression_data[feature_cols]
            y = regression_data['value_dl']
            
            # Train-test split
            test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5, key="reg_test_size") / 100
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            
            st.info(f"📊 Training on {len(X_train)} samples, Testing on {len(X_test)} samples")
            
            # Train models
            st.markdown("#### 🤖 Model Comparison")
            
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            }
            
            results = {}
            predictions = {}
            
            with st.spinner("Training models..."):
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    results[name] = {
                        'R² Score': r2_score(y_test, y_pred),
                        'MAE': mean_absolute_error(y_test, y_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
                    }
                    predictions[name] = y_pred
            
            # Display results
            results_df = pd.DataFrame(results).T
            results_df = results_df.round(4)
            st.dataframe(results_df, use_container_width=True)
            
            # Best model
            best_model = results_df['R² Score'].idxmax()
            st.success(f"🏆 **Best Model:** {best_model} (R² = {results_df.loc[best_model, 'R² Score']:.4f})")
            
            st.markdown("""
            <div class="insight-box">
            <h4>📊 Metrics Explained:</h4>
            <ul>
                <li><b>R² Score:</b> 0-1, higher is better (1 = perfect predictions)</li>
                <li><b>MAE:</b> Mean Absolute Error (average prediction error in USD)</li>
                <li><b>RMSE:</b> Root Mean Squared Error (penalizes large errors more)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualizations
            st.markdown("---")
            st.markdown("#### 📈 Actual vs Predicted Values")
            
            # Plot for best model
            fig_reg = go.Figure()
            
            # Actual values
            fig_reg.add_trace(go.Scatter(
                x=list(range(len(y_test))),
                y=y_test.values,
                mode='lines+markers',
                name='Actual',
                line=dict(color='#1E88E5', width=2)
            ))
            
            # Predicted values
            fig_reg.add_trace(go.Scatter(
                x=list(range(len(predictions[best_model]))),
                y=predictions[best_model],
                mode='lines+markers',
                name=f'Predicted ({best_model})',
                line=dict(color='#FF6B6B', width=2, dash='dash')
            ))
            
            fig_reg.update_layout(
                title=f'Actual vs Predicted Import Values - {best_model}',
                xaxis_title='Test Sample Index',
                yaxis_title='Import Value (USD)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_reg, use_container_width=True)
            
            # Feature Importance (for tree-based models)
            if best_model in ['Random Forest', 'XGBoost']:
                st.markdown("---")
                st.markdown("#### 🎯 Feature Importance")
                
                model = models[best_model]
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig_imp = px.bar(
                    feature_importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance Ranking',
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_imp, use_container_width=True)
                
                st.markdown("""
                <div class="insight-box">
                <p><b>Interpretation:</b> Higher importance = feature contributes more to predictions. 
                Lag features (past values) typically have high importance in time series.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 2: CLASSIFICATION MODELS
    # ========================================================================
    with ml_tabs[1]:
        st.markdown("### 🎯 Classification: Categorize Import Patterns")
        
        st.markdown("""
        <div class="insight-box">
        <h4>🎓 What is Classification?</h4>
        <p>Classification models predict <b>categories or classes</b>. Here, we classify transactions into value categories and detect patterns.</p>
        <p><b>Use Cases:</b> Risk assessment, pattern recognition, anomaly detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Create classification target - categorize transaction values
        classification_data = ml_data.copy()
        
        # Define value categories based on quartiles
        quartiles = classification_data['value_dl'].quantile([0.33, 0.67])
        
        def categorize_value(val):
            if val <= quartiles[0.33]:
                return 'Low Value'
            elif val <= quartiles[0.67]:
                return 'Medium Value'
            else:
                return 'High Value'
        
        classification_data['value_category'] = classification_data['value_dl'].apply(categorize_value)
        
        st.markdown("#### 📊 Classification Task: Predict Transaction Value Category")
        st.write(f"- **Low Value:** ≤ ${quartiles[0.33]:,.2f}")
        st.write(f"- **Medium Value:** ${quartiles[0.33]:,.2f} - ${quartiles[0.67]:,.2f}")
        st.write(f"- **High Value:** > ${quartiles[0.67]:,.2f}")
        
        # Show class distribution
        class_dist = classification_data['value_category'].value_counts()
        fig_dist = px.pie(
            values=class_dist.values,
            names=class_dist.index,
            title='Class Distribution',
            color_discrete_sequence=['#1E88E5', '#FF6B6B', '#4CAF50']
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Prepare features
        classification_data['month'] = classification_data['date'].dt.month
        classification_data['quarter'] = classification_data['date'].dt.quarter
        classification_data['year'] = classification_data['date'].dt.year
        classification_data['day_of_week'] = classification_data['date'].dt.dayofweek
        
        # Encode country and commodity if needed
        if selected_country_ml == 'All Countries':
            le_country = LabelEncoder()
            classification_data['country_encoded'] = le_country.fit_transform(classification_data['country_name'])
        else:
            classification_data['country_encoded'] = 0
        
        if selected_commodity_ml == 'All Commodities':
            le_commodity = LabelEncoder()
            classification_data['commodity_encoded'] = le_commodity.fit_transform(classification_data['commodity'])
        else:
            classification_data['commodity_encoded'] = 0
        
        feature_cols_class = ['month', 'quarter', 'year', 'day_of_week', 'country_encoded', 'commodity_encoded', 'value_qt']
        classification_data = classification_data.dropna(subset=feature_cols_class + ['value_category'])
        
        if len(classification_data) < 50:
            st.warning("⚠️ Insufficient data for classification (need at least 50 samples)")
        else:
            X_class = classification_data[feature_cols_class]
            y_class = classification_data['value_category']
            
            # Train-test split
            test_size_class = st.slider("Test Set Size (%)", 10, 40, 20, 5, key="class_test_size") / 100
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                X_class, y_class, test_size=test_size_class, random_state=42, stratify=y_class
            )
            
            st.info(f"📊 Training on {len(X_train_c)} samples, Testing on {len(X_test_c)} samples")
            
            # Train classification models
            st.markdown("---")
            st.markdown("#### 🤖 Model Comparison")
            
            clf_models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            }
            
            clf_results = {}
            clf_predictions = {}
            
            with st.spinner("Training classification models..."):
                for name, model in clf_models.items():
                    model.fit(X_train_c, y_train_c)
                    y_pred_c = model.predict(X_test_c)
                    
                    clf_results[name] = {
                        'Accuracy': accuracy_score(y_test_c, y_pred_c),
                        'Precision': precision_score(y_test_c, y_pred_c, average='weighted'),
                        'Recall': recall_score(y_test_c, y_pred_c, average='weighted'),
                        'F1 Score': f1_score(y_test_c, y_pred_c, average='weighted')
                    }
                    clf_predictions[name] = y_pred_c
            
            # Display results
            clf_results_df = pd.DataFrame(clf_results).T
            clf_results_df = clf_results_df.round(4)
            st.dataframe(clf_results_df, use_container_width=True)
            
            # Best classifier
            best_clf = clf_results_df['Accuracy'].idxmax()
            st.success(f"🏆 **Best Model:** {best_clf} (Accuracy = {clf_results_df.loc[best_clf, 'Accuracy']:.4f})")
            
            st.markdown("""
            <div class="insight-box">
            <h4>📊 Metrics Explained:</h4>
            <ul>
                <li><b>Accuracy:</b> % of correct predictions</li>
                <li><b>Precision:</b> Of predicted positives, how many are actually positive?</li>
                <li><b>Recall:</b> Of actual positives, how many did we find?</li>
                <li><b>F1 Score:</b> Harmonic mean of precision and recall</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Confusion Matrix
            st.markdown("---")
            st.markdown("#### 📊 Confusion Matrix")
            
            cm = confusion_matrix(y_test_c, clf_predictions[best_clf])
            labels = sorted(y_class.unique())
            
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=labels,
                y=labels,
                text_auto=True,
                color_continuous_scale='Blues',
                title=f'Confusion Matrix - {best_clf}'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <p><b>Interpretation:</b> Diagonal cells = correct predictions. Off-diagonal = misclassifications. 
            Perfect model would have all values on diagonal.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 3: CLUSTERING MODELS
    # ========================================================================
    with ml_tabs[2]:
        st.markdown("### 🔍 Clustering: Discover Hidden Patterns")
        
        st.markdown("""
        <div class="insight-box">
        <h4>🎓 What is Clustering?</h4>
        <p>Clustering is <b>unsupervised learning</b> that groups similar data points together without predefined labels.</p>
        <p><b>Use Cases:</b> Market segmentation, pattern discovery, anomaly detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Choose clustering level
        cluster_type = st.radio(
            "Choose what to cluster:",
            ["Commodities", "Countries", "Time Periods"],
            horizontal=True
        )
        
        if cluster_type == "Commodities":
            st.markdown("#### 📦 Commodity Clustering")
            st.write("Group commodities with similar import patterns")
            
            # Aggregate by commodity
            cluster_data = ml_data.groupby('commodity').agg({
                'value_dl': ['sum', 'mean', 'std'],
                'value_qt': 'sum',
                'date': 'count'
            }).reset_index()
            cluster_data.columns = ['commodity', 'total_value', 'avg_value', 'std_value', 'total_qty', 'num_transactions']
            cluster_data['std_value'] = cluster_data['std_value'].fillna(0)
            
            # Add time-based features
            commodity_monthly = ml_data.groupby(['commodity', ml_data['date'].dt.to_period('M')])['value_dl'].sum().reset_index()
            volatility = commodity_monthly.groupby('commodity')['value_dl'].std().reset_index()
            volatility.columns = ['commodity', 'volatility']
            cluster_data = cluster_data.merge(volatility, on='commodity', how='left')
            cluster_data['volatility'] = cluster_data['volatility'].fillna(0)
            
        elif cluster_type == "Countries":
            st.markdown("#### 🌍 Country Clustering")
            st.write("Group countries with similar trade patterns")
            
            # Aggregate by country
            cluster_data = ml_data.groupby('country_name').agg({
                'value_dl': ['sum', 'mean', 'std'],
                'commodity': 'nunique',
                'date': 'count'
            }).reset_index()
            cluster_data.columns = ['country', 'total_value', 'avg_value', 'std_value', 'commodity_diversity', 'num_transactions']
            cluster_data['std_value'] = cluster_data['std_value'].fillna(0)
            
            # Add growth rate
            country_yearly = ml_data.groupby(['country_name', 'year'])['value_dl'].sum().reset_index()
            growth_rates = []
            for country in cluster_data['country'].unique():
                country_years = country_yearly[country_yearly['country_name'] == country].sort_values('year')
                if len(country_years) > 1:
                    growth = (country_years['value_dl'].iloc[-1] - country_years['value_dl'].iloc[0]) / country_years['value_dl'].iloc[0]
                else:
                    growth = 0
                growth_rates.append({'country': country, 'growth_rate': growth})
            
            growth_df = pd.DataFrame(growth_rates)
            cluster_data = cluster_data.merge(growth_df, on='country', how='left')
            cluster_data['growth_rate'] = cluster_data['growth_rate'].fillna(0)
            
        else:  # Time Periods
            st.markdown("#### ⏰ Time Period Clustering")
            st.write("Group similar months/quarters based on trade patterns")
            
            # Aggregate by month
            cluster_data = ml_data.groupby(ml_data['date'].dt.to_period('M')).agg({
                'value_dl': ['sum', 'mean'],
                'commodity': 'nunique',
                'country_name': 'nunique',
                'date': 'count'
            }).reset_index()
            cluster_data.columns = ['period', 'total_value', 'avg_value', 'num_commodities', 'num_countries', 'num_transactions']
            cluster_data['period'] = cluster_data['period'].dt.to_timestamp()
            cluster_data['month'] = cluster_data['period'].dt.month
            cluster_data['year'] = cluster_data['period'].dt.year
        
        if len(cluster_data) < 3:
            st.warning("⚠️ Insufficient data for clustering (need at least 3 items)")
        else:
            # Select features for clustering
            if cluster_type == "Commodities":
                feature_cols_cluster = ['total_value', 'avg_value', 'std_value', 'total_qty', 'num_transactions', 'volatility']
                id_col = 'commodity'
            elif cluster_type == "Countries":
                feature_cols_cluster = ['total_value', 'avg_value', 'std_value', 'commodity_diversity', 'num_transactions', 'growth_rate']
                id_col = 'country'
            else:
                feature_cols_cluster = ['total_value', 'avg_value', 'num_commodities', 'num_countries', 'num_transactions']
                id_col = 'period'
            
            X_cluster = cluster_data[feature_cols_cluster].fillna(0)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            
            # Determine optimal number of clusters with elbow method
            st.markdown("---")
            st.markdown("#### 📊 Elbow Method - Finding Optimal Clusters")
            
            max_clusters = min(10, len(cluster_data) - 1)
            inertias = []
            silhouette_scores = []
            K_range = range(2, max_clusters + 1)
            
            with st.spinner("Computing optimal clusters..."):
                for k in K_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(X_scaled)
                    inertias.append(kmeans.inertia_)
                    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            
            # Plot elbow curve
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=list(K_range),
                y=inertias,
                mode='lines+markers',
                name='Inertia',
                line=dict(color='#1E88E5', width=2)
            ))
            fig_elbow.update_layout(
                title='Elbow Method for Optimal K',
                xaxis_title='Number of Clusters (K)',
                yaxis_title='Inertia (Within-cluster sum of squares)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_elbow, use_container_width=True)
            
            # Plot silhouette scores
            fig_silhouette = go.Figure()
            fig_silhouette.add_trace(go.Scatter(
                x=list(K_range),
                y=silhouette_scores,
                mode='lines+markers',
                name='Silhouette Score',
                line=dict(color='#4CAF50', width=2),
                fill='tozeroy'
            ))
            fig_silhouette.update_layout(
                title='Silhouette Score by Number of Clusters',
                xaxis_title='Number of Clusters (K)',
                yaxis_title='Silhouette Score (higher is better)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_silhouette, use_container_width=True)
            
            # User selects number of clusters
            optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
            st.success(f"💡 Recommended: **{optimal_k} clusters** (highest silhouette score: {max(silhouette_scores):.3f})")
            
            n_clusters = st.slider("Select Number of Clusters:", 2, max_clusters, optimal_k, 1, key="n_clusters")
            
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            cluster_data['Cluster'] = cluster_labels
            cluster_data['Cluster'] = cluster_data['Cluster'].apply(lambda x: f'Cluster {x+1}')
            
            # PCA for visualization
            st.markdown("---")
            st.markdown("#### 🎨 Cluster Visualization (PCA 2D Projection)")
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            cluster_data['PCA1'] = X_pca[:, 0]
            cluster_data['PCA2'] = X_pca[:, 1]
            
            fig_clusters = px.scatter(
                cluster_data,
                x='PCA1',
                y='PCA2',
                color='Cluster',
                hover_data=[id_col],
                title=f'{cluster_type} Clustering Visualization',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_clusters, use_container_width=True)
            
            st.markdown(f"""
            <div class="insight-box">
            <p><b>PCA Explained:</b> Principal Component Analysis reduces multi-dimensional data to 2D for visualization. 
            Variance explained: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Cluster Summary
            st.markdown("---")
            st.markdown("#### 📋 Cluster Summary Statistics")
            
            cluster_summary = cluster_data.groupby('Cluster')[feature_cols_cluster].mean().round(2)
            cluster_summary['Count'] = cluster_data.groupby('Cluster').size()
            st.dataframe(cluster_summary, use_container_width=True)
            
            # Show items in each cluster
            st.markdown("#### 🔍 Cluster Members")
            selected_cluster = st.selectbox("Select cluster to view members:", cluster_data['Cluster'].unique())
            
            cluster_members = cluster_data[cluster_data['Cluster'] == selected_cluster][
                [id_col] + feature_cols_cluster
            ].sort_values('total_value', ascending=False)
            
            st.dataframe(cluster_members, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>💡 Clustering Insights:</h4>
            <p><b>Silhouette Score:</b> Measures how well-defined clusters are (-1 to 1, higher is better)</p>
            <p><b>Business Value:</b> Use clusters to identify similar trading partners, optimize strategies per segment, and detect outliers</p>
            </div>
            """, unsafe_allow_html=True)

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
