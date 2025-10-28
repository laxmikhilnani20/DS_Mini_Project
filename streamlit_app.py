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
        width='stretch',
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
        st.plotly_chart(fig_monthly_eda, width='stretch')
        
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
            st.plotly_chart(fig_yearly_eda, width='stretch')
        
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
                st.plotly_chart(fig_yoy_eda, width='stretch')
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
        st.plotly_chart(fig_seasonal_eda, width='stretch')
        
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
        st.plotly_chart(fig_quarterly_eda, width='stretch')
        
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
        st.plotly_chart(fig_countries, width='stretch')
        
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
        st.plotly_chart(fig_map, width='stretch')
        
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
            st.plotly_chart(fig_regional, width='stretch')
    
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
        st.plotly_chart(fig_commodities, width='stretch')
        
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
        st.plotly_chart(fig_diversity, width='stretch')
        
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
            st.plotly_chart(fig_hist_eda, width='stretch')
        
        with col2:
            fig_box_eda = px.box(
                eda_filtered,
                y='value_dl',
                title='Statistical Summary (Box Plot)',
                labels={'value_dl': 'Transaction Value (USD)'}
            )
            st.plotly_chart(fig_box_eda, width='stretch')
        
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
            st.plotly_chart(fig_ma_eda, width='stretch')
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
            st.plotly_chart(fig_scatter_eda, width='stretch')
            
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
                st.plotly_chart(fig_price_eda, width='stretch')
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
        st.plotly_chart(fig_cumulative_eda, width='stretch')
        
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
            st.plotly_chart(fig_box, width='stretch')
    
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
            st.plotly_chart(fig_corr, width='stretch')
            
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
            st.plotly_chart(fig_compare, width='stretch')
        
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
            st.plotly_chart(fig_conc, width='stretch')
        
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
            st.plotly_chart(fig_comm_conc, width='stretch')

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
            st.dataframe(results_df, width='stretch')
            
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
            
            st.plotly_chart(fig_reg, width='stretch')
            
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
                st.plotly_chart(fig_imp, width='stretch')
                
                st.markdown("""
                <div class="insight-box">
                <p><b>Interpretation:</b> Higher importance = feature contributes more to predictions. 
                Lag features (past values) typically have high importance in time series.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # === NEW: TIME SERIES FORECAST ===
            st.markdown("---")
            st.markdown("#### 🔮 Future Import Forecast (Time Series Prediction)")
            
            st.info("📊 Using the trained model to predict future import values for strategic planning")
            
            # User control for forecast horizon
            forecast_years = st.slider("Forecast Horizon (years)", 5, 10, 5, 1, key="forecast_years_reg")
            forecast_months = forecast_years * 12
            
            # Prepare future data
            last_date = regression_data['date'].max()
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_months, freq='M')
            
            # Create future features
            future_df = pd.DataFrame({'date': future_dates})
            future_df['month'] = future_df['date'].dt.month
            future_df['year'] = future_df['date'].dt.year
            future_df['quarter'] = future_df['date'].dt.quarter
            future_df['days_since_start'] = (future_df['date'] - regression_data['date'].min()).dt.days
            
            # For lag features, use last known values and then rolling predictions
            last_values = regression_data['value_dl'].tail(3).values
            future_predictions = []
            
            for i in range(len(future_df)):
                if i == 0:
                    lag1 = last_values[-1]
                    lag3 = last_values[-3] if len(last_values) >= 3 else last_values[0]
                    rolling_mean = np.mean(last_values)
                elif i == 1:
                    lag1 = future_predictions[0]
                    lag3 = last_values[-2] if len(last_values) >= 2 else last_values[0]
                    rolling_mean = np.mean([last_values[-1], last_values[-2], future_predictions[0]])
                elif i == 2:
                    lag1 = future_predictions[1]
                    lag3 = last_values[-1]
                    rolling_mean = np.mean([last_values[-1], future_predictions[0], future_predictions[1]])
                else:
                    lag1 = future_predictions[i-1]
                    lag3 = future_predictions[i-3]
                    rolling_mean = np.mean(future_predictions[i-3:i])
                
                future_df.loc[i, 'value_lag1'] = lag1
                future_df.loc[i, 'value_lag3'] = lag3
                future_df.loc[i, 'value_rolling_mean_3'] = rolling_mean
                
                # Predict
                X_future = future_df.loc[i:i, feature_cols]
                pred = models[best_model].predict(X_future)[0]
                future_predictions.append(pred)
            
            future_df['predicted_value'] = future_predictions
            
            # Calculate confidence intervals (simple approach using std of errors)
            errors = y_test.values - predictions[best_model]
            error_std = np.std(errors)
            future_df['lower_bound'] = future_df['predicted_value'] - 1.96 * error_std
            future_df['upper_bound'] = future_df['predicted_value'] + 1.96 * error_std
            
            # Create comprehensive timeline visualization
            fig_forecast = go.Figure()
            
            # Historical actual data
            fig_forecast.add_trace(go.Scatter(
                x=regression_data['date'],
                y=regression_data['value_dl'],
                mode='lines',
                name='Historical Data',
                line=dict(color='#1E88E5', width=2)
            ))
            
            # Future predictions
            fig_forecast.add_trace(go.Scatter(
                x=future_df['date'],
                y=future_df['predicted_value'],
                mode='lines',
                name=f'Forecast ({best_model})',
                line=dict(color='#FF6B6B', width=2, dash='dash')
            ))
            
            # Confidence interval
            fig_forecast.add_trace(go.Scatter(
                x=future_df['date'],
                y=future_df['upper_bound'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=future_df['date'],
                y=future_df['lower_bound'],
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                name='95% Confidence Interval',
                fillcolor='rgba(255, 107, 107, 0.2)'
            ))
            
            # Add annotation to mark forecast start (avoid add_vline compatibility issues)
            fig_forecast.add_annotation(
                x=last_date,
                y=max(regression_data['value_dl'].max(), future_df['predicted_value'].max()),
                text="← Historical | Forecast →",
                showarrow=False,
                yshift=10,
                font=dict(size=12, color="gray")
            )
            
            fig_forecast.update_layout(
                title=f'{forecast_years}-Year Import Value Forecast',
                xaxis_title='Date',
                yaxis_title='Import Value (USD)',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_forecast, width='stretch')
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_forecast = future_df['predicted_value'].mean()
                st.metric("Avg Forecast Value", f"${avg_forecast:,.0f}")
            with col2:
                total_forecast = future_df['predicted_value'].sum()
                st.metric("Total Forecast ({} yrs)".format(forecast_years), f"${total_forecast/1e6:.2f}M")
            with col3:
                forecast_trend = ((future_df['predicted_value'].iloc[-1] - future_df['predicted_value'].iloc[0]) / future_df['predicted_value'].iloc[0]) * 100
                st.metric("Forecast Trend", f"{forecast_trend:+.1f}%")
            
            st.markdown("""
            <div class="insight-box">
            <h4>📊 How to Read This Forecast:</h4>
            <ul>
                <li><b>Blue Line:</b> Historical import values (actual data)</li>
                <li><b>Red Dashed Line:</b> Predicted future values using trained ML model</li>
                <li><b>Pink Shaded Area:</b> 95% confidence interval (uncertainty range)</li>
                <li><b>Wider confidence band = higher uncertainty</b> as predictions go further into future</li>
            </ul>
            <p><b>Business Use:</b> Use this forecast for long-term procurement planning, budget allocation, and market trend analysis.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # ========================================================================
    # TAB 2: CLASSIFICATION MODELS - IMPORT TREND PREDICTION
    # ========================================================================
    with ml_tabs[1]:
        st.markdown("### 🎯 Classification: Predict Import Trend Direction")
        
        st.markdown("""
        <div class="insight-box">
        <h4>🎓 What is Import Trend Classification?</h4>
        <p>This model predicts whether a country-commodity trade relationship is <b>Growing, Stable, or Declining</b>.</p>
        <p><b>Use Cases:</b> Identify emerging markets, detect declining trade relationships, strategic procurement planning</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Prepare trend classification data
        st.markdown("#### 📊 Building Trend Dataset")
        
        # Aggregate by country-commodity-year
        trend_data = ml_data.groupby(['country_name', 'commodity', 'year'])['value_dl'].sum().reset_index()
        
        # Calculate year-over-year growth
        trend_data = trend_data.sort_values(['country_name', 'commodity', 'year'])
        trend_data['yoy_growth'] = trend_data.groupby(['country_name', 'commodity'])['value_dl'].pct_change() * 100
        
        # Create trend category based on growth rate
        def categorize_trend(growth):
            if pd.isna(growth):
                return None
            elif growth > 10:
                return 'Growing'
            elif growth < -10:
                return 'Declining'
            else:
                return 'Stable'
        
        trend_data['trend_category'] = trend_data['yoy_growth'].apply(categorize_trend)
        trend_data = trend_data.dropna(subset=['trend_category'])
        
        if len(trend_data) < 50:
            st.warning("⚠️ Insufficient data for trend classification (need at least 50 year-over-year records)")
        else:
            st.success(f"✅ Built trend dataset with {len(trend_data):,} year-over-year observations")
            
            # Show trend distribution
            st.markdown("#### 📈 Trend Distribution")
            trend_dist = trend_data['trend_category'].value_counts()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_trend_dist = px.pie(
                    values=trend_dist.values,
                    names=trend_dist.index,
                    title='Import Trend Categories',
                    color_discrete_sequence=['#4CAF50', '#FF9800', '#F44336'],
                    color_discrete_map={'Growing': '#4CAF50', 'Stable': '#FF9800', 'Declining': '#F44336'}
                )
                st.plotly_chart(fig_trend_dist, width='stretch')
            
            with col2:
                st.metric("Growing Trades", f"{trend_dist.get('Growing', 0):,}")
                st.metric("Stable Trades", f"{trend_dist.get('Stable', 0):,}")
                st.metric("Declining Trades", f"{trend_dist.get('Declining', 0):,}")
            
            # Prepare features for classification
            # Encode country and commodity
            le_country_trend = LabelEncoder()
            le_commodity_trend = LabelEncoder()
            trend_data['country_encoded'] = le_country_trend.fit_transform(trend_data['country_name'])
            trend_data['commodity_encoded'] = le_commodity_trend.fit_transform(trend_data['commodity'])
            
            # Add lag features
            trend_data = trend_data.sort_values(['country_name', 'commodity', 'year'])
            trend_data['value_lag1'] = trend_data.groupby(['country_name', 'commodity'])['value_dl'].shift(1)
            trend_data['growth_lag1'] = trend_data.groupby(['country_name', 'commodity'])['yoy_growth'].shift(1)
            
            trend_data = trend_data.dropna()
            
            feature_cols_trend = ['year', 'country_encoded', 'commodity_encoded', 
                                 'value_dl', 'value_lag1', 'yoy_growth', 'growth_lag1']
            
            X_trend = trend_data[feature_cols_trend].copy()
            
            # Replace infinity and extreme values
            X_trend = X_trend.replace([np.inf, -np.inf], np.nan)
            X_trend = X_trend.fillna(0)
            # Cap extremely large values
            for col in X_trend.columns:
                if X_trend[col].dtype in [np.float64, np.int64]:
                    X_trend[col] = X_trend[col].clip(-1e10, 1e10)
            
            y_trend = trend_data['trend_category']
            
            # Encode target
            le_target_trend = LabelEncoder()
            y_trend_encoded = le_target_trend.fit_transform(y_trend)
            trend_class_names = le_target_trend.classes_
            
            # Train-test split
            st.markdown("---")
            test_size_trend = st.slider("Test Set Size (%)", 10, 40, 20, 5, key="trend_test_size") / 100
            X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
                X_trend, y_trend_encoded, test_size=test_size_trend, random_state=42, stratify=y_trend_encoded
            )
            
            st.info(f"📊 Training on {len(X_train_t)} samples, Testing on {len(X_test_t)} samples")
            
            # Train models
            st.markdown("---")
            st.markdown("#### 🤖 Model Performance Comparison")
            
            trend_models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            }
            
            trend_results = {}
            trend_predictions = {}
            
            with st.spinner("Training trend prediction models..."):
                for name, model in trend_models.items():
                    model.fit(X_train_t, y_train_t)
                    y_pred_t = model.predict(X_test_t)
                    
                    trend_results[name] = {
                        'Accuracy': accuracy_score(y_test_t, y_pred_t),
                        'Precision': precision_score(y_test_t, y_pred_t, average='weighted'),
                        'Recall': recall_score(y_test_t, y_pred_t, average='weighted'),
                        'F1 Score': f1_score(y_test_t, y_pred_t, average='weighted')
                    }
                    trend_predictions[name] = y_pred_t
            
            # Display results
            trend_results_df = pd.DataFrame(trend_results).T
            trend_results_df = trend_results_df.round(4)
            st.dataframe(trend_results_df, width='stretch')
            
            best_trend_model = trend_results_df['Accuracy'].idxmax()
            st.success(f"🏆 **Best Model:** {best_trend_model} (Accuracy = {trend_results_df.loc[best_trend_model, 'Accuracy']:.4f})")
            
            st.markdown("""
            <div class="insight-box">
            <h4>💡 Business Interpretation:</h4>
            <ul>
                <li><b>Growing:</b> YoY growth > 10% - Emerging market opportunities</li>
                <li><b>Stable:</b> YoY growth between -10% and +10% - Mature, predictable trade</li>
                <li><b>Declining:</b> YoY growth < -10% - At-risk trade relationships</li>
            </ul>
            <p><b>Use this to:</b> Prioritize growing markets, investigate declining trends, maintain stable relationships</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confusion Matrix
            st.markdown("---")
            st.markdown("#### 📊 Confusion Matrix")
            
            cm_trend = confusion_matrix(y_test_t, trend_predictions[best_trend_model])
            
            fig_cm_trend = px.imshow(
                cm_trend,
                labels=dict(x="Predicted Trend", y="Actual Trend", color="Count"),
                x=trend_class_names,
                y=trend_class_names,
                text_auto=True,
                color_continuous_scale='RdYlGn',
                title=f'Trend Prediction Confusion Matrix - {best_trend_model}'
            )
            st.plotly_chart(fig_cm_trend, width='stretch')
            
            # Feature Importance (for tree models)
            if best_trend_model in ['Random Forest', 'XGBoost']:
                st.markdown("---")
                st.markdown("#### 🎯 Feature Importance - What Drives Trend Predictions?")
                
                model_trend = trend_models[best_trend_model]
                importances_trend = model_trend.feature_importances_
                feature_importance_trend_df = pd.DataFrame({
                    'Feature': feature_cols_trend,
                    'Importance': importances_trend
                }).sort_values('Importance', ascending=False)
                
                fig_imp_trend = px.bar(
                    feature_importance_trend_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance for Trend Prediction',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_imp_trend, width='stretch')
    
    # ========================================================================
    # TAB 3: CLUSTERING MODELS - CO-OCCURRENCE ANALYSIS
    # ========================================================================
    with ml_tabs[2]:
        st.markdown("### 🔍 Clustering: Commodity Co-Occurrence Patterns")
        
        st.markdown("""
        <div class="insight-box">
        <h4>🎓 What is Co-Occurrence Clustering?</h4>
        <p>Discover which <b>commodities are imported together</b> by the same countries. This reveals trade dependencies and bundling opportunities.</p>
        <p><b>Use Cases:</b> Bundle procurement, identify trade dependencies, market basket analysis for imports</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("#### 📊 Building Co-Occurrence Matrix")
        
        # Create country-commodity matrix
        country_commodity_matrix = ml_data.groupby(['country_name', 'commodity'])['value_dl'].sum().unstack(fill_value=0)
        country_commodity_binary = (country_commodity_matrix > 0).astype(int)
        
        # Calculate co-occurrence
        from sklearn.metrics.pairwise import cosine_similarity
        commodity_country_matrix = country_commodity_binary.T
        co_occurrence_matrix = cosine_similarity(commodity_country_matrix)
        co_occurrence_df = pd.DataFrame(
            co_occurrence_matrix,
            index=commodity_country_matrix.index,
            columns=commodity_country_matrix.index
        )
        
        st.success(f"✅ Built co-occurrence matrix for {len(co_occurrence_df)} commodities")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Commodities", len(co_occurrence_df))
        with col2:
            st.metric("Total Countries", len(country_commodity_binary))
        with col3:
            avg_co_occurrence = co_occurrence_df.values[np.triu_indices_from(co_occurrence_df.values, k=1)].mean()
            st.metric("Avg Co-Occurrence", f"{avg_co_occurrence:.3f}")
        
        # Top co-occurring pairs
        st.markdown("---")
        st.markdown("#### 🔗 Top Co-Occurring Commodity Pairs")
        
        co_occur_pairs = []
        for i in range(len(co_occurrence_df)):
            for j in range(i+1, len(co_occurrence_df)):
                if co_occurrence_df.iloc[i, j] > 0.5:
                    co_occur_pairs.append({
                        'Commodity 1': co_occurrence_df.index[i],
                        'Commodity 2': co_occurrence_df.columns[j],
                        'Co-Occurrence Score': co_occurrence_df.iloc[i, j]
                    })
        
        if co_occur_pairs:
            co_occur_df = pd.DataFrame(co_occur_pairs).sort_values('Co-Occurrence Score', ascending=False).head(20)
            st.dataframe(co_occur_df, width='stretch')
            
            st.markdown("""
            <div class="insight-box">
            <p><b>Interpretation:</b> Score of 1.0 = always imported together. Score of 0.5+ = frequently imported together.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No strong co-occurrence patterns found (score > 0.5)")
        
        # Cluster commodities
        st.markdown("---")
        st.markdown("#### 🎨 Clustering Commodities by Import Patterns")
        # Prepare data for clustering
        X_cluster = co_occurrence_df.values
        
        # Replace any inf/nan values
        X_cluster = np.nan_to_num(X_cluster, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        max_clusters = min(10, len(co_occurrence_df) // 5)
        
        if max_clusters >= 2:
            inertias = []
            silhouette_scores = []
            K_range = range(2, max_clusters + 1)
            
            with st.spinner("Finding optimal commodity clusters..."):
                for k in K_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(X_scaled)
                    inertias.append(kmeans.inertia_)
                    if k > 1:
                        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=list(K_range),
                y=inertias,
                mode='lines+markers',
                line=dict(color='#1E88E5', width=2)
            ))
            fig_elbow.update_layout(
                title='Elbow Method for Commodity Clusters',
                xaxis_title='Number of Clusters (K)',
                yaxis_title='Inertia',
                height=400
            )
            st.plotly_chart(fig_elbow, width='stretch')
            
            if silhouette_scores:
                optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
                st.success(f"💡 Recommended: **{optimal_k} clusters** (best silhouette score)")
            else:
                optimal_k = 3
            
            n_clusters = st.slider("Select Number of Commodity Clusters:", 2, max_clusters, optimal_k, 1, key="n_commodity_clusters")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            cluster_assignments = pd.DataFrame({
                'Commodity': co_occurrence_df.index,
                'Cluster': cluster_labels
            })
            cluster_assignments['Cluster'] = cluster_assignments['Cluster'].apply(lambda x: f'Group {x+1}')
            
            # PCA visualization
            st.markdown("---")
            st.markdown("#### 🗺️ Commodity Cluster Map (PCA 2D)")
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            cluster_assignments['PCA1'] = X_pca[:, 0]
            cluster_assignments['PCA2'] = X_pca[:, 1]
            
            fig_clusters = px.scatter(
                cluster_assignments,
                x='PCA1',
                y='PCA2',
                color='Cluster',
                hover_data=['Commodity'],
                title='Commodity Clusters Based on Co-Import Patterns',
                color_discrete_sequence=px.colors.qualitative.Set3,
                height=600
            )
            st.plotly_chart(fig_clusters, width='stretch')
            
            st.markdown(f"""
            <div class="insight-box">
            <p><b>PCA Explained Variance:</b> PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show cluster members
            st.markdown("---")
            st.markdown("#### 📋 Commodity Groups (Clusters)")
            
            selected_cluster_view = st.selectbox(
                "Select cluster to view members:",
                cluster_assignments['Cluster'].unique()
            )
            
            cluster_members = cluster_assignments[cluster_assignments['Cluster'] == selected_cluster_view]['Commodity'].tolist()
            st.write(f"**{selected_cluster_view}** contains {len(cluster_members)} commodities:")
            st.write(", ".join(cluster_members))
            
            # Cluster characteristics
            st.markdown("---")
            st.markdown("#### 📊 Cluster Characteristics")
            
            cluster_chars = []
            for cluster in cluster_assignments['Cluster'].unique():
                cluster_commodities = cluster_assignments[cluster_assignments['Cluster'] == cluster]['Commodity'].tolist()
                cluster_imports = ml_data[ml_data['commodity'].isin(cluster_commodities)]
                
                cluster_chars.append({
                    'Cluster': cluster,
                    'Num Commodities': len(cluster_commodities),
                    'Num Countries': cluster_imports['country_name'].nunique(),
                    'Total Value': cluster_imports['value_dl'].sum(),
                    'Avg Import Value': cluster_imports['value_dl'].mean()
                })
            
            cluster_chars_df = pd.DataFrame(cluster_chars).sort_values('Total Value', ascending=False)
            st.dataframe(cluster_chars_df, width='stretch')
            
            st.markdown("""
            <div class="insight-box">
            <h4>💡 Business Value of Co-Occurrence Clustering:</h4>
            <ul>
                <li><b>Bundle Procurement:</b> Commodities in the same cluster are often imported together - negotiate bundle deals</li>
                <li><b>Trade Dependencies:</b> Understand which products have correlated demand across countries</li>
                <li><b>Market Diversification:</b> Identify commodity groups with different import patterns for risk management</li>
                <li><b>Strategic Planning:</b> Countries importing one commodity in a cluster likely need others from same cluster</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Insufficient commodities for meaningful clustering")


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
