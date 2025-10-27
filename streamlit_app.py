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
        st.metric("Total Value", f"${total_value/1e9:.2f}B")
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
        st.success(f"**Top Importing Country:** {top_country} (${top_country_value/1e9:.2f}B)")
        
        top_commodity = df.groupby('commodity')['value_dl'].sum().idxmax()
        st.success(f"**Top Commodity:** {top_commodity}")
    
    with insight_col2:
        most_diverse = df.groupby('country_name')['commodity'].nunique().idxmax()
        diversity_count = df.groupby('country_name')['commodity'].nunique().max()
        st.info(f"**Most Diverse Imports:** {most_diverse} ({diversity_count:,} commodities)")
        
        avg_transaction = df['value_dl'].mean()
        st.info(f"**Average Transaction Value:** ${avg_transaction:,.2f}")


# ======================================================================
# DASHBOARD 2: EDA EXPLORER
# ======================================================================
elif page == "🔍 EDA Explorer":
    st.markdown('<p class="main-header">🔍 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown("### Comprehensive Visual Analysis of Import Patterns")
    
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
        st.markdown("### 📈 Temporal Analysis")
        
        # Monthly trends
        st.markdown("#### Overall Monthly Import Trends")
        monthly_agg = df.resample('M', on='date')['value_dl'].sum().reset_index()
        monthly_agg.columns = ['Date', 'Total Import Value (USD)']
        
        fig_monthly = px.line(
            monthly_agg,
            x='Date',
            y='Total Import Value (USD)',
            title='Total Monthly Import Value Over Time',
            labels={'Total Import Value (USD)': 'Value (USD)'}
        )
        fig_monthly.add_vline(x=pd.Timestamp('2020-03-01'), line_dash="dash", line_color="red",
                             annotation_text="COVID-19", annotation_position="top")
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        st.info("📊 **Insight**: Notice the sharp decline in 2020-2021 (COVID-19 impact) followed by strong recovery in 2022.")
        
        # Year-over-Year comparison
        st.markdown("#### Year-over-Year Comparison")
        yearly_agg = df.groupby('year')['value_dl'].sum().reset_index()
        yearly_agg.columns = ['Year', 'Total Value']
        yearly_agg['YoY %'] = yearly_agg['Total Value'].pct_change() * 100
        
        col1, col2 = st.columns(2)
        with col1:
            fig_yearly = px.bar(
                yearly_agg,
                x='Year',
                y='Total Value',
                title='Yearly Total Import Value',
                text_auto='.2s'
            )
            st.plotly_chart(fig_yearly, use_container_width=True)
        
        with col2:
            fig_yoy = px.line(
                yearly_agg.dropna(),
                x='Year',
                y='YoY %',
                title='Year-over-Year Growth Rate (%)',
                markers=True
            )
            fig_yoy.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_yoy, use_container_width=True)
        
        # Seasonal patterns
        st.markdown("#### Seasonal Patterns")
        df_season = df.copy()
        df_season['Month'] = df_season['date'].dt.month_name()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        seasonal = df_season.groupby('Month')['value_dl'].sum().reindex(month_order).reset_index()
        
        fig_seasonal = px.bar(
            seasonal,
            x='Month',
            y='value_dl',
            title='Average Import Value by Month',
            labels={'value_dl': 'Total Value (USD)'}
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # ===== TAB 2: GEOGRAPHIC ANALYSIS =====
    with eda_tabs[1]:
        st.markdown("### 🌍 Geographic Distribution")
        
        # Top countries
        st.markdown("#### Top 15 Countries by Import Value")
        country_totals = df.groupby('country_name')['value_dl'].sum().sort_values(ascending=False).head(15).reset_index()
        country_totals.columns = ['Country', 'Total Value (USD)']
        
        fig_countries = px.bar(
            country_totals,
            y='Country',
            x='Total Value (USD)',
            orientation='h',
            title='Top 15 Importing Countries',
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
        st.markdown("#### Top 20 Commodities by Value")
        commodity_totals = df.groupby('commodity')['value_dl'].sum().sort_values(ascending=False).head(20).reset_index()
        commodity_totals.columns = ['Commodity', 'Total Value (USD)']
        
        fig_commodities = px.bar(
            commodity_totals,
            y='Commodity',
            x='Total Value (USD)',
            orientation='h',
            title='Top 20 Commodities',
            text_auto='.2s',
            height=600
        )
        st.plotly_chart(fig_commodities, use_container_width=True)
        
        # Commodity diversity
        st.markdown("#### Commodity Diversity by Country")
        diversity = df.groupby('country_name')['commodity'].nunique().sort_values(ascending=False).head(15).reset_index()
        diversity.columns = ['Country', 'Unique Commodities']
        
        fig_diversity = px.bar(
            diversity,
            x='Country',
            y='Unique Commodities',
            title='Top 15 Countries by Commodity Diversity',
            text_auto=True
        )
        st.plotly_chart(fig_diversity, use_container_width=True)
        
        st.info("📊 **Insight**: Higher commodity diversity indicates more complex import portfolios and varied industrial needs.")
    
    # ===== TAB 4: DISTRIBUTIONS =====
    with eda_tabs[3]:
        st.markdown("### 📊 Statistical Distributions")
        
        # Value distributions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Import Value Distribution")
            fig_hist = px.histogram(
                df[df['value_dl'] < df['value_dl'].quantile(0.95)],  # Remove extreme outliers for better viz
                x='value_dl',
                nbins=50,
                title='Distribution of Import Values (95th percentile)',
                labels={'value_dl': 'Value (USD)'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.markdown("#### Log-Scale Distribution")
            fig_log = px.histogram(
                df[df['value_dl'] > 0],
                x='value_dl',
                nbins=50,
                title='Log-Scale Import Value Distribution',
                labels={'value_dl': 'Value (USD)'},
                log_y=True
            )
            st.plotly_chart(fig_log, use_container_width=True)
        
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
        st.write("Analyzing correlation between import patterns of top 15 commodities")
        
        # Get top commodities
        top_commodities = df.groupby('commodity')['value_dl'].sum().nlargest(15).index.tolist()
        
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
# DASHBOARD 3: INTERACTIVE MODELS
# ======================================================================
elif page == "🎯 Interactive Models":
    st.markdown('<p class="main-header">🎯 Interactive Deep-Dive Analysis</p>', unsafe_allow_html=True)
    st.markdown("### Select Country + Commodity for Detailed Insights")
    
    # === SELECTION PANEL ===
    st.markdown("## 🎛️ Selection Panel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Select Country")
        all_countries = sorted(df['country_name'].unique())
        selected_country = st.selectbox(
            "Choose a country:",
            options=all_countries,
            index=0,
            key="model_country"
        )
    
    with col2:
        st.markdown("#### Select Commodity")
        # Filter commodities based on selected country
        available_commodities = sorted(
            df[df['country_name'] == selected_country]['commodity'].unique()
        )
        
        if available_commodities:
            selected_commodity = st.selectbox(
                "Choose a commodity:",
                options=available_commodities,
                index=0,
                key="model_commodity"
            )
        else:
            st.error("No commodities available for selected country")
            st.stop()
    
    # Filter data for selected combination
    filtered_data = df[
        (df['country_name'] == selected_country) &
        (df['commodity'] == selected_commodity)
    ].copy()
    
    if filtered_data.empty:
        st.warning("⚠️ No data available for this country-commodity combination.")
        st.stop()
    
    # === KEY METRICS FOR SELECTION ===
    st.markdown("---")
    st.markdown(f"## 📊 Analysis: **{selected_commodity}** from **{selected_country}**")
    
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    
    with metric_col1:
        total_transactions = len(filtered_data)
        st.metric("Transactions", f"{total_transactions:,}")
    
    with metric_col2:
        total_value = filtered_data['value_dl'].sum()
        st.metric("Total Value", f"${total_value/1e6:.2f}M")
    
    with metric_col3:
        avg_value = filtered_data['value_dl'].mean()
        st.metric("Avg Value", f"${avg_value:,.0f}")
    
    with metric_col4:
        total_quantity = filtered_data['value_qt'].sum()
        st.metric("Total Quantity", f"{total_quantity:,.0f}")
    
    with metric_col5:
        date_range = (filtered_data['date'].max() - filtered_data['date'].min()).days
        st.metric("Days Span", f"{date_range}")
    
    # === TABBED ANALYSIS ===
    analysis_tabs = st.tabs([
        "📈 Trend Analysis",
        "🎯 Contribution & Share",
        "⏰ Temporal Patterns",
        "💰 Value Insights",
        "📊 Statistical Summary",
        "🔮 Forecasting"
    ])
    
    # ===== TAB 1: TREND ANALYSIS =====
    with analysis_tabs[0]:
        st.markdown("### 📈 Historical Trend Analysis")
        
        # Monthly trend
        monthly_data = filtered_data.resample('M', on='date')['value_dl'].sum().reset_index()
        monthly_data.columns = ['Date', 'Value']
        
        if len(monthly_data) > 0:
            fig_trend = px.line(
                monthly_data,
                x='Date',
                y='Value',
                title=f'Monthly Import Value Trend: {selected_commodity} from {selected_country}',
                labels={'Value': 'Import Value (USD)'}
            )
            fig_trend.update_traces(line_color='#1E88E5', line_width=2.5)
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Trend statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Trend Statistics**")
                peak_month = monthly_data.loc[monthly_data['Value'].idxmax(), 'Date']
                peak_value = monthly_data['Value'].max()
                st.write(f"- **Peak Month:** {peak_month.strftime('%B %Y')}")
                st.write(f"- **Peak Value:** ${peak_value:,.2f}")
                
                avg_monthly = monthly_data['Value'].mean()
                st.write(f"- **Average Monthly:** ${avg_monthly:,.2f}")
            
            with col2:
                st.markdown("**Growth Analysis**")
                if len(monthly_data) > 1:
                    first_val = monthly_data.iloc[0]['Value']
                    last_val = monthly_data.iloc[-1]['Value']
                    
                    if first_val > 0:
                        growth_pct = ((last_val - first_val) / first_val) * 100
                        st.write(f"- **Overall Growth:** {growth_pct:+.1f}%")
                    
                    volatility = monthly_data['Value'].std() / monthly_data['Value'].mean() * 100
                    st.write(f"- **Volatility (CV):** {volatility:.1f}%")
        
        # Yearly comparison
        st.markdown("#### Year-by-Year Comparison")
        yearly_data = filtered_data.groupby('year')['value_dl'].sum().reset_index()
        yearly_data.columns = ['Year', 'Value']
        
        if len(yearly_data) > 0:
            fig_yearly = px.bar(
                yearly_data,
                x='Year',
                y='Value',
                title='Yearly Import Values',
                text_auto='.2s',
                color='Value',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_yearly, use_container_width=True)
    
    # ===== TAB 2: CONTRIBUTION & SHARE =====
    with analysis_tabs[1]:
        st.markdown("### 🎯 Contribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Contribution to Country's Total Imports")
            
            # Calculate country total
            country_total = df[df['country_name'] == selected_country]['value_dl'].sum()
            commodity_value = filtered_data['value_dl'].sum()
            contribution_pct = (commodity_value / country_total) * 100
            
            st.metric(
                "Percentage of Total",
                f"{contribution_pct:.2f}%",
                help=f"This commodity represents {contribution_pct:.2f}% of all imports from {selected_country}"
            )
            
            # Create gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=contribution_pct,
                title={'text': f"% of {selected_country}'s Total Imports"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1E88E5"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"},
                        {'range': [50, 75], 'color': "darkgray"},
                        {'range': [75, 100], 'color': "dimgray"}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.markdown("#### Ranking Within Country")
            
            # Get commodity rankings for this country
            country_commodities = df[df['country_name'] == selected_country].groupby('commodity')['value_dl'].sum().sort_values(ascending=False).reset_index()
            country_commodities['rank'] = range(1, len(country_commodities) + 1)
            
            commodity_rank = country_commodities[country_commodities['commodity'] == selected_commodity]['rank'].iloc[0] if len(country_commodities[country_commodities['commodity'] == selected_commodity]) > 0 else None
            total_commodities = len(country_commodities)
            
            if commodity_rank:
                st.metric(
                    "Commodity Rank",
                    f"#{commodity_rank} of {total_commodities}",
                    help=f"This commodity ranks #{commodity_rank} among all commodities imported by {selected_country}"
                )
                
                # Show top 10 for context
                st.markdown("**Top 10 Commodities for This Country:**")
                top_10 = country_commodities.head(10)
                
                # Highlight selected commodity
                def highlight_selected(row):
                    if row['commodity'] == selected_commodity:
                        return ['background-color: #e3f2fd'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    top_10[['rank', 'commodity', 'value_dl']].style.apply(highlight_selected, axis=1),
                    use_container_width=True,
                    height=400
                )
        
        # Share over time
        st.markdown("#### Share Evolution Over Time")
        
        # Calculate monthly share
        monthly_country = df[df['country_name'] == selected_country].resample('M', on='date')['value_dl'].sum()
        monthly_commodity = filtered_data.resample('M', on='date')['value_dl'].sum()
        
        share_over_time = pd.DataFrame({
            'Date': monthly_commodity.index,
            'Share %': (monthly_commodity.values / monthly_country.values * 100)
        })
        
        fig_share = px.line(
            share_over_time,
            x='Date',
            y='Share %',
            title=f'Monthly Share of {selected_commodity} in {selected_country} Imports',
            labels={'Share %': 'Share of Total Imports (%)'}
        )
        fig_share.update_traces(line_color='#43A047', line_width=2)
        st.plotly_chart(fig_share, use_container_width=True)
    
    # ===== TAB 3: TEMPORAL PATTERNS =====
    with analysis_tabs[2]:
        st.markdown("### ⏰ Temporal Patterns & Seasonality")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Monthly Seasonality")
            
            monthly_pattern = filtered_data.groupby('month_name')['value_dl'].mean().reindex(
                ['January', 'February', 'March', 'April', 'May', 'June', 
                 'July', 'August', 'September', 'October', 'November', 'December']
            ).reset_index()
            monthly_pattern.columns = ['Month', 'Avg Value']
            
            fig_seasonal = px.bar(
                monthly_pattern,
                x='Month',
                y='Avg Value',
                title='Average Import Value by Month',
                labels={'Avg Value': 'Average Value (USD)'},
                color='Avg Value',
                color_continuous_scale='Teal'
            )
            st.plotly_chart(fig_seasonal, use_container_width=True)
        
        with col2:
            st.markdown("#### Quarterly Patterns")
            
            quarterly_pattern = filtered_data.groupby('quarter')['value_dl'].mean().reset_index()
            quarterly_pattern.columns = ['Quarter', 'Avg Value']
            quarterly_pattern['Quarter'] = 'Q' + quarterly_pattern['Quarter'].astype(str)
            
            fig_quarterly = px.bar(
                quarterly_pattern,
                x='Quarter',
                y='Avg Value',
                title='Average Import Value by Quarter',
                labels={'Avg Value': 'Average Value (USD)'},
                color='Avg Value',
                color_continuous_scale='Purp'
            )
            st.plotly_chart(fig_quarterly, use_container_width=True)
        
        # Day of week analysis (if enough data)
        st.markdown("#### Transaction Frequency by Day of Week")
        filtered_data['day_of_week'] = filtered_data['date'].dt.day_name()
        dow_pattern = filtered_data.groupby('day_of_week').size().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ).reset_index()
        dow_pattern.columns = ['Day', 'Count']
        
        fig_dow = px.bar(
            dow_pattern,
            x='Day',
            y='Count',
            title='Transaction Count by Day of Week',
            labels={'Count': 'Number of Transactions'}
        )
        st.plotly_chart(fig_dow, use_container_width=True)
    
    # ===== TAB 4: VALUE INSIGHTS =====
    with analysis_tabs[3]:
        st.markdown("### 💰 Value & Pricing Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Value Distribution")
            
            fig_hist_value = px.histogram(
                filtered_data,
                x='value_dl',
                nbins=30,
                title='Distribution of Transaction Values',
                labels={'value_dl': 'Value (USD)'}
            )
            st.plotly_chart(fig_hist_value, use_container_width=True)
            
            # Value statistics
            st.markdown("**Value Statistics:**")
            st.write(f"- **Min:** ${filtered_data['value_dl'].min():,.2f}")
            st.write(f"- **Max:** ${filtered_data['value_dl'].max():,.2f}")
            st.write(f"- **Median:** ${filtered_data['value_dl'].median():,.2f}")
            st.write(f"- **Std Dev:** ${filtered_data['value_dl'].std():,.2f}")
        
        with col2:
            st.markdown("#### Quantity vs Value Relationship")
            
            # Filter valid data
            valid_data = filtered_data[(filtered_data['value_qt'] > 0) & (filtered_data['value_dl'] > 0)].copy()
            
            if len(valid_data) > 0:
                valid_data['unit_price'] = valid_data['value_dl'] / valid_data['value_qt']
                
                fig_scatter = px.scatter(
                    valid_data.sample(min(1000, len(valid_data))),  # Sample for performance
                    x='value_qt',
                    y='value_dl',
                    title='Quantity vs Value Relationship',
                    labels={'value_qt': 'Quantity', 'value_dl': 'Value (USD)'},
                    trendline='ols'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                st.markdown("**Unit Price Analysis:**")
                st.write(f"- **Avg Unit Price:** ${valid_data['unit_price'].mean():,.2f}")
                st.write(f"- **Median Unit Price:** ${valid_data['unit_price'].median():,.2f}")
            else:
                st.info("Insufficient data for quantity-value analysis")
        
        # Price trends over time
        if len(valid_data) > 0:
            st.markdown("#### Unit Price Trend")
            
            monthly_price = valid_data.resample('M', on='date').apply(
                lambda x: (x['value_dl'].sum() / x['value_qt'].sum()) if x['value_qt'].sum() > 0 else np.nan
            ).reset_index()
            monthly_price.columns = ['Date', 'Avg Unit Price']
            monthly_price = monthly_price.dropna()
            
            fig_price_trend = px.line(
                monthly_price,
                x='Date',
                y='Avg Unit Price',
                title='Average Unit Price Over Time',
                labels={'Avg Unit Price': 'Price per Unit (USD)'}
            )
            st.plotly_chart(fig_price_trend, use_container_width=True)
    
    # ===== TAB 5: STATISTICAL SUMMARY =====
    with analysis_tabs[4]:
        st.markdown("### 📊 Comprehensive Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Descriptive Statistics")
            
            stats_df = filtered_data[['value_dl', 'value_qt']].describe().T
            stats_df['column'] = stats_df.index
            stats_df = stats_df.reset_index(drop=True)
            st.dataframe(stats_df, use_container_width=True)
            
            st.markdown("#### Data Quality Metrics")
            st.write(f"- **Missing Values (value_dl):** {filtered_data['value_dl'].isna().sum()}")
            st.write(f"- **Missing Values (value_qt):** {filtered_data['value_qt'].isna().sum()}")
            st.write(f"- **Zero Values (value_dl):** {(filtered_data['value_dl'] == 0).sum()}")
            st.write(f"- **Zero Values (value_qt):** {(filtered_data['value_qt'] == 0).sum()}")
        
        with col2:
            st.markdown("#### Time Range Analysis")
            st.write(f"- **First Transaction:** {filtered_data['date'].min().strftime('%Y-%m-%d')}")
            st.write(f"- **Last Transaction:** {filtered_data['date'].max().strftime('%Y-%m-%d')}")
            st.write(f"- **Total Days:** {(filtered_data['date'].max() - filtered_data['date'].min()).days}")
            st.write(f"- **Number of Years:** {len(filtered_data['year'].unique())}")
            st.write(f"- **Number of Months:** {len(filtered_data.groupby(filtered_data['date'].dt.to_period('M')))}")
            
            st.markdown("#### Transaction Patterns")
            st.write(f"- **Transactions per Year:** {len(filtered_data) / len(filtered_data['year'].unique()):.1f}")
            st.write(f"- **Transactions per Month:** {len(filtered_data) / len(filtered_data.groupby(filtered_data['date'].dt.to_period('M'))):.1f}")
        
        # Detailed data view
        st.markdown("#### Raw Data Sample")
        st.dataframe(
            filtered_data[['date', 'value_dl', 'value_qt', 'unit', 'year', 'month_name']].head(20),
            use_container_width=True
        )
    
    # ===== TAB 6: FORECASTING =====
    with analysis_tabs[5]:
        st.markdown("### 🔮 Forecast Future Imports")
        
        if not PROPHET_AVAILABLE:
            st.error("❌ Prophet library is not installed. Forecasting features are unavailable.")
            st.info("To enable forecasting, install Prophet: `pip install prophet`")
        else:
            st.info("📊 Using Prophet time series forecasting model to predict future import values")
            
            forecast_months = st.slider("Forecast Horizon (months)", min_value=3, max_value=24, value=12, step=3)
            
            # Prepare data for Prophet
            prophet_data = filtered_data.resample('M', on='date')['value_dl'].sum().reset_index()
            prophet_data.columns = ['ds', 'y']
            
            if len(prophet_data) >= 3:
                try:
                    with st.spinner("Training forecast model..."):
                        # Train Prophet model
                        model = Prophet(
                            yearly_seasonality=True,
                            weekly_seasonality=False,
                            daily_seasonality=False
                        )
                        model.fit(prophet_data)
                        
                        # Make future predictions
                        future = model.make_future_dataframe(periods=forecast_months, freq='M')
                        forecast = model.predict(future)
                    
                    # Plot forecast
                    fig_forecast = go.Figure()
                    
                    # Historical data
                    fig_forecast.add_trace(go.Scatter(
                        x=prophet_data['ds'],
                        y=prophet_data['y'],
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color='#1E88E5', width=2)
                    ))
                    
                    # Forecast
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='#E53935', width=2, dash='dash')
                    ))
                    
                    # Confidence interval
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_upper'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(229, 57, 53, 0.2)',
                        showlegend=False,
                        name='Upper Bound'
                    ))
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_lower'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(229, 57, 53, 0.2)',
                        name='Confidence Interval'
                    ))
                    
                    fig_forecast.update_layout(
                        title=f'{forecast_months}-Month Forecast: {selected_commodity} from {selected_country}',
                        xaxis_title='Date',
                        yaxis_title='Import Value (USD)',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Forecast statistics
                    st.markdown("#### Forecast Statistics")
                    
                    future_forecast = forecast[forecast['ds'] > prophet_data['ds'].max()]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_forecast = future_forecast['yhat'].mean()
                        st.metric("Avg Forecast Value", f"${avg_forecast:,.0f}")
                    
                    with col2:
                        total_forecast = future_forecast['yhat'].sum()
                        st.metric(f"Total {forecast_months}-Month", f"${total_forecast/1e6:.2f}M")
                    
                    with col3:
                        historical_avg = prophet_data['y'].mean()
                        growth = ((avg_forecast - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0
                        st.metric("Expected Growth", f"{growth:+.1f}%")
                    
                    # Show forecast table
                    with st.expander("📋 View Detailed Forecast"):
                        forecast_display = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                        forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                        forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m')
                        st.dataframe(forecast_display, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Forecasting failed: {str(e)}")
                    st.info("This could be due to insufficient data or irregular patterns. Try a different commodity or country combination.")
            else:
                st.warning("⚠️ Insufficient data for forecasting. Need at least 3 months of data.")
    
    # === DOWNLOAD SECTION ===
    st.markdown("---")
    st.markdown("## 💾 Download Analysis Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Selected Data (CSV)",
            data=csv_data,
            file_name=f"{selected_country}_{selected_commodity}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create summary report
        summary_report = pd.DataFrame({
            'Metric': [
                'Country', 'Commodity', 'Total Transactions', 'Total Value (USD)',
                'Average Value (USD)', 'Date Range', 'Peak Month', 'Peak Value'
            ],
            'Value': [
                selected_country, selected_commodity, len(filtered_data),
                f"${filtered_data['value_dl'].sum():,.2f}",
                f"${filtered_data['value_dl'].mean():,.2f}",
                f"{filtered_data['date'].min()} to {filtered_data['date'].max()}",
                monthly_data.loc[monthly_data['Value'].idxmax(), 'Date'].strftime('%B %Y') if len(monthly_data) > 0 else 'N/A',
                f"${monthly_data['Value'].max():,.2f}" if len(monthly_data) > 0 else 'N/A'
            ]
        })
        
        csv_summary = summary_report.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Summary Report (CSV)",
            data=csv_summary,
            file_name=f"summary_{selected_country}_{selected_commodity}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


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
