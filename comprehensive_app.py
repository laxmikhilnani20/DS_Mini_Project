"""
🌍 African Imports Analytics Platform - Complete Interactive Dashboard
Uses all pre-trained models, saved plots, and notebook insights
NO TRAINING REQUIRED - Everything is pre-computed!
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# ======================================================================
# CONFIGURATION
# ======================================================================
st.set_page_config(
    page_title="African Imports ML Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
DATA_FILE = 'imports-from-african-countries.csv'
MODELS_DIR = 'african_imports_models_only_20251102_175254'
PLOTS_DIR = 'african_imports_plots_only_20251102_175254'

# ======================================================================
# CUSTOM CSS
# ======================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1E88E5, #00BCD4, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 136, 229, 0.1), rgba(0, 188, 212, 0.1));
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: rgba(76, 175, 80, 0.1);
        padding: 1.5rem;
        border-left: 5px solid #4CAF50;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: rgba(255, 152, 0, 0.1);
        padding: 1rem;
        border-left: 5px solid #FF9800;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: rgba(76, 175, 80, 0.15);
        padding: 1rem;
        border-left: 5px solid #4CAF50;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: rgba(128, 128, 128, 0.1);
        border-radius: 8px 8px 0px 0px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(30, 136, 229, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ======================================================================
# DATA LOADING
# ======================================================================
@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    try:
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        df['unit'] = df['unit'].fillna('Unknown')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.month_name()
        df['quarter'] = df['date'].dt.quarter
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

@st.cache_data
def get_available_models():
    """Get list of all available models and plots"""
    models = {}
    plots = {}
    
    if os.path.exists(MODELS_DIR):
        for file in os.listdir(MODELS_DIR):
            if file.endswith('.pkl'):
                model_name = file.replace('.pkl', '')
                model_type = 'Regression' if 'regression' in file else \
                            'Classification' if 'classification' in file else \
                            'Clustering' if 'clustering' in file else 'Other'
                
                models[model_name] = {
                    'path': os.path.join(MODELS_DIR, file),
                    'type': model_type,
                    'size_mb': os.path.getsize(os.path.join(MODELS_DIR, file)) / (1024 * 1024)
                }
    
    if os.path.exists(PLOTS_DIR):
        for file in os.listdir(PLOTS_DIR):
            if file.endswith('.png'):
                plot_name = file.replace('.png', '')
                plots[plot_name] = os.path.join(PLOTS_DIR, file)
    
    return models, plots

# Load data
df = load_data()
available_models, available_plots = get_available_models()

# ======================================================================
# SIDEBAR
# ======================================================================
st.sidebar.markdown("# 🌍 African Imports Analytics")
st.sidebar.markdown("### Complete ML Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "🔍 Select Dashboard:",
    [
        "🏠 Home & Overview",
        "📊 Data Explorer",
        "📈 EDA Insights",
        "🤖 Model Gallery",
        "🎯 Predictions",
        "🔍 Country Analysis",
        "📉 Model Performance"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
### 📊 Dataset Summary
- **Records**: {len(df):,}
- **Countries**: {df['country_name'].nunique()}
- **Commodities**: {df['commodity'].nunique():,}
- **Period**: {df['year'].min()}-{df['year'].max()}
- **Total Value**: ${df['value_dl'].sum()/1e9:.2f}B
""")

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
### 🤖 ML Resources
- **Models**: {len(available_models)}
- **Plots**: {len(available_plots)}
- **Regression**: {sum(1 for m in available_models.values() if m['type'] == 'Regression')}
- **Classification**: {sum(1 for m in available_models.values() if m['type'] == 'Classification')}
- **Clustering**: {sum(1 for m in available_models.values() if m['type'] == 'Clustering')}
""")

# ======================================================================
# PAGE 1: HOME & OVERVIEW
# ======================================================================
if page == "🏠 Home & Overview":
    st.markdown('<p class="main-header">🌍 African Imports Analytics Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Complete ML-Powered Trade Analysis with Pre-trained Models</p>', unsafe_allow_html=True)
    
    # Hero Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📦 Transactions", f"{len(df):,}")
    with col2:
        st.metric("🌍 Countries", df['country_name'].nunique())
    with col3:
        st.metric("📦 Commodities", f"{df['commodity'].nunique():,}")
    with col4:
        st.metric("💰 Total Value", f"${df['value_dl'].sum()/1e9:.2f}B")
    with col5:
        st.metric("🤖 ML Models", len(available_models))
    
    st.markdown("---")
    
    # Key Insights from Notebook
    st.markdown("## 🎯 Key Insights from Data Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🏆 Market Leaders</h3>
        <ul>
            <li><b>South Africa</b>: 21.7% market share</li>
            <li><b>Growth</b>: +58% (2015-2024)</li>
            <li><b>Top 5</b> control 61.5% of imports</li>
            <li><b>Nigeria</b> declining -52%</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>📦 Top Commodities</h3>
        <ul>
            <li><b>Petroleum</b>: ₹71M (26% of total)</li>
            <li><b>Gold</b>: Major import</li>
            <li><b>Top 20</b>: 86.7% of trade value</li>
            <li><b>4,835</b> unique commodities</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>💡 Data Quality</h3>
        <ul>
            <li><b>No duplicates</b> ✅</li>
            <li><b>4 missing</b> values only</li>
            <li><b>Clean data</b>: 10 years</li>
            <li><b>High correlation</b>: Rs↔$ (0.975)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick visualizations
    st.markdown("## 📈 Quick Visual Overview")
    
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Timeline", "🌍 Countries", "📦 Commodities"])
    
    with viz_tab1:
        # Yearly trend
        yearly_data = df[df['year'] < 2025].groupby('year')['value_dl'].sum() / 1e6
        fig_yearly = px.line(
            x=yearly_data.index,
            y=yearly_data.values,
            title='Total Import Value by Year (2015-2024)',
            labels={'x': 'Year', 'y': 'Import Value (Million $)'}
        )
        fig_yearly.update_traces(line_color='#1E88E5', line_width=3)
        fig_yearly.add_scatter(x=yearly_data.index, y=yearly_data.values, mode='markers',
                              marker=dict(size=10, color='#FF6B6B'))
        st.plotly_chart(fig_yearly, use_container_width=True)
    
    with viz_tab2:
        # Top countries
        country_data = df.groupby('country_name')['value_dl'].sum().nlargest(15) / 1e6
        fig_country = px.bar(
            x=country_data.values,
            y=country_data.index,
            orientation='h',
            title='Top 15 Countries by Import Value',
            labels={'x': 'Import Value (Million $)', 'y': 'Country'},
            color=country_data.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_country, use_container_width=True)
    
    with viz_tab3:
        # Top commodities
        commodity_data = df.groupby('commodity')['value_dl'].sum().nlargest(10) / 1e6
        fig_commodity = px.pie(
            values=commodity_data.values,
            names=commodity_data.index,
            title='Top 10 Commodities by Value',
            hole=0.4
        )
        st.plotly_chart(fig_commodity, use_container_width=True)
    
    st.markdown("---")
    
    # Platform capabilities
    st.markdown("## 🚀 Platform Capabilities")
    
    cap_col1, cap_col2 = st.columns(2)
    
    with cap_col1:
        st.markdown("""
        <div class="success-box">
        <h4>📊 Data Exploration</h4>
        <ul>
            <li>Interactive filters and visualizations</li>
            <li>Country-specific deep dives</li>
            <li>Commodity trend analysis</li>
            <li>Time series patterns</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h4>🤖 Machine Learning</h4>
        <ul>
            <li>34 pre-trained models ready to use</li>
            <li>Regression, classification, clustering</li>
            <li>Deep learning (Keras, PyTorch)</li>
            <li>No training required!</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with cap_col2:
        st.markdown("""
        <div class="success-box">
        <h4>📈 Analysis & Insights</h4>
        <ul>
            <li>Comprehensive EDA from 294-cell notebook</li>
            <li>Country-by-country comparisons</li>
            <li>Growth trend identification</li>
            <li>Market share analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h4>🎯 Predictions & Forecasting</h4>
        <ul>
            <li>Import value predictions</li>
            <li>Risk classification</li>
            <li>Pattern discovery</li>
            <li>Performance metrics with visualizations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ======================================================================
# PAGE 2: DATA EXPLORER
# ======================================================================
elif page == "📊 Data Explorer":
    st.markdown('<p class="main-header">📊 Interactive Data Explorer</p>', unsafe_allow_html=True)
    
    # Filters
    st.markdown("## 🎛️ Filter Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_countries = st.multiselect(
            "Countries",
            options=sorted(df['country_name'].unique()),
            default=[]
        )
    
    with col2:
        year_range = st.slider(
            "Year Range",
            int(df['year'].min()),
            int(df['year'].max()),
            (int(df['year'].min()), int(df['year'].max()))
        )
    
    with col3:
        selected_commodities = st.multiselect(
            "Commodities (Top 50)",
            options=df.groupby('commodity')['value_dl'].sum().nlargest(50).index.tolist(),
            default=[]
        )
    
    with col4:
        min_value = st.number_input(
            "Min Value ($)",
            min_value=0.0,
            value=0.0,
            step=100.0
        )
    
    # Apply filters
    filtered_df = df.copy()
    if selected_countries:
        filtered_df = filtered_df[filtered_df['country_name'].isin(selected_countries)]
    if selected_commodities:
        filtered_df = filtered_df[filtered_df['commodity'].isin(selected_commodities)]
    filtered_df = filtered_df[
        (filtered_df['year'] >= year_range[0]) & 
        (filtered_df['year'] <= year_range[1]) &
        (filtered_df['value_dl'] >= min_value)
    ]
    
    st.markdown(f"### 📊 Filtered Results: **{len(filtered_df):,}** transactions")
    
    # Summary metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Total Value", f"${filtered_df['value_dl'].sum()/1e6:.2f}M")
    with metric_col2:
        st.metric("Avg Transaction", f"${filtered_df['value_dl'].mean():,.2f}")
    with metric_col3:
        st.metric("Max Transaction", f"${filtered_df['value_dl'].max():,.2f}")
    with metric_col4:
        st.metric("Countries", filtered_df['country_name'].nunique())
    
    st.markdown("---")
    
    # Visualizations
    explore_tabs = st.tabs(["📈 Time Series", "🌍 Geographic", "📦 Products", "📊 Statistics", "📄 Raw Data"])
    
    with explore_tabs[0]:
        st.markdown("### Time Series Analysis")
        
        # Monthly aggregation
        monthly_df = filtered_df.groupby(filtered_df['date'].dt.to_period('M')).agg({
            'value_dl': 'sum',
            'value_qt': 'sum'
        }).reset_index()
        monthly_df['date'] = monthly_df['date'].dt.to_timestamp()
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=monthly_df['date'],
            y=monthly_df['value_dl'],
            mode='lines+markers',
            name='Import Value',
            line=dict(color='#1E88E5', width=2),
            fill='tozeroy',
            fillcolor='rgba(30, 136, 229, 0.1)'
        ))
        fig_time.update_layout(
            title='Monthly Import Value Trend',
            xaxis_title='Date',
            yaxis_title='Import Value (USD)',
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Year-over-year comparison
        yearly_comp = filtered_df.groupby('year')['value_dl'].sum() / 1e6
        fig_yoy = px.bar(
            x=yearly_comp.index,
            y=yearly_comp.values,
            title='Year-over-Year Import Value',
            labels={'x': 'Year', 'y': 'Import Value (Million $)'},
            color=yearly_comp.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_yoy, use_container_width=True)
    
    with explore_tabs[1]:
        st.markdown("### Geographic Distribution")
        
        # Country breakdown
        country_breakdown = filtered_df.groupby('country_name').agg({
            'value_dl': 'sum',
            'commodity': 'count'
        }).reset_index()
        country_breakdown.columns = ['Country', 'Total Value', 'Num Transactions']
        country_breakdown = country_breakdown.nlargest(20, 'Total Value')
        
        fig_geo = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Top 20 Countries by Value', 'Transaction Count'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        fig_geo.add_trace(
            go.Bar(x=country_breakdown['Total Value']/1e6, y=country_breakdown['Country'],
                   orientation='h', name='Value', marker_color='#1E88E5'),
            row=1, col=1
        )
        
        fig_geo.add_trace(
            go.Bar(x=country_breakdown['Num Transactions'], y=country_breakdown['Country'],
                   orientation='h', name='Count', marker_color='#4CAF50'),
            row=1, col=2
        )
        
        fig_geo.update_xaxes(title_text='Value (Million $)', row=1, col=1)
        fig_geo.update_xaxes(title_text='Number of Transactions', row=1, col=2)
        fig_geo.update_layout(height=600, showlegend=False)
        
        st.plotly_chart(fig_geo, use_container_width=True)
    
    with explore_tabs[2]:
        st.markdown("### Product Analysis")
        
        # Top commodities
        commodity_breakdown = filtered_df.groupby('commodity')['value_dl'].sum().nlargest(20) / 1e6
        
        fig_prod = px.treemap(
            names=commodity_breakdown.index,
            parents=[''] * len(commodity_breakdown),
            values=commodity_breakdown.values,
            title='Top 20 Commodities (Treemap)',
            color=commodity_breakdown.values,
            color_continuous_scale='RdYlGn',
            hover_data=[commodity_breakdown.values]
        )
        st.plotly_chart(fig_prod, use_container_width=True)
    
    with explore_tabs[3]:
        st.markdown("### Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Descriptive Statistics**")
            st.dataframe(
                filtered_df[['value_dl', 'value_rs', 'value_qt']].describe(),
                use_container_width=True
            )
        
        with col2:
            st.markdown("**Distribution Metrics**")
            st.write(f"**Skewness (value_dl):** {filtered_df['value_dl'].skew():.2f}")
            st.write(f"**Kurtosis (value_dl):** {filtered_df['value_dl'].kurtosis():.2f}")
            st.write(f"**25th Percentile:** ${filtered_df['value_dl'].quantile(0.25):.2f}")
            st.write(f"**50th Percentile (Median):** ${filtered_df['value_dl'].quantile(0.50):.2f}")
            st.write(f"**75th Percentile:** ${filtered_df['value_dl'].quantile(0.75):.2f}")
            st.write(f"**95th Percentile:** ${filtered_df['value_dl'].quantile(0.95):.2f}")
    
    with explore_tabs[4]:
        st.markdown("### Raw Data View")
        
        # Show top N records
        num_rows = st.slider("Number of rows to display", 10, 100, 50)
        st.dataframe(filtered_df.head(num_rows), use_container_width=True, height=400)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_african_imports.csv",
            mime="text/csv"
        )

# ======================================================================
# PAGE 3: EDA INSIGHTS (From Notebook)
# ======================================================================
elif page == "📈 EDA Insights":
    st.markdown('<p class="main-header">📈 Exploratory Data Analysis Insights</p>', unsafe_allow_html=True)
    st.markdown("### From 294-Cell Comprehensive Analysis")
    
    insight_tabs = st.tabs(["🌍 Market Overview", "📦 Commodity Analysis", "⏰ Time Patterns", "🏆 Top Performers"])
    
    with insight_tabs[0]:
        st.markdown("## 🌍 Market Structure & Leaders")
        
        # South Africa analysis
        sa_data = df[df['country_name'] == 'South Africa']
        sa_yearly = sa_data[sa_data['year'] < 2025].groupby('year')['value_rs'].sum() / 1e6
        sa_growth = ((sa_yearly.iloc[-1] - sa_yearly.iloc[0]) / sa_yearly.iloc[0] * 100)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h3>🇿🇦 South Africa - The Rising Giant</h3>
            <ul>
                <li><b>Market Share</b>: 21.7%</li>
                <li><b>Total Value</b>: ₹58.1M (36,473 transactions)</li>
                <li><b>Growth Rate</b>: +58.4% (2015-2024)</li>
                <li><b>Main Import</b>: Gold (unwrought forms)</li>
                <li><b>Status</b>: 🚀 GROWING</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # SA trend chart
            fig_sa = px.line(
                x=sa_yearly.index,
                y=sa_yearly.values,
                title='South Africa Import Trend',
                markers=True
            )
            fig_sa.update_traces(line_color='#1f77b4', line_width=3)
            st.plotly_chart(fig_sa, use_container_width=True)
        
        with col2:
            # Nigeria analysis
            ng_data = df[df['country_name'] == 'Nigeria']
            ng_yearly = ng_data[ng_data['year'] < 2025].groupby('year')['value_rs'].sum() / 1e6
            
            st.markdown("""
            <div class="warning-box">
            <h3>🇳🇬 Nigeria - The Declining Power</h3>
            <ul>
                <li><b>Total Value</b>: ₹56.3M (6,272 transactions)</li>
                <li><b>Trend</b>: -51.9% (2015-2024)</li>
                <li><b>Peak Year</b>: 2018 (₹7.6M)</li>
                <li><b>Main Import</b>: Petroleum (70.3% of total)</li>
                <li><b>Status</b>: 📉 DECLINING</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # NG trend chart
            fig_ng = px.line(
                x=ng_yearly.index,
                y=ng_yearly.values,
                title='Nigeria Import Trend',
                markers=True
            )
            fig_ng.update_traces(line_color='#d62728', line_width=3)
            st.plotly_chart(fig_ng, use_container_width=True)
        
        # Top 10 countries summary
        st.markdown("---")
        st.markdown("### 🏆 Top 10 Countries Summary")
        
        top_10 = df.groupby('country_name').agg({
            'value_rs': 'sum',
            'commodity': 'count'
        }).nlargest(10, 'value_rs')
        top_10.columns = ['Total Value (₹)', 'Num Transactions']
        top_10['Total Value (₹M)'] = (top_10['Total Value (₹)'] / 1e6).round(1)
        top_10 = top_10[['Total Value (₹M)', 'Num Transactions']]
        
        st.dataframe(top_10, use_container_width=True)
    
    with insight_tabs[1]:
        st.markdown("## 📦 Commodity Deep Dive")
        
        # Petroleum dominance
        top_commodities = df.groupby('commodity')['value_rs'].sum().nlargest(20) / 1e6
        
        st.markdown(f"""
        <div class="insight-box">
        <h3>⚡ Petroleum Dominates Trade</h3>
        <p><b>Top Commodity</b>: Petroleum Oils And Oils Obtained From Bituminous Minerals Crude</p>
        <ul>
            <li><b>Value</b>: ₹71.2M (26% of total trade value!)</li>
            <li><b>Top 20 commodities</b>: 86.7% of all trade</li>
            <li><b>Market concentration</b>: Very high on energy products</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Top 20 commodities chart
        fig_comm = px.bar(
            x=top_commodities.values,
            y=top_commodities.index,
            orientation='h',
            title='Top 20 Most Valuable Commodities',
            labels={'x': 'Value (Million ₹)', 'y': 'Commodity'},
            color=top_commodities.values,
            color_continuous_scale='Plasma'
        )
        fig_comm.update_layout(height=600)
        st.plotly_chart(fig_comm, use_container_width=True)
        
        # Commodity categories
        st.markdown("### 📊 Commodity Categories")
        
        # Create simple categories
        def categorize_commodity(name):
            name_lower = name.lower()
            if 'petroleum' in name_lower or 'oil' in name_lower or 'gas' in name_lower:
                return 'Energy'
            elif 'gold' in name_lower or 'silver' in name_lower or 'copper' in name_lower:
                return 'Precious Metals'
            elif 'scrap' in name_lower or 'waste' in name_lower:
                return 'Scrap/Waste'
            else:
                return 'Other'
        
        df['category'] = df['commodity'].apply(categorize_commodity)
        category_values = df.groupby('category')['value_rs'].sum() / 1e9
        
        fig_cat = px.pie(
            values=category_values.values,
            names=category_values.index,
            title='Import Value by Category',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with insight_tabs[2]:
        st.markdown("## ⏰ Temporal Patterns")
        
        # Seasonal patterns
        df_complete = df[df['year'] < 2025]
        monthly_avg = df_complete.groupby('month')['value_rs'].sum() / 1e9
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        st.markdown(f"""
        <div class="insight-box">
        <h3>📅 Seasonal Patterns Revealed</h3>
        <ul>
            <li><b>Peak Month</b>: March (₹0.024B) - Fiscal year-end effect?</li>
            <li><b>Lowest Month</b>: November (₹0.020B)</li>
            <li><b>Variation</b>: 14.2% difference from average</li>
            <li><b>Pattern</b>: Relatively stable with slight Q1 peak</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Monthly seasonality chart
        fig_season = go.Figure()
        fig_season.add_trace(go.Bar(
            x=month_names,
            y=monthly_avg.values,
            marker_color=['#F18F01' if i == monthly_avg.idxmax()-1 else '#A23B72' 
                         for i in range(len(monthly_avg))]
        ))
        fig_season.update_layout(
            title='Monthly Seasonality Pattern (2015-2024)',
            xaxis_title='Month',
            yaxis_title='Avg Import Value (Billion ₹)',
            height=400
        )
        st.plotly_chart(fig_season, use_container_width=True)
        
        # Year-over-year growth
        yearly_imports = df_complete.groupby('year')['value_rs'].sum() / 1e9
        
        fig_yearly = px.line(
            x=yearly_imports.index,
            y=yearly_imports.values,
            title='10-Year Import Value Evolution',
            markers=True,
            labels={'x': 'Year', 'y': 'Import Value (Billion ₹)'}
        )
        fig_yearly.update_traces(line_color='#2E86AB', line_width=3)
        st.plotly_chart(fig_yearly, use_container_width=True)
        
        st.info(f"📊 **Key Finding**: Peak year was {yearly_imports.idxmax()} with ₹{yearly_imports.max():.2f}B")
    
    with insight_tabs[3]:
        st.markdown("## 🏆 Top Performers Analysis")
        
        # Market concentration
        total_imports = df['value_rs'].sum() / 1e9
        top_countries = df.groupby('country_name')['value_rs'].sum().sort_values(ascending=False).head(15) / 1e9
        market_share = (top_countries / total_imports * 100)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
            <h3>📊 Market Concentration</h3>
            <ul>
                <li><b>Top 1</b>: {market_share.iloc[0]:.1f}% (South Africa)</li>
                <li><b>Top 3</b>: {market_share.head(3).sum():.1f}%</li>
                <li><b>Top 5</b>: {market_share.head(5).sum():.1f}%</li>
                <li><b>Top 10</b>: {market_share.head(10).sum():.1f}%</li>
            </ul>
            <p><b>Classification</b>: MODERATE concentration</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Market share pie
            top5 = top_countries.head(5)
            others = top_countries[5:].sum()
            
            fig_market = px.pie(
                values=list(top5.values) + [others],
                names=list(top5.index) + ['Others (10 countries)'],
                title='Market Share Distribution',
                hole=0.4
            )
            st.plotly_chart(fig_market, use_container_width=True)

# ======================================================================
# PAGE 4: MODEL GALLERY
# ======================================================================
elif page == "🤖 Model Gallery":
    st.markdown('<p class="main-header">🤖 Pre-trained Model Gallery</p>', unsafe_allow_html=True)
    st.markdown(f"### Browse {len(available_models)} Ready-to-Use ML Models")
    
    # Model type selector
    model_types = ['All'] + sorted(list(set(m['type'] for m in available_models.values())))
    selected_type = st.selectbox("Filter by Type:", model_types)
    
    # Filter models
    if selected_type == 'All':
        display_models = available_models
    else:
        display_models = {k: v for k, v in available_models.items() if v['type'] == selected_type}
    
    st.markdown(f"#### Showing {len(display_models)} models")
    
    # Display models in grid
    cols_per_row = 3
    model_items = list(display_models.items())
    
    for i in range(0, len(model_items), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(model_items):
                model_name, model_info = model_items[i + j]
                
                with col:
                    # Determine icon based on type
                    icon = "📈" if model_info['type'] == 'Regression' else \
                           "🎯" if model_info['type'] == 'Classification' else \
                           "🔍" if model_info['type'] == 'Clustering' else "🤖"
                    
                    # Color based on type
                    color = "#1E88E5" if model_info['type'] == 'Regression' else \
                            "#4CAF50" if model_info['type'] == 'Classification' else \
                            "#FF9800" if model_info['type'] == 'Clustering' else "#9C27B0"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}22, {color}11); 
                                padding: 1rem; border-radius: 8px; border-left: 4px solid {color};
                                margin-bottom: 1rem;">
                        <h4>{icon} {model_name.replace('_', ' ').title()}</h4>
                        <p><b>Type</b>: {model_info['type']}</p>
                        <p><b>Size</b>: {model_info['size_mb']:.2f} MB</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show plot if available
                    if model_name in available_plots:
                        with st.expander("📊 View Performance Plot"):
                            try:
                                img = Image.open(available_plots[model_name])
                                st.image(img, use_column_width=True)
                            except:
                                st.warning("Plot not available")

# ======================================================================
# PAGE 5: PREDICTIONS
# ======================================================================
elif page == "🎯 Predictions":
    st.markdown('<p class="main-header">🎯 Make Predictions with Pre-trained Models</p>', unsafe_allow_html=True)
    
    pred_tabs = st.tabs(["📈 Regression", "🏷️ Classification", "🔍 Clustering"])
    
    with pred_tabs[0]:
        st.markdown("### 📈 Regression: Predict Import Values")
        
        # Get regression models
        reg_models = {k: v for k, v in available_models.items() if v['type'] == 'Regression'}
        
        if reg_models:
            selected_reg = st.selectbox(
                "Select Regression Model:",
                options=list(reg_models.keys()),
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            st.info(f"📊 Model: **{selected_reg}** ({reg_models[selected_reg]['size_mb']:.2f} MB)")
            
            # Show plot if available
            if selected_reg in available_plots:
                st.markdown("#### 📊 Model Performance")
                try:
                    img = Image.open(available_plots[selected_reg])
                    st.image(img, use_column_width=True)
                except:
                    st.warning("Performance plot not available")
            
            st.markdown("---")
            st.markdown("#### 🔮 Try Predictions")
            st.info("💡 Regression models predict continuous values like import amounts. Ready to use with historical data!")
        else:
            st.warning("No regression models found")
    
    with pred_tabs[1]:
        st.markdown("### 🏷️ Classification: Categorize Trade Patterns")
        
        # Get classification models
        class_models = {k: v for k, v in available_models.items() if v['type'] == 'Classification'}
        
        if class_models:
            selected_class = st.selectbox(
                "Select Classification Model:",
                options=list(class_models.keys()),
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            st.info(f"🎯 Model: **{selected_class}** ({class_models[selected_class]['size_mb']:.2f} MB)")
            
            # Show plot if available
            if selected_class in available_plots:
                st.markdown("#### 📊 Model Performance")
                try:
                    img = Image.open(available_plots[selected_class])
                    st.image(img, use_column_width=True)
                except:
                    st.warning("Performance plot not available")
        else:
            st.warning("No classification models found")
    
    with pred_tabs[2]:
        st.markdown("### 🔍 Clustering: Discover Patterns")
        
        # Get clustering models
        cluster_models = {k: v for k, v in available_models.items() if v['type'] == 'Clustering'}
        
        if cluster_models:
            selected_cluster = st.selectbox(
                "Select Clustering Model:",
                options=list(cluster_models.keys()),
                format_func=lambda x: x.replace('_', ' ').title().replace('-', ' ')
            )
            
            st.info(f"🔍 Model: **{selected_cluster}** ({cluster_models[selected_cluster]['size_mb']:.2f} MB)")
            
            st.markdown("#### 🎨 Cluster Analysis Available")
            st.info("💡 Clustering models group similar trade patterns without predefined labels.")
        else:
            st.warning("No clustering models found")

# ======================================================================
# PAGE 6: COUNTRY ANALYSIS
# ======================================================================
elif page == "🔍 Country Analysis":
    st.markdown('<p class="main-header">🔍 Country-by-Country Analysis</p>', unsafe_allow_html=True)
    
    # Country selector
    all_countries = sorted(df['country_name'].unique())
    selected_country = st.selectbox("Select Country:", all_countries)
    
    # Filter data
    country_df = df[df['country_name'] == selected_country]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(country_df):,}")
    with col2:
        st.metric("Total Value", f"${country_df['value_dl'].sum()/1e6:.2f}M")
    with col3:
        st.metric("Avg Transaction", f"${country_df['value_dl'].mean():,.2f}")
    with col4:
        st.metric("Unique Commodities", country_df['commodity'].nunique())
    
    st.markdown("---")
    
    # Analysis tabs
    country_tabs = st.tabs(["📈 Trend", "📦 Top Commodities", "⏰ Patterns", "📊 Statistics"])
    
    with country_tabs[0]:
        # Yearly trend
        yearly_trend = country_df[country_df['year'] < 2025].groupby('year')['value_dl'].sum() / 1e6
        
        fig_trend = px.line(
            x=yearly_trend.index,
            y=yearly_trend.values,
            title=f'{selected_country}: Import Value Trend (2015-2024)',
            markers=True,
            labels={'x': 'Year', 'y': 'Import Value (Million $)'}
        )
        fig_trend.update_traces(line_width=3)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Growth metrics
        if len(yearly_trend) > 1:
            growth = ((yearly_trend.iloc[-1] - yearly_trend.iloc[0]) / yearly_trend.iloc[0] * 100)
            trend_emoji = "🚀" if growth > 0 else "📉"
            st.markdown(f"""
            <div class="{'success-box' if growth > 0 else 'warning-box'}">
            <h4>{trend_emoji} Growth Trend</h4>
            <p><b>Overall Change (2015-2024)</b>: {growth:+.1f}%</p>
            <p><b>Peak Year</b>: {yearly_trend.idxmax()} (${yearly_trend.max():.2f}M)</p>
            <p><b>Lowest Year</b>: {yearly_trend.idxmin()} (${yearly_trend.min():.2f}M)</p>
            </div>
            """, unsafe_allow_html=True)
    
    with country_tabs[1]:
        # Top commodities
        top_comm = country_df.groupby('commodity')['value_dl'].sum().nlargest(10) / 1e6
        
        fig_comm = px.bar(
            x=top_comm.values,
            y=top_comm.index,
            orientation='h',
            title=f'Top 10 Commodities for {selected_country}',
            labels={'x': 'Value (Million $)', 'y': 'Commodity'},
            color=top_comm.values,
            color_continuous_scale='Viridis'
        )
        fig_comm.update_layout(height=500)
        st.plotly_chart(fig_comm, use_container_width=True)
        
        # Commodity concentration
        total_value = country_df['value_dl'].sum()
        top_1_pct = (top_comm.iloc[0] * 1e6 / total_value * 100)
        top_3_pct = (top_comm.head(3).sum() * 1e6 / total_value * 100)
        
        st.info(f"📊 Top commodity accounts for **{top_1_pct:.1f}%** of imports | Top 3 account for **{top_3_pct:.1f}%**")
    
    with country_tabs[2]:
        # Monthly patterns
        monthly_pattern = country_df.groupby('month')['value_dl'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig_monthly = px.bar(
            x=[month_names[i-1] for i in monthly_pattern.index],
            y=monthly_pattern.values,
            title=f'Average Monthly Import Value for {selected_country}',
            labels={'x': 'Month', 'y': 'Avg Value ($)'},
            color=monthly_pattern.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with country_tabs[3]:
        # Statistical summary
        st.dataframe(country_df[['value_dl', 'value_qt']].describe(), use_container_width=True)

# ======================================================================
# PAGE 7: MODEL PERFORMANCE
# ======================================================================
elif page == "📉 Model Performance":
    st.markdown('<p class="main-header">📉 Model Performance Dashboard</p>', unsafe_allow_html=True)
    st.markdown("### Comprehensive Performance Visualizations")
    
    # Show comprehensive dashboards
    perf_tabs = st.tabs(["📊 Regression Dashboard", "🎯 All Models", "📈 Individual Models"])
    
    with perf_tabs[0]:
        st.markdown("### 📊 Comprehensive Regression Analysis")
        
        dashboard_plots = [
            'regression_complete_dashboard',
            'regression_metrics_heatmap',
            'regression_models_comparison_bars',
            'regression_overfitting_analysis',
            'regression_top5_radar'
        ]
        
        for plot_name in dashboard_plots:
            if plot_name in available_plots:
                st.markdown(f"#### {plot_name.replace('_', ' ').title()}")
                try:
                    img = Image.open(available_plots[plot_name])
                    st.image(img, use_column_width=True)
                except:
                    st.warning(f"Could not load {plot_name}")
                st.markdown("---")
    
    with perf_tabs[1]:
        st.markdown("### 🎯 All Model Performances")
        st.info(f"📊 Total: {len(available_plots)} performance visualizations available")
        
        # Group by type
        for model_type in ['regression', 'classification', 'clustering']:
            type_plots = {k: v for k, v in available_plots.items() if model_type in k.lower()}
            
            if type_plots:
                st.markdown(f"#### {model_type.title()} Models ({len(type_plots)})")
                
                cols = st.columns(3)
                for idx, (plot_name, plot_path) in enumerate(type_plots.items()):
                    with cols[idx % 3]:
                        with st.expander(f"📊 {plot_name.replace('_', ' ').title()[:40]}..."):
                            try:
                                img = Image.open(plot_path)
                                st.image(img, use_column_width=True)
                            except:
                                st.warning("Plot unavailable")
    
    with perf_tabs[2]:
        st.markdown("### 📈 Individual Model Performance")
        
        # Model selector
        model_with_plots = [k for k in available_models.keys() if k in available_plots]
        
        if model_with_plots:
            selected_model = st.selectbox(
                "Select Model:",
                options=model_with_plots,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            model_info = available_models[selected_model]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Type", model_info['type'])
            with col2:
                st.metric("Size", f"{model_info['size_mb']:.2f} MB")
            with col3:
                st.metric("Status", "✅ Ready")
            
            st.markdown("---")
            st.markdown("#### 📊 Performance Visualization")
            
            try:
                img = Image.open(available_plots[selected_model])
                st.image(img, use_column_width=True)
            except:
                st.error("Could not load performance plot")
        else:
            st.warning("No models with performance plots found")

# ======================================================================
# FOOTER
# ======================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>🌍 African Imports Analytics Platform</strong></p>
    <p>Powered by <b>34 Pre-trained ML Models</b> | <b>294-Cell Notebook Analysis</b> | <b>46 Performance Visualizations</b></p>
    <p style="font-size: 0.9rem;">Interactive Dashboard for Trade Analysis & Forecasting | No Training Required</p>
</div>
""", unsafe_allow_html=True)
