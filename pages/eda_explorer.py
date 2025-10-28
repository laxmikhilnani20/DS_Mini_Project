"""
Dashboard 2: EDA Explorer
Deep-dive exploratory data analysis with interactive visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import format_currency


def render(df):
    """Render the EDA Explorer dashboard"""
    
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
