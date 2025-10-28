"""
Dashboard 1: Data Overview
Complete dataset analysis and export functionality
"""

import streamlit as st
from datetime import datetime
from utils import format_currency


def render(df):
    """Render the Data Overview dashboard"""
    
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
