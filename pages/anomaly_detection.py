"""
Dashboard 4: Anomaly Detection
Statistical and ML-based anomaly detection for trade data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.ensemble import IsolationForest
from utils import format_currency


def detect_statistical_anomalies(df, column='value_dl', method='zscore', threshold=3):
    """Detect anomalies using statistical methods"""
    df = df.copy()
    
    if method == 'zscore':
        # Z-score method
        df['z_score'] = np.abs(stats.zscore(df[column].fillna(0)))
        df['is_anomaly'] = df['z_score'] > threshold
        df['anomaly_score'] = df['z_score']
        
    elif method == 'iqr':
        # IQR method
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df['is_anomaly'] = (df[column] < lower_bound) | (df[column] > upper_bound)
        # Normalize score
        df['anomaly_score'] = np.abs(df[column] - df[column].median()) / (IQR + 1e-10)
        
    elif method == 'mad':
        # Median Absolute Deviation
        median = df[column].median()
        mad = np.median(np.abs(df[column] - median))
        df['modified_z_score'] = 0.6745 * (df[column] - median) / (mad + 1e-10)
        df['is_anomaly'] = np.abs(df['modified_z_score']) > threshold
        df['anomaly_score'] = np.abs(df['modified_z_score'])
    
    return df


def detect_time_series_anomalies(df, value_col='value_dl', window=30, threshold=2):
    """Detect anomalies in time series using moving statistics"""
    df = df.copy().sort_values('date')
    
    # Calculate rolling statistics
    df['rolling_mean'] = df[value_col].rolling(window=window, min_periods=1).mean()
    df['rolling_std'] = df[value_col].rolling(window=window, min_periods=1).std()
    
    # Deviation from moving average
    df['deviation'] = (df[value_col] - df['rolling_mean']) / (df['rolling_std'] + 1e-10)
    df['is_ts_anomaly'] = np.abs(df['deviation']) > threshold
    df['ts_anomaly_score'] = np.abs(df['deviation'])
    
    return df


def detect_price_anomalies(df):
    """Detect unit price anomalies"""
    df = df.copy()
    
    # Calculate unit price where quantity available
    mask = df['value_qt'] > 0
    df.loc[mask, 'unit_price'] = df.loc[mask, 'value_dl'] / df.loc[mask, 'value_qt']
    
    # For each commodity, detect price anomalies
    df['price_anomaly'] = False
    df['price_anomaly_score'] = 0.0
    
    for commodity in df['commodity'].unique():
        commodity_mask = df['commodity'] == commodity
        commodity_data = df[commodity_mask & mask]
        
        if len(commodity_data) > 10:  # Need sufficient data
            median_price = commodity_data['unit_price'].median()
            mad = np.median(np.abs(commodity_data['unit_price'] - median_price))
            
            if mad > 0:
                z_scores = 0.6745 * (df.loc[commodity_mask & mask, 'unit_price'] - median_price) / mad
                df.loc[commodity_mask & mask, 'price_anomaly'] = np.abs(z_scores) > 3
                df.loc[commodity_mask & mask, 'price_anomaly_score'] = np.abs(z_scores)
    
    return df


def render(df):
    """Render the Anomaly Detection dashboard"""
    
    st.markdown('<p class="main-header">🔍 Anomaly Detection</p>', unsafe_allow_html=True)
    st.markdown("### Identify unusual patterns, outliers, and potential data quality issues")
    
    st.markdown("""
    <div class="insight-box">
    <h4>🎯 What This Dashboard Does:</h4>
    <ul>
        <li><b>Statistical Anomalies:</b> Detect unusual values using Z-score, IQR, and MAD methods</li>
        <li><b>Time Series Anomalies:</b> Find sudden spikes, drops, or trend breaks</li>
        <li><b>Price Anomalies:</b> Identify unusual unit prices compared to commodity norms</li>
        <li><b>Transaction Patterns:</b> Spot unusual import frequencies or volumes</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    anomaly_tabs = st.tabs([
        "📊 Overview",
        "📈 Time Series Anomalies",
        "💰 Price Anomalies",
        "🔬 Statistical Analysis",
        "🚨 Critical Alerts"
    ])
    
    # ===== TAB 1: OVERVIEW =====
    with anomaly_tabs[0]:
        st.markdown("## 🎯 Anomaly Detection Overview")
        
        # Detect anomalies using multiple methods
        with st.spinner("Detecting anomalies across all methods..."):
            df_anomalies = detect_statistical_anomalies(df, method='zscore', threshold=3)
            df_anomalies = detect_time_series_anomalies(df_anomalies, window=30, threshold=2)
            df_anomalies = detect_price_anomalies(df_anomalies)
        
        # Count anomalies
        total_records = len(df_anomalies)
        statistical_anomalies = df_anomalies['is_anomaly'].sum()
        ts_anomalies = df_anomalies['is_ts_anomaly'].sum()
        price_anomalies = df_anomalies['price_anomaly'].sum()
        
        # Any anomaly
        df_anomalies['any_anomaly'] = (
            df_anomalies['is_anomaly'] | 
            df_anomalies['is_ts_anomaly'] | 
            df_anomalies['price_anomaly']
        )
        total_anomalies = df_anomalies['any_anomaly'].sum()
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Records", f"{total_records:,}")
        with col2:
            anomaly_pct = (total_anomalies / total_records) * 100
            st.metric("Total Anomalies", f"{total_anomalies:,}", f"{anomaly_pct:.1f}%")
        with col3:
            st.metric("Statistical", f"{statistical_anomalies:,}")
        with col4:
            st.metric("Time Series", f"{ts_anomalies:,}")
        with col5:
            st.metric("Price", f"{price_anomalies:,}")
        
        st.markdown("---")
        
        # Anomaly distribution
        st.markdown("### 📊 Anomaly Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomalies by type
            anomaly_counts = pd.DataFrame({
                'Type': ['Statistical', 'Time Series', 'Price'],
                'Count': [statistical_anomalies, ts_anomalies, price_anomalies]
            })
            
            fig_types = px.bar(
                anomaly_counts,
                x='Type',
                y='Count',
                title='Anomalies by Detection Method',
                color='Count',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_types, use_container_width=True)
        
        with col2:
            # Severity distribution
            df_anomalies['max_score'] = df_anomalies[[
                'anomaly_score', 'ts_anomaly_score', 'price_anomaly_score'
            ]].max(axis=1)
            
            # Categorize severity
            df_anomalies['severity'] = pd.cut(
                df_anomalies['max_score'],
                bins=[0, 2, 3, 5, np.inf],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
            
            severity_counts = df_anomalies[df_anomalies['any_anomaly']]['severity'].value_counts()
            
            fig_severity = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title='Anomaly Severity Distribution',
                color_discrete_map={
                    'Low': '#90EE90',
                    'Medium': '#FFD700',
                    'High': '#FFA500',
                    'Critical': '#FF4500'
                }
            )
            st.plotly_chart(fig_severity, use_container_width=True)
        
        # Top anomalies
        st.markdown("---")
        st.markdown("### 🚨 Top 20 Most Severe Anomalies")
        
        top_anomalies = df_anomalies[df_anomalies['any_anomaly']].nlargest(20, 'max_score')
        
        display_cols = ['date', 'country_name', 'commodity', 'value_dl', 'severity', 'max_score']
        display_anomalies = top_anomalies[display_cols].copy()
        display_anomalies['value_dl'] = display_anomalies['value_dl'].apply(lambda x: format_currency(x))
        display_anomalies.columns = ['Date', 'Country', 'Commodity', 'Value', 'Severity', 'Score']
        
        st.dataframe(display_anomalies, use_container_width=True, height=400)
        
        # Download anomalies
        st.markdown("---")
        csv_anomalies = df_anomalies[df_anomalies['any_anomaly']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download All Anomalies (CSV)",
            data=csv_anomalies,
            file_name=f"anomalies_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # ===== TAB 2: TIME SERIES ANOMALIES =====
    with anomaly_tabs[1]:
        st.markdown("## 📈 Time Series Anomaly Analysis")
        
        st.markdown("""
        <div class="insight-box">
        <p><b>What to look for:</b> Sudden spikes or drops that deviate significantly from the moving average. 
        These could indicate supply disruptions, policy changes, data errors, or market events.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            selected_country_ts = st.selectbox(
                "Select Country:",
                options=['All'] + sorted(df['country_name'].unique().tolist()),
                key='ts_country'
            )
        
        with col2:
            selected_commodity_ts = st.selectbox(
                "Select Commodity:",
                options=['All'] + sorted(df['commodity'].unique().tolist()),
                key='ts_commodity'
            )
        
        # Filter data
        ts_data = df_anomalies.copy()
        if selected_country_ts != 'All':
            ts_data = ts_data[ts_data['country_name'] == selected_country_ts]
        if selected_commodity_ts != 'All':
            ts_data = ts_data[ts_data['commodity'] == selected_commodity_ts]
        
        if len(ts_data) == 0:
            st.warning("No data available for selected filters")
        else:
            # Aggregate by month
            ts_monthly = ts_data.groupby(ts_data['date'].dt.to_period('M').dt.to_timestamp()).agg({
                'value_dl': 'sum',
                'is_ts_anomaly': 'any'
            }).reset_index()
            
            # Recalculate rolling stats on monthly data
            ts_monthly['rolling_mean'] = ts_monthly['value_dl'].rolling(window=3, min_periods=1).mean()
            ts_monthly['rolling_std'] = ts_monthly['value_dl'].rolling(window=3, min_periods=1).std()
            
            # Plot
            fig_ts = go.Figure()
            
            # Actual values
            fig_ts.add_trace(go.Scatter(
                x=ts_monthly['date'],
                y=ts_monthly['value_dl'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='#1E88E5', width=2),
                marker=dict(size=6)
            ))
            
            # Moving average
            fig_ts.add_trace(go.Scatter(
                x=ts_monthly['date'],
                y=ts_monthly['rolling_mean'],
                mode='lines',
                name='3-Month MA',
                line=dict(color='#FF6B6B', width=2, dash='dash')
            ))
            
            # Confidence bands
            upper_band = ts_monthly['rolling_mean'] + 2 * ts_monthly['rolling_std']
            lower_band = ts_monthly['rolling_mean'] - 2 * ts_monthly['rolling_std']
            
            fig_ts.add_trace(go.Scatter(
                x=ts_monthly['date'],
                y=upper_band,
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig_ts.add_trace(go.Scatter(
                x=ts_monthly['date'],
                y=lower_band,
                mode='lines',
                name='Lower Bound',
                fill='tonexty',
                fillcolor='rgba(30, 136, 229, 0.1)',
                line=dict(width=0),
                showlegend=False
            ))
            
            # Mark anomalies
            anomaly_points = ts_monthly[ts_monthly['is_ts_anomaly']]
            if len(anomaly_points) > 0:
                fig_ts.add_trace(go.Scatter(
                    x=anomaly_points['date'],
                    y=anomaly_points['value_dl'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=12, symbol='x', line=dict(width=2))
                ))
            
            fig_ts.update_layout(
                title='Time Series with Anomaly Detection',
                xaxis_title='Date',
                yaxis_title='Import Value (USD)',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_ts, use_container_width=True)
            
            # Stats
            st.markdown("### 📊 Time Series Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Months Analyzed", len(ts_monthly))
            with col2:
                anomaly_months = ts_monthly['is_ts_anomaly'].sum()
                st.metric("Anomalous Months", anomaly_months)
            with col3:
                avg_value = ts_monthly['value_dl'].mean()
                st.metric("Avg Monthly Value", format_currency(avg_value))
            with col4:
                volatility = ts_monthly['value_dl'].std() / (ts_monthly['value_dl'].mean() + 1e-10)
                st.metric("Volatility (CV)", f"{volatility:.2f}")
    
    # ===== TAB 3: PRICE ANOMALIES =====
    with anomaly_tabs[2]:
        st.markdown("## 💰 Price Anomaly Detection")
        
        st.markdown("""
        <div class="insight-box">
        <p><b>Purpose:</b> Identify transactions where the unit price significantly deviates from the commodity's 
        typical price. This can reveal data errors, fraud, or genuine market anomalies.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Filter to records with price data
        price_data = df_anomalies[df_anomalies['value_qt'] > 0].copy()
        
        if len(price_data) == 0:
            st.warning("No quantity data available for price analysis")
        else:
            # Top commodities by price anomalies
            price_anomaly_counts = price_data[price_data['price_anomaly']].groupby('commodity').size().nlargest(20)
            
            fig_price_comm = px.bar(
                x=price_anomaly_counts.values,
                y=price_anomaly_counts.index,
                orientation='h',
                title='Top 20 Commodities by Price Anomaly Count',
                labels={'x': 'Number of Anomalies', 'y': 'Commodity'},
                color=price_anomaly_counts.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_price_comm, use_container_width=True)
            
            # Select commodity for detailed analysis
            st.markdown("---")
            st.markdown("### 🔍 Detailed Price Analysis")
            
            selected_commodity_price = st.selectbox(
                "Select Commodity:",
                options=sorted(price_data['commodity'].unique()),
                key='price_commodity'
            )
            
            commodity_prices = price_data[price_data['commodity'] == selected_commodity_price].copy()
            
            if len(commodity_prices) > 0:
                # Price distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_price_dist = px.histogram(
                        commodity_prices,
                        x='unit_price',
                        nbins=50,
                        title=f'{selected_commodity_price} - Unit Price Distribution',
                        labels={'unit_price': 'Unit Price (USD)'},
                        color_discrete_sequence=['#1E88E5']
                    )
                    
                    # Add median line
                    median_price = commodity_prices['unit_price'].median()
                    fig_price_dist.add_vline(
                        x=median_price,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Median: {median_price:.2f}"
                    )
                    
                    st.plotly_chart(fig_price_dist, use_container_width=True)
                
                with col2:
                    fig_price_box = px.box(
                        commodity_prices,
                        y='unit_price',
                        title=f'{selected_commodity_price} - Price Box Plot',
                        labels={'unit_price': 'Unit Price (USD)'}
                    )
                    st.plotly_chart(fig_price_box, use_container_width=True)
                
                # Price over time
                st.markdown("#### 📈 Price Evolution Over Time")
                
                price_monthly = commodity_prices.groupby(
                    commodity_prices['date'].dt.to_period('M').dt.to_timestamp()
                ).agg({
                    'unit_price': ['mean', 'median', 'std'],
                    'price_anomaly': 'sum'
                }).reset_index()
                
                price_monthly.columns = ['date', 'mean_price', 'median_price', 'std_price', 'anomaly_count']
                
                fig_price_trend = go.Figure()
                
                # Mean price
                fig_price_trend.add_trace(go.Scatter(
                    x=price_monthly['date'],
                    y=price_monthly['mean_price'],
                    mode='lines+markers',
                    name='Mean Price',
                    line=dict(color='#1E88E5', width=2)
                ))
                
                # Median price
                fig_price_trend.add_trace(go.Scatter(
                    x=price_monthly['date'],
                    y=price_monthly['median_price'],
                    mode='lines',
                    name='Median Price',
                    line=dict(color='#FF6B6B', width=2, dash='dash')
                ))
                
                fig_price_trend.update_layout(
                    title=f'{selected_commodity_price} - Price Trends',
                    xaxis_title='Date',
                    yaxis_title='Unit Price (USD)',
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig_price_trend, use_container_width=True)
                
                # Anomalous transactions
                st.markdown("#### 🚨 Anomalous Price Transactions")
                
                price_anomalies_comm = commodity_prices[commodity_prices['price_anomaly']].nlargest(10, 'price_anomaly_score')
                
                if len(price_anomalies_comm) > 0:
                    display_price = price_anomalies_comm[[
                        'date', 'country_name', 'value_dl', 'value_qt', 'unit_price', 'price_anomaly_score'
                    ]].copy()
                    display_price['value_dl'] = display_price['value_dl'].apply(lambda x: format_currency(x))
                    display_price.columns = ['Date', 'Country', 'Value', 'Quantity', 'Unit Price', 'Anomaly Score']
                    
                    st.dataframe(display_price, use_container_width=True)
                else:
                    st.info("No price anomalies detected for this commodity")
    
    # ===== TAB 4: STATISTICAL ANALYSIS =====
    with anomaly_tabs[3]:
        st.markdown("## 🔬 Statistical Anomaly Analysis")
        
        st.markdown("### ⚙️ Detection Method Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            method = st.selectbox(
                "Statistical Method:",
                options=['zscore', 'iqr', 'mad'],
                format_func=lambda x: {
                    'zscore': 'Z-Score (Standard Deviations)',
                    'iqr': 'IQR (Interquartile Range)',
                    'mad': 'MAD (Median Absolute Deviation)'
                }[x]
            )
        
        with col2:
            threshold = st.slider(
                "Sensitivity Threshold:",
                min_value=1.5,
                max_value=5.0,
                value=3.0,
                step=0.5,
                help="Lower = more sensitive (more anomalies)"
            )
        
        # Detect with selected method
        stat_anomalies = detect_statistical_anomalies(df, method=method, threshold=threshold)
        
        anomaly_count = stat_anomalies['is_anomaly'].sum()
        anomaly_pct = (anomaly_count / len(stat_anomalies)) * 100
        
        st.metric("Anomalies Detected", f"{anomaly_count:,}", f"{anomaly_pct:.2f}%")
        
        # Distribution visualization
        st.markdown("---")
        st.markdown("### 📊 Value Distribution with Anomalies")
        
        fig_stat = px.histogram(
            stat_anomalies,
            x='value_dl',
            nbins=100,
            title='Import Value Distribution',
            labels={'value_dl': 'Import Value (USD)'},
            color='is_anomaly',
            color_discrete_map={True: 'red', False: '#1E88E5'},
            marginal='box'
        )
        
        st.plotly_chart(fig_stat, use_container_width=True)
        
        # Anomalies by country
        st.markdown("---")
        st.markdown("### 🌍 Anomalies by Country")
        
        country_anomalies = stat_anomalies[stat_anomalies['is_anomaly']].groupby('country_name').size().nlargest(15)
        
        fig_country = px.bar(
            x=country_anomalies.values,
            y=country_anomalies.index,
            orientation='h',
            title='Top 15 Countries by Anomaly Count',
            labels={'x': 'Number of Anomalies', 'y': 'Country'},
            color=country_anomalies.values,
            color_continuous_scale='Reds'
        )
        
        st.plotly_chart(fig_country, use_container_width=True)
    
    # ===== TAB 5: CRITICAL ALERTS =====
    with anomaly_tabs[4]:
        st.markdown("## 🚨 Critical Anomaly Alerts")
        
        # Filter to critical anomalies
        critical_anomalies = df_anomalies[
            (df_anomalies['any_anomaly']) & 
            (df_anomalies['severity'].isin(['High', 'Critical']))
        ].copy()
        
        st.metric("Critical Anomalies", f"{len(critical_anomalies):,}")
        
        if len(critical_anomalies) > 0:
            # Group by type
            st.markdown("### 📋 Critical Anomalies by Category")
            
            critical_stats = pd.DataFrame({
                'Category': ['Statistical', 'Time Series', 'Price'],
                'Count': [
                    critical_anomalies['is_anomaly'].sum(),
                    critical_anomalies['is_ts_anomaly'].sum(),
                    critical_anomalies['price_anomaly'].sum()
                ]
            })
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(critical_stats, use_container_width=True)
            
            with col2:
                fig_crit = px.pie(
                    critical_stats,
                    values='Count',
                    names='Category',
                    title='Critical Anomalies Breakdown'
                )
                st.plotly_chart(fig_crit, use_container_width=True)
            
            # Recent critical anomalies
            st.markdown("---")
            st.markdown("### 🕐 Recent Critical Anomalies (Last 30 Days)")
            
            recent_date = df_anomalies['date'].max() - pd.Timedelta(days=30)
            recent_critical = critical_anomalies[critical_anomalies['date'] >= recent_date].nlargest(20, 'max_score')
            
            if len(recent_critical) > 0:
                display_recent = recent_critical[[
                    'date', 'country_name', 'commodity', 'value_dl', 'severity', 'max_score'
                ]].copy()
                display_recent['value_dl'] = display_recent['value_dl'].apply(lambda x: format_currency(x))
                display_recent.columns = ['Date', 'Country', 'Commodity', 'Value', 'Severity', 'Score']
                
                st.dataframe(display_recent, use_container_width=True, height=400)
            else:
                st.info("No critical anomalies in the last 30 days")
            
            # Top affected commodities
            st.markdown("---")
            st.markdown("### 📦 Most Affected Commodities")
            
            affected_commodities = critical_anomalies.groupby('commodity').agg({
                'value_dl': 'sum',
                'date': 'count'
            }).nlargest(10, 'date')
            
            affected_commodities.columns = ['Total Value', 'Anomaly Count']
            affected_commodities['Total Value'] = affected_commodities['Total Value'].apply(lambda x: format_currency(x))
            
            st.dataframe(affected_commodities, use_container_width=True)
        else:
            st.success("✅ No critical anomalies detected!")
        
        # Action recommendations
        st.markdown("---")
        st.markdown("""
        <div class="insight-box">
        <h4>💡 Recommended Actions for Anomalies:</h4>
        <ul>
            <li><b>Investigate High/Critical:</b> Review transaction details, verify data accuracy, check for external factors</li>
            <li><b>Data Quality:</b> Anomalies may indicate data entry errors or missing information</li>
            <li><b>Market Events:</b> Sudden changes might reflect policy changes, supply disruptions, or price shocks</li>
            <li><b>Opportunities:</b> Positive anomalies could indicate emerging markets or new trade routes</li>
            <li><b>False Positives:</b> Adjust thresholds if too many legitimate transactions flagged</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
