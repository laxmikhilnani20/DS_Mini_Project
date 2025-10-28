"""
Dashboard 5: Risk Assessment & Cost Optimization
Concentration risk, volatility analysis, and timing optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import format_currency


def calculate_hhi(data, column):
    """Calculate Herfindahl-Hirschman Index"""
    shares = data[column].value_counts(normalize=True) * 100
    hhi = (shares ** 2).sum()
    return hhi


def calculate_gini(values):
    """Calculate Gini coefficient"""
    sorted_values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
    return gini


def render(df):
    """Render the Risk Assessment & Optimization dashboard"""
    
    st.markdown('<p class="main-header">⚠️ Risk Assessment & Cost Optimization</p>', unsafe_allow_html=True)
    st.markdown("### Identify risks, optimize timing, and reduce costs")
    
    tabs = st.tabs([
        "🎯 Concentration Risk",
        "⏰ Seasonal Timing",
        "📉 Volatility Analysis",
        "💰 Cost Optimization"
    ])
    
    # TAB 1: CONCENTRATION RISK
    with tabs[0]:
        st.markdown("## 🎯 Concentration Risk Analysis")
        
        st.markdown("""
        <div class="insight-box">
        <p><b>Why it matters:</b> Over-reliance on single countries or commodities creates vulnerability.
        High concentration = higher supply chain risk.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate HHI scores
        country_hhi = calculate_hhi(df, 'country_name')
        commodity_hhi = calculate_hhi(df, 'commodity')
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_level = "🔴 HIGH" if country_hhi > 2500 else "🟡 MEDIUM" if country_hhi > 1500 else "🟢 LOW"
            st.metric("Country Concentration (HHI)", f"{country_hhi:.0f}", risk_level)
            
            st.markdown("""
            **HHI Scale:**
            - < 1500: Low risk (diversified)
            - 1500-2500: Moderate risk
            - > 2500: High risk (concentrated)
            """)
        
        with col2:
            risk_level = "🔴 HIGH" if commodity_hhi > 2500 else "🟡 MEDIUM" if commodity_hhi > 1500 else "🟢 LOW"
            st.metric("Commodity Concentration (HHI)", f"{commodity_hhi:.0f}", risk_level)
        
        # Top concentrations
        st.markdown("---")
        st.markdown("### 📊 Market Share Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            country_shares = df.groupby('country_name')['value_dl'].sum().nlargest(10)
            total = df['value_dl'].sum()
            country_shares_pct = (country_shares / total * 100).round(1)
            
            fig_country = px.bar(
                x=country_shares_pct.values,
                y=country_shares_pct.index,
                orientation='h',
                title='Top 10 Countries by Import Share (%)',
                labels={'x': 'Market Share (%)', 'y': 'Country'}
            )
            st.plotly_chart(fig_country, use_container_width=True)
        
        with col2:
            commodity_shares = df.groupby('commodity')['value_dl'].sum().nlargest(10)
            commodity_shares_pct = (commodity_shares / total * 100).round(1)
            
            fig_commodity = px.bar(
                x=commodity_shares_pct.values,
                y=commodity_shares_pct.index,
                orientation='h',
                title='Top 10 Commodities by Import Share (%)',
                labels={'x': 'Market Share (%)', 'y': 'Commodity'}
            )
            st.plotly_chart(fig_commodity, use_container_width=True)
        
        # Lorenz curve
        st.markdown("---")
        st.markdown("### 📈 Inequality Analysis (Lorenz Curve)")
        
        country_values = df.groupby('country_name')['value_dl'].sum().sort_values()
        cumsum = country_values.cumsum()
        cumsum_pct = (cumsum / cumsum.iloc[-1] * 100).values
        x = np.linspace(0, 100, len(cumsum_pct))
        
        gini = calculate_gini(country_values.values)
        
        fig_lorenz = go.Figure()
        fig_lorenz.add_trace(go.Scatter(
            x=x, y=cumsum_pct,
            mode='lines',
            name='Actual Distribution',
            line=dict(color='#1E88E5', width=3)
        ))
        fig_lorenz.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100],
            mode='lines',
            name='Perfect Equality',
            line=dict(color='red', dash='dash', width=2)
        ))
        fig_lorenz.update_layout(
            title=f'Lorenz Curve (Gini Coefficient: {gini:.3f})',
            xaxis_title='Cumulative % of Countries',
            yaxis_title='Cumulative % of Import Value'
        )
        st.plotly_chart(fig_lorenz, use_container_width=True)
        
        st.info(f"**Gini Coefficient: {gini:.3f}** (0 = perfect equality, 1 = total inequality)")
    
    # TAB 2: SEASONAL TIMING
    with tabs[1]:
        st.markdown("## ⏰ Optimal Import Timing Analysis")
        
        st.markdown("""
        <div class="insight-box">
        <p><b>Strategy:</b> Import during cheaper months to save costs. 
        Avoid peak price periods unless urgency requires it.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Select commodity
        commodity_list = sorted(df['commodity'].unique())
        selected_comm = st.selectbox("Select Commodity for Timing Analysis:", commodity_list)
        
        comm_data = df[df['commodity'] == selected_comm].copy()
        
        if len(comm_data) > 0:
            # Monthly price analysis
            comm_data['month_name'] = comm_data['date'].dt.month_name()
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            
            monthly_stats = comm_data.groupby('month_name')['value_dl'].agg(['mean', 'median', 'std', 'count'])
            monthly_stats = monthly_stats.reindex(month_order)
            
            # Calculate seasonal index
            overall_mean = monthly_stats['mean'].mean()
            monthly_stats['seasonal_index'] = (monthly_stats['mean'] / overall_mean * 100).round(1)
            monthly_stats['price_vs_avg'] = ((monthly_stats['mean'] - overall_mean) / overall_mean * 100).round(1)
            
            # Visualization
            fig_seasonal = go.Figure()
            
            fig_seasonal.add_trace(go.Bar(
                x=month_order,
                y=monthly_stats['seasonal_index'].values,
                marker_color=np.where(monthly_stats['seasonal_index'].values < 100, 'green', 'red'),
                text=monthly_stats['seasonal_index'].values,
                texttemplate='%{text:.1f}',
                textposition='outside'
            ))
            
            fig_seasonal.add_hline(y=100, line_dash="dash", line_color="gray", 
                                  annotation_text="Average = 100")
            
            fig_seasonal.update_layout(
                title=f'{selected_comm} - Seasonal Price Index',
                xaxis_title='Month',
                yaxis_title='Seasonal Index (100 = Average)',
                height=500
            )
            
            st.plotly_chart(fig_seasonal, use_container_width=True)
            
            # Best and worst months
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ Best Months to Import (Cheapest)")
                best_months = monthly_stats.nsmallest(3, 'mean')
                for month in best_months.index:
                    savings = abs(monthly_stats.loc[month, 'price_vs_avg'])
                    st.success(f"**{month}**: {savings:.1f}% below average")
            
            with col2:
                st.markdown("### ❌ Months to Avoid (Most Expensive)")
                worst_months = monthly_stats.nlargest(3, 'mean')
                for month in worst_months.index:
                    premium = monthly_stats.loc[month, 'price_vs_avg']
                    st.error(f"**{month}**: {premium:.1f}% above average")
            
            # Savings calculator
            st.markdown("---")
            st.markdown("### 💰 Potential Savings Calculator")
            
            planned_volume = st.number_input(
                "Planned Annual Import Value (USD):",
                min_value=0,
                value=1000000,
                step=100000
            )
            
            if planned_volume > 0:
                avg_price = monthly_stats['mean'].mean()
                best_month_price = monthly_stats['mean'].min()
                worst_month_price = monthly_stats['mean'].max()
                
                # Normalize to monthly portion
                monthly_portion = planned_volume / 12
                
                best_case_cost = (monthly_portion / avg_price) * best_month_price
                worst_case_cost = (monthly_portion / avg_price) * worst_month_price
                
                best_savings = monthly_portion - best_case_cost
                worst_premium = worst_case_cost - monthly_portion
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Monthly Budget", format_currency(monthly_portion))
                with col2:
                    st.metric("Best Month Savings", format_currency(best_savings), 
                             f"{(best_savings/monthly_portion*100):.1f}%")
                with col3:
                    st.metric("Worst Month Premium", format_currency(worst_premium),
                             f"{(worst_premium/monthly_portion*100):.1f}%")
        else:
            st.warning("No data available for selected commodity")
    
    # TAB 3: VOLATILITY
    with tabs[2]:
        st.markdown("## 📉 Price Volatility Analysis")
        
        st.markdown("""
        <div class="insight-box">
        <p><b>Purpose:</b> Identify commodities with unstable prices requiring hedging or safety stock.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate volatility for top commodities
        top_commodities = df.groupby('commodity')['value_dl'].sum().nlargest(20).index
        
        volatility_data = []
        for commodity in top_commodities:
            comm_df = df[df['commodity'] == commodity]
            monthly = comm_df.groupby(comm_df['date'].dt.to_period('M'))['value_dl'].sum()
            
            if len(monthly) > 3:
                mean_val = monthly.mean()
                std_val = monthly.std()
                cv = (std_val / mean_val * 100) if mean_val > 0 else 0
                
                volatility_data.append({
                    'Commodity': commodity,
                    'Mean Value': mean_val,
                    'Std Dev': std_val,
                    'CV (%)': cv,
                    'Risk Level': 'High' if cv > 30 else 'Medium' if cv > 15 else 'Low'
                })
        
        vol_df = pd.DataFrame(volatility_data).sort_values('CV (%)', ascending=False)
        
        # Visualization
        fig_vol = px.bar(
            vol_df,
            x='CV (%)',
            y='Commodity',
            orientation='h',
            title='Price Volatility Ranking (Coefficient of Variation)',
            color='CV (%)',
            color_continuous_scale='Reds',
            labels={'CV (%)': 'Volatility (%)'}
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Table
        st.markdown("### �� Volatility Details")
        display_vol = vol_df.copy()
        display_vol['Mean Value'] = display_vol['Mean Value'].apply(lambda x: format_currency(x))
        display_vol['Std Dev'] = display_vol['Std Dev'].apply(lambda x: format_currency(x))
        display_vol['CV (%)'] = display_vol['CV (%)'].round(1)
        
        st.dataframe(display_vol, use_container_width=True, height=400)
        
        st.markdown("""
        <div class="insight-box">
        <p><b>Risk Categories:</b></p>
        <ul>
            <li><b>Low (CV < 15%):</b> Stable pricing - standard procurement</li>
            <li><b>Medium (CV 15-30%):</b> Moderate volatility - monitor closely</li>
            <li><b>High (CV > 30%):</b> Very volatile - consider hedging strategies</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 4: COST OPTIMIZATION
    with tabs[3]:
        st.markdown("## 💰 Cost Optimization Opportunities")
        
        st.markdown("""
        <div class="insight-box">
        <p><b>Quick Wins:</b> Identify immediate cost-saving opportunities through timing and sourcing.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary recommendations
        st.markdown("### 🎯 Top Optimization Opportunities")
        
        # Calculate potential savings from seasonal timing
        total_value = df['value_dl'].sum()
        
        # Estimate 5-10% savings from optimal timing
        timing_savings = total_value * 0.075
        
        # Estimate 3-5% from concentration reduction (risk premium)
        concentration_savings = total_value * 0.04
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Total Spend", format_currency(total_value))
        with col2:
            st.metric("Seasonal Timing Savings", format_currency(timing_savings), "~7.5%")
        with col3:
            st.metric("Diversification Savings", format_currency(concentration_savings), "~4%")
        
        total_opportunity = timing_savings + concentration_savings
        st.success(f"**Total Optimization Potential: {format_currency(total_opportunity)} ({total_opportunity/total_value*100:.1f}%)**")
        
        # Action items
        st.markdown("---")
        st.markdown("### ✅ Recommended Actions")
        
        st.markdown("""
        1. **Seasonal Procurement Planning**
           - Review commodity seasonal patterns
           - Shift non-urgent imports to cheaper months
           - Build inventory during low-price periods
        
        2. **Supplier Diversification**
           - Reduce concentration in top country/commodity
           - Target HHI < 1500 for lower risk
           - Develop alternate sourcing relationships
        
        3. **Hedging Strategy**
           - Focus on high-volatility commodities (CV > 30%)
           - Consider fixed-price contracts
           - Use financial instruments if available
        
        4. **Volume Optimization**
           - Analyze bulk discount opportunities
           - Balance storage costs vs price savings
           - Consolidate orders for better negotiation
        """)

