"""
Dashboard 3: ML Models
Interactive machine learning models for trade analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
from lightgbm import LGBMRegressor


def render(df):
    """Render the ML Models dashboard"""
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