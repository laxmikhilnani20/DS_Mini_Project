# 🌍 African Import Analysis - Streamlit Dashboard

## 📋 Overview

This is a **comprehensive Streamlit application** showcasing the complete African Import Analysis project featuring:

- **32 Machine Learning Models** (Regression + Classification + Clustering + Deep Learning)
- **10 Years of Trade Data** (2015-2025)
- **139,566 Transactions** from 59 African countries
- **Interactive Visualizations** with Plotly
- **Production-Ready Models** with full EDA

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd /Users/laxmikhilnani/Documents/GitHub/DS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📊 Dashboard Features

### 🏠 **Page 1: Project Overview**
- Project statistics and key metrics
- Data quality assessment
- Goals achieved summary
- Data leakage discovery and fix explanation
- Model summary table

### 📈 **Page 2: EDA & Insights**
- **Temporal Analysis:** Yearly trends, seasonal patterns, COVID impact
- **Geographic Analysis:** Top importing countries, market concentration
- **Commodity Analysis:** Top commodities, concentration risk
- **Country Spotlights:** Deep dive into South Africa, Nigeria, Egypt

### 🤖 **Page 3: ML Models (32 Total)**
- **Regression (11 models):** Performance comparison, best model (Random Forest, R²=0.7979)
- **Classification (11 models):** Accuracy comparison, best model (Bagging Classifier, 70.80%)
- **Clustering (10 models):** Silhouette scores, best model (Mini-Batch K-Means, 0.4332)
- **Deep Learning (4 models):** TensorFlow/Keras + PyTorch architectures

### 🎯 **Page 4: Interactive Predictions**
- Regression: Predict import values
- Classification: Classify transaction sizes
- Demo interface for model testing

### 📊 **Page 5: Model Comparison**
- Side-by-side comparison of best models
- Performance visualization across categories
- Summary of all 32 models

### 💡 **Page 6: Business Insights**
- **Applications:** Forecasting, risk assessment, market segmentation
- **Strategic Insights:** Critical findings and recommendations
- **Future Work:** Enhancement roadmap

---

## 🗂️ Project Structure

```
DS/
├── streamlit_app.py                          # Main Streamlit application
├── requirements.txt                          # Python dependencies
├── imports-from-african-countries.csv        # Dataset (139K rows)
├── african_imports_models_only_20251102_175254/  # Saved models (32 files)
│   ├── random_forest_(clean)_regression.pkl
│   ├── bagging_(clean)_classification.pkl
│   ├── mini-batch_k-means_clustering.pkl
│   ├── keras_dnn_regression.h5
│   ├── pytorch_nn_regression.pt
│   └── ... (28 more models)
├── README_STREAMLIT_APP.md                   # This file
└── pages/                                    # Optional modular pages
```

---

## 📦 Dataset Information

**File:** `imports-from-african-countries.csv`

**Size:** 139,566 transactions

**Features:**
- **Original (15 columns):** id, date, country_name, alpha_3_code, region, sub_region, hs_code, commodity, unit, value_qt, value_rs, value_dl
- **Engineered (10 features):** year, month, quarter, country_total_imports, country_market_share, frequency encodings, interaction features

**Time Range:** 2015-01-01 to 2025-03-01 (10+ years)

**Coverage:**
- 59 African countries
- 4,835 unique commodities
- ₹267.66 Million total trade value

---

## 🤖 Machine Learning Models

### Regression Models (11)
1. **Random Forest** 🏆 - R² = 0.7979 (BEST)
2. Gradient Boosting - R² = 0.7909
3. Decision Tree - R² = 0.6607
4. Keras DNN - R² = 0.5245
5. PyTorch NN - R² = 0.4153
6. AdaBoost - R² = 0.2518
7. Ridge Regression - R² = 0.0219
8. Linear Regression - R² = 0.0219
9. ElasticNet - R² = 0.0053
10. Lasso - R² = 0.0020
11. K-Neighbors - R² = -0.1648

### Classification Models (11)
1. **Bagging Classifier** 🏆 - Accuracy = 70.80% (BEST)
2. Random Forest - Accuracy = 70.59%
3. Extra Trees - Accuracy = 67.80%
4. Decision Tree - Accuracy = 67.38%
5. Gradient Boosting - Accuracy = 66.21%
6. Keras DNN - Accuracy = 57.74%
7. AdaBoost - Accuracy = 56.74%
8. PyTorch NN - Accuracy = 53.90%
9. K-Neighbors - Accuracy = 52.54%
10. Logistic Regression - Accuracy = 39.72%
11. Naive Bayes - Accuracy = 35.68%

### Clustering Models (10)
1. **Mini-Batch K-Means** 🏆 - Silhouette = 0.4332 (BEST)
2. Mean Shift - Silhouette = 0.4299
3. DBSCAN - Silhouette = 0.4254
4. Agglomerative - Silhouette = 0.4249
5. K-Means - Silhouette = 0.3466
6. BIRCH - Silhouette = 0.3346
7. Affinity Propagation - Silhouette = 0.3298
8. OPTICS - Silhouette = 0.3219
9. Gaussian Mixture - Silhouette = 0.3465
10. Spectral Clustering - Silhouette = -0.3087

### Deep Learning Models (4)
- Keras DNN (Regression) - TensorFlow 2.18
- Keras DNN (Classification) - TensorFlow 2.18
- PyTorch NN (Regression) - PyTorch 2.6 + CUDA
- PyTorch NN (Classification) - PyTorch 2.6 + CUDA

**Total:** 32 Models ✅

---

## 📈 Key Insights

### 🏆 Top Performers
- **Best Country:** South Africa (₹58.1M, +58.4% growth)
- **Best Commodity:** Petroleum Oils (₹71.2M, 26.6% of total)
- **Peak Season:** March (₹24.4M)
- **Best Model:** Random Forest Regression (R²=0.7979)

### ⚠️ Critical Findings
- **Nigeria:** 70.3% petroleum dependency (CRITICAL RISK)
- **Market Concentration:** Top 5 countries = 61.5% of imports
- **Commodity Risk:** Top 20 = 86.7% of trade value
- **COVID Impact:** -15.3% decline in 2020

### ✅ Quality Metrics
- Zero duplicate rows (0.00%)
- Minimal missing values (4 values)
- No data leakage (verified)
- Realistic model performance

---

## 💻 Technical Details

### Dependencies
- **Python:** 3.8+
- **Streamlit:** 1.28.0+
- **Pandas:** 2.0.0+
- **NumPy:** 1.24.0+
- **Plotly:** 5.17.0+
- **Scikit-learn:** 1.3.0+
- **Matplotlib, Seaborn**

### Performance
- **Load Time:** < 3 seconds
- **Data Processing:** Cached with `@st.cache_data`
- **Model Loading:** Cached with `@st.cache_resource`
- **Interactive:** Real-time chart updates

### Features
- Responsive layout (wide mode)
- Interactive Plotly charts
- Custom CSS styling
- Tab-based navigation
- Gradient metric cards
- Insight boxes with color coding

---

## 🎯 Business Applications

### 1. Forecasting & Planning
- **Model:** Random Forest Regression (R²=0.80)
- **Use:** Predict future import values, budget allocation

### 2. Risk Assessment
- **Model:** Bagging Classifier (70.80% accuracy)
- **Use:** Classify transaction risk, fraud detection

### 3. Market Segmentation
- **Model:** Mini-Batch K-Means (Silhouette=0.43)
- **Use:** Identify patterns, market opportunities

### 4. Country Analysis
- **Source:** EDA Insights
- **Use:** Monitor trends, identify growth markets

---

## 🚀 Future Enhancements

### Model Improvements
- Time series forecasting (ARIMA, Prophet, LSTM)
- Hyperparameter tuning (Grid Search, Bayesian)
- Advanced models (XGBoost, LightGBM, CatBoost)

### Deployment
- RESTful API (FastAPI/Flask)
- Cloud deployment (AWS/GCP)
- Docker containerization
- Real-time monitoring

### Business Expansion
- Live dashboards
- Automated reporting
- ERP/CRM integration
- Mobile app

---

## 📝 Notes

### Data Leakage Fix
The project initially had data leakage issues with 99.99% accuracy. This was fixed by:
1. Removing `log_value_rs` (derived from target)
2. Removing `price_per_unit` (calculated from target)
3. Removing `rolling_3m_avg` (used future data)
4. Removing `commodity_avg_price` (from target)

All 32 models were retrained with clean data, resulting in realistic metrics.

### Model Files
All models are saved in `african_imports_models_only_20251102_175254/`:
- Traditional ML: `.pkl` files (pickle)
- Keras/TensorFlow: `.h5` files
- PyTorch: `.pt` files (state dict)

**Total Size:** ~346 MB

---

## 📞 Support

For questions or issues:
1. Check the inline code comments
2. Review the project summary document
3. Examine the Jupyter notebook: `Import-ananlysis.ipynb`

---

## ✅ Project Status

**Status:** ✅ COMPLETE

**Date:** November 2025

**Features:**
- ✅ Complete EDA (15+ visualizations)
- ✅ 32 ML models (exceeded goal)
- ✅ All 3 types (Regression, Classification, Clustering)
- ✅ Deep Learning (TensorFlow + PyTorch)
- ✅ Feature Engineering (26 created, 10 selected)
- ✅ Data leakage fixed
- ✅ Production-ready models
- ✅ Interactive Streamlit dashboard

---

## 🎓 Project Achievements

### Goals Met
1. ✅ Complete EDA with 15+ visualizations
2. ✅ 32 models (exceeded 25-30 goal!)
3. ✅ All 3 ML types implemented
4. ✅ Deep Learning with GPU acceleration
5. ✅ Feature Engineering pipeline
6. ✅ Data quality verification
7. ✅ Production-ready deployment

### Quality Metrics
- **Code Quality:** Production-ready, well-commented
- **Data Quality:** Clean, no leakage, realistic metrics
- **Model Quality:** Properly evaluated, cross-validated
- **Documentation:** Comprehensive, clear, actionable

---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| Total Records | 139,566 |
| Countries | 59 |
| Commodities | 4,835 |
| Time Period | 2015-2025 |
| ML Models | 32 |
| Best R² Score | 0.7979 |
| Best Accuracy | 70.80% |
| Best Silhouette | 0.4332 |

---

**🌍 African Import Analysis - Complete ML Pipeline**

*Developed: November 2025*
