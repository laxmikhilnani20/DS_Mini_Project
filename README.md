# 🌍 DS Mini Project: African Countries Import Analytics

> **Interactive ML-powered dashboard for exploring and predicting import trade patterns from African countries**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ds-mini-project.streamlit.app)

**🚀 Live Demo:** [https://ds-mini-project.streamlit.app](https://ds-mini-project.streamlit.app)

---

## 📖 Overview

This project provides a comprehensive data analysis and machine learning platform for exploring import transaction data from African countries. It combines exploratory data analysis (EDA), advanced visualizations, and production-ready ML models in an interactive Streamlit dashboard.

### ✨ Key Features

#### 📊 **Interactive Data Explorer**
- **Dynamic filtering** by country, commodity, date range, and transaction value
- **Smart currency formatting** (Billions/Millions/Thousands)
- **Real-time statistics** with responsive visualizations
- **Comprehensive EDA** with 15+ visualization types

#### 🤖 **Advanced Machine Learning Models**

1. **📈 Regression Models** - Predict Future Import Values
   - Multiple algorithms: Linear Regression, Random Forest, XGBoost, LightGBM
   - Feature engineering with lag features and time-based patterns
   - **5-10 Year Time Series Forecasting** with confidence intervals
   - Model performance comparison with R², MAE, RMSE metrics

2. **🎯 Classification Models** - Import Trend Prediction
   - Predicts if trade is **Growing** (>10% YoY), **Stable** (-10% to +10%), or **Declining** (<-10%)
   - Logistic Regression, Random Forest, and XGBoost classifiers
   - Confusion matrix visualization
   - Feature importance analysis

3. **🔍 Clustering Models** - Co-Occurrence Analysis
   - Discovers which **commodities are imported together**
   - Uses cosine similarity for market basket analysis
   - K-Means clustering with PCA visualization
   - Elbow method for optimal cluster selection
   - Business insights for bundle procurement strategies

---

## 📁 Repository Contents

```
DS/
├── streamlit_app.py              # Main interactive dashboard application
├── imports-from-african-countries.csv  # Dataset (~139k transactions)
├── requirements.txt              # Python dependencies
├── Untitled5 (1).ipynb          # Original analysis notebook
├── .streamlit/
│   └── config.toml              # Streamlit configuration
└── README.md                    # This file
```

---

## 🗃️ Dataset Information

**Source:** Import transaction records from African countries  
**Size:** ~139,000 transactions  
**Time Period:** Multi-year import data  
**Countries:** 59 unique African nations  
**Commodities:** 123+ different commodity types

**Key Columns:**
- `date` - Transaction date
- `country_name` - Importing country
- `commodity` - Product/commodity name
- `value_dl` - Transaction value (USD)
- `value_qt` - Quantity imported
- `unit` - Unit of measurement

---

## 🚀 Quick Start

### Option 1: Use the Live App (Recommended)
**Visit:** [https://ds-mini-project.streamlit.app](https://ds-mini-project.streamlit.app)

No installation needed! Explore the data and run ML models directly in your browser.

### Option 2: Run Locally

#### Prerequisites
- Python 3.8 or higher
- pip package manager

#### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/laxmikhilnani20/DS_Mini_Project.git
cd DS_Mini_Project
```

2. **Create a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**
```bash
streamlit run streamlit_app.py
```

5. **Open in browser**
- The app will automatically open at `http://localhost:8501`

---

## 🛠️ Technology Stack

### **Frontend & Visualization**
- **Streamlit** - Interactive web application framework
- **Plotly** - Interactive plotting and visualizations
- **Matplotlib & Seaborn** - Statistical graphics

### **Data Processing**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **SciPy** - Scientific computing

### **Machine Learning**
- **scikit-learn** - ML algorithms and preprocessing
- **XGBoost** - Gradient boosting framework
- **LightGBM** - Fast gradient boosting
- **Prophet** - Time series forecasting

---

## 📊 Dashboard Sections

### 1. 📈 Data Overview
- Summary statistics and key metrics
- Dataset description and data quality insights
- Currency-formatted displays (B/M/K notation)

### 2. 🔍 EDA Explorer
- **15+ Interactive Visualizations:**
  - Time series analysis (monthly, yearly, seasonal)
  - Top countries and commodities
  - Distribution analysis (boxplots, histograms)
  - Quantity vs. Value scatter plots
  - Moving averages and cumulative trends
  - Year-over-year growth analysis
  - Quarterly patterns

### 3. 🤖 Interactive Models
- **Regression Tab**: Predict import values with 5-10 year forecasts
- **Classification Tab**: Identify growing/stable/declining trade relationships
- **Clustering Tab**: Discover commodity co-occurrence patterns

---

## 📈 Key Insights & Use Cases

### Business Applications
- **Strategic Procurement**: Identify commodities frequently imported together for bundle deals
- **Market Intelligence**: Detect emerging (growing) and declining trade relationships
- **Budget Planning**: 5-10 year import value forecasts for financial planning
- **Risk Management**: Understand trade volatility and seasonal patterns
- **Supplier Negotiation**: Data-driven insights on import trends and pricing

### Data Science Applications
- Feature engineering techniques for time series data
- Handling high-cardinality categorical features
- Multi-model comparison frameworks
- Co-occurrence analysis using cosine similarity
- Time series forecasting with confidence intervals

---

## 🔄 Recent Updates

### Latest Enhancements
- ✅ **Import Trend Classification** - Predict Growing/Stable/Declining patterns
- ✅ **5-10 Year Time Series Forecasting** - Long-term predictions with confidence intervals
- ✅ **Co-Occurrence Clustering** - Market basket analysis for commodities
- ✅ **Smart Currency Formatting** - Automatic B/M/K notation
- ✅ **Dynamic Filtering** - All 59 countries and 123 commodities selectable
- ✅ **Infinity/NaN Handling** - Robust data preprocessing for edge cases

---

## 📝 Notes & Considerations

- **Dataset Size**: The CSV file (~139k rows) is included in the repository for reproducibility
- **Model Performance**: Tree-based models (Random Forest, XGBoost) perform best due to high-cardinality categorical features
- **Scalability**: Models train on filtered data; full dataset training may require additional computational resources
- **Time Series**: Forecasting uses rolling predictions with lag features for realistic future projections

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests for improvements
- Share feedback on the ML models

---

## 📄 License

This project is for educational and analytical purposes. Add a license file (e.g., MIT) if you wish to publish.

---

## 📧 Contact

**Project Repository:** [https://github.com/laxmikhilnani20/DS_Mini_Project](https://github.com/laxmikhilnani20/DS_Mini_Project)  
**Live Application:** [https://ds-mini-project.streamlit.app](https://ds-mini-project.streamlit.app)

---

**⭐ If you find this project useful, please consider starring the repository!**
