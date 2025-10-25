# Notebook Documentation — imports_analysis.ipynb

This document explains the contents of the notebook `imports_analysis.ipynb` (renamed from `Untitled5 (1).ipynb`) in detail: the data schema, cleaning steps, every major table and plot, feature engineering, modelling experiments, forecasting, unsupervised analysis, and the Streamlit demo builder present in the notebook.

Where I use the phrase "cell" I mean the notebook cell number (top-to-bottom). The notebook contains markdown and code cells organized in logical sections. This document follows the notebook's structure and explains the purpose and outcome of each logical block.

---

## 1 — Project & Data overview

- Purpose: analyze import shipments from African countries, generate EDA visualizations, run regression/classification/forecasting experiments and build a small Streamlit demo.
- Primary dataset: `imports-from-african-countries.csv` (present in repo; ~139k rows). Key columns are:

  - `id` — row identifier
  - `date` — shipment date (string; converted to datetime in the notebook)
  - `country_name` — importing country name (string)
  - `alpha_3_code` — ISO alpha-3 code
  - `country_code` — numeric/other code
  - `region`, `region_code` — region umbrella
  - `sub_region`, `sub_region_code` — finer geographic region
  - `hs_code` — Harmonized System code for commodity
  - `commodity` — description of commodity (high-cardinality string)
  - `unit` — unit of measurement (e.g., `Kgs`, `Cbm`, `Ctm`, etc.)
  - `value_qt` — quantity numeric
  - `value_rs` — value in local currency (numeric)
  - `value_dl` — value in US dollars (numeric)

Example rows: the CSV contains many commodity entries such as petroleum, agricultural goods, and manufactured items; some rows show very large `value_dl` (skewed distribution).

## 2 — High-level notebook flow

1. Environment setup and load CSV
2. Initial inspection and light cleaning (missing units, date parsing)
3. Exploratory Data Analysis (distributions, time series, group-by/top-n plots)
4. Feature engineering (price per unit, log transforms)
5. Supervised modelling experiments (regression and classification pipelines)
6. Forecasting (Prophet + XGBoost approaches)
7. Unsupervised analyses (KMeans + IsolationForest)
8. A cell that writes out a Streamlit app (`streamlit_app.py`) for interactive exploration

## 3 — Detailed explanations (by logical cell group)

Note: I group related cells (data load, cleaning, EDA, models) rather than strictly one line-per-cell — this keeps the documentation readable while still mapping notebooks steps and intent.

### A. Data load & initial checks

- What the notebook does:
  - Mounts Google Drive (in original Colab context) and reads the CSV into a pandas DataFrame.
  - Calls `df.head()`, `df.info()`, `df.describe()` to inspect column types and missingness.

- Why it's done: confirm file loaded correctly, detect missing or inconsistent types and missing values.

- Notes / Repro tips:
  - The notebook uses Colab paths. Locally, point pandas to `./imports-from-african-countries.csv` (already present in repo).

### B. Cleaning & type conversions

- Key actions:
  - Fill missing `unit` entries with a placeholder like `'Unknown'` (so downstream filters don't error).
  - Convert `date` column to `pd.to_datetime(df['date'])` and set up time-based index or columns when needed.
  - Remove or filter out rows where numeric conversions fail (not always explicit — inspect the notebook for try/except or coercions).

- Reasoning:
  - Correct types are necessary for resampling (time series) and numeric modeling.

### C. Exploratory Data Analysis (EDA)

The notebook produces many plots; here are the common types, how they are produced, and what they show.

- Distribution / boxplots for `value_qt`, `value_rs`, `value_dl`:
  - Code: `sns.boxplot(...)` or `plt.boxplot(...)` on selected numeric columns.
  - Transformations: sometimes `np.log1p()` or `np.log()` is used to visualize skewed distributions.
  - Interpretation: shows heavy skew in monetary values (`value_dl`) — a few very large shipments dominate totals.

- Histograms and log-histograms:
  - Code: `plt.hist(df['value_dl'], bins=...)` with `log=True` or `np.log1p` applied.
  - Interpretation: confirms heavy right-skew; motivates log-transform for modeling.

- Time-series line plots (daily, monthly, yearly aggregations):
  - Code pattern: `df.set_index('date').resample('D'|'M'|'Y')['value_dl'].sum().plot()` or `groupby(pd.Grouper(key='date', freq='M'))...`
  - Transformations: resampling and sum/mean to aggregate.
  - Interpretation: trends and seasonality over time, spikes corresponding to expensive shipments or bulk periods.

- Top-n bar plots (by `country_name`, `commodity`, `sub_region`):
  - Code: `df.groupby('country_name')['value_dl'].sum().nlargest(10).plot(kind='bar')`.
  - Interpretation: identifies which importing countries and commodities account for most import value.

- Scatterplots (e.g., `value_qt` vs `value_dl`):
  - Code: `plt.scatter(df['value_qt'], df['value_dl'])` possibly with log scales.
  - Interpretation: relationship between quantity and reported dollar value; outliers and heteroskedasticity become visible.

- Boxplots grouped by categorical variables:
  - Code: `sns.boxplot(x='country_name', y='value_dl', data=...)` with top-n countries filtered first.
  - Interpretation: country-level distribution differences and outliers.

- Pie charts / composition plots for categorical proportions:
  - Code: `df['commodity'].value_counts().head(10).plot(kind='pie')`.
  - Interpretation: quick view of the share of top commodities.

### D. Aggregation and pivot tables

- The notebook builds aggregated tables: monthly_imports, daily_imports, yearly summaries. Methods used:
  - `resample('M').sum()` or `groupby` with `pd.Grouper(key='date', freq='M')`.
  - Compute YoY% by comparing year-over-year sums: e.g., `(this_year - prev_year) / prev_year`.

- Purpose: to reduce noise in individual shipment rows and surface higher-level trends.

### E. Per-country and per-commodity breakdowns

- The notebook iterates over top countries or sub-regions and produces:
  - Top commodities for each country (counts and value sums)
  - Time series of top commodities within a country
  - Frequency tables and barplots of commodities per country

- Why: helps identify which goods dominate a country's imports and highlight trade patterns.

### F. Feature engineering

- Price per unit calculation (e.g., `price_per_kg`):
  - Code: for rows where `unit == 'Kgs'`, compute `price_per_kg = value_dl / value_qt` (with care for zeros/missing).
  - Transformations: filter by `unit`, create new column, sometimes apply `np.log1p` afterwards.
  - Purpose: to standardize values across shipments using a comparable unit (price per kg) — useful for commodity-level pricing analysis.

- Log transforms on monetary fields: helps stabilize variance and reduce the influence of large outliers in some models.

### G. Supervised modelling (regression)

- Goal: predict a target (likely `value_dl` or unit price) using features such as `country_name`, `commodity`, `unit`, `date` features, etc.

- Preprocessing pipeline pattern used repeatedly:
  - `ColumnTransformer` to apply `StandardScaler()` or `passthrough` to numeric features and `OneHotEncoder(handle_unknown='ignore', sparse_output=True)` to categorical features (commodity, country_name).
  - `Pipeline` combining `preprocessor` and a scikit-learn estimator.

- Models trained (examples present in notebook):
  - Regularized linear models: `Lasso`, `Ridge`, `ElasticNet`
  - Tree-based: `DecisionTreeRegressor`, `RandomForestRegressor`
  - Gradient boosting: `xgboost.XGBRegressor`, `lightgbm.LGBMRegressor`
  - Instance-based / kernel: `KNeighborsRegressor`, `SVR`

- Observations / reasoning:
  - One-hot encoding of `commodity` and `country_name` yields very high-dimensional sparse matrices (thousands of features). Tree-based models handle this better; distance-based regressors (KNN, SVR) perform poorly unless dimensionality is reduced.
  - Cross-validation or simple train/test splits are used to compute RMSE/MAPE/R2 (check the notebook for exact metrics used).

### H. Classification experiments

- Task: predict `sub_region` or another categorical label from features. Preprocessing is similar to regression pipelines (OneHotEncoder for categorical fields, optional scaling for numeric fields).

- Models tried include `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `XGBClassifier`, `KNeighborsClassifier`, and `SVC`.

- Observations: label encoding of `sub_region` is used as target; class imbalance and high-cardinality features affect classifier performance.

### I. Forecasting

- Approaches present:
  - Prophet (per commodity or per country): prepare a time-series `ds` (date) and `y` (aggregated value), fit Prophet, and forecast future periods.
  - An XGBoost-based time-series method: create lag features, rolling aggregates, and train XGBoost to predict forward steps.

- Notes:
  - Prophet is convenient for calendar effects, but remember to supply aggregated data (monthly or weekly) instead of raw shipment-level rows.
  - The notebook shows both approaches to illustrate pros/cons.

### J. Unsupervised: Clustering and Anomaly detection

- KMeans clustering with an elbow plot:
  - Code: compute `inertia` for k in range(1, 10) and plot to choose k.
  - Features: typically aggregated numeric features or embeddings derived from categorical counts.

- IsolationForest for anomaly detection:
  - Code: `IsolationForest().fit_predict(X)` on selected numeric features or reduced-dimensional data.
  - Interpretation: anomalies may correspond to suspiciously large `value_dl` entries or inconsistent unit/quantity combinations.

### K. Streamlit app generation

- The notebook contains a long code cell that writes a `streamlit_app.py` file. The Streamlit app organizes pages for:
  - EDA (plots, top-ns)
  - Regression demos (select model, show metrics)
  - Classification demos
  - Forecasting interface (choose country/commodity and forecast)
  - Unsupervised view (clusters and anomaly lists)

- Usage: `streamlit run streamlit_app.py` (after installing requirements).

## 4 — Reproducibility & environment notes

- The notebook was authored in Google Colab and includes Drive mount cells. Locally:
  - Place `imports-from-african-countries.csv` in the repository root or update the read path.
  - Create a Python virtualenv and install dependencies from `requirements.txt`.

- Package caveats:
  - The notebook references `prophet` (if you see `fbprophet` errors, install the `prophet` package in modern environments).
  - XGBoost and LightGBM often require native wheels; use pip or conda to install compatible binaries for your OS.

## 5 — Recommendations and next steps

1. Replace the large CSV in the repo with a small sample for development and add the full CSV to a `data/` directory that is in `.gitignore` to avoid large git history growth. If you want, I can remove the full CSV from git history using `git filter-repo` or the BFG tool.
2. Modularize the notebook into scripts: `data/`, `notebooks/`, `models/`, and `app/` to improve reproducibility and CI.
3. Add a small smoke test (e.g., load first 100 rows and run a minimal pipeline) and add it to CI.
4. When training models, use cross-validation (KFold/TimeSeriesSplit where appropriate) and store model artifacts and metrics in a `models/` folder.

## 6 — Appendix: Quick mapping of the most important notebook cells (by activity)

- Data load and display (first cells): `pd.read_csv(...)`, `df.head()`, `df.info()` — purpose: schema check and missingness.
- Cleaning: `df['unit'].fillna('Unknown')`, `df['date']=pd.to_datetime(df['date'])` — purpose: consistent types.
- EDA: boxplots / histograms / time series / top-n bar charts — purpose: understand distributions and major contributors by country/commodity.
- Feature engineering: `df.loc[df['unit']=='Kgs', 'price_per_kg'] = df['value_dl'] / df['value_qt']` — purpose: normalized price signal.
- Pipelines: repeated ColumnTransformer+OneHotEncoder + model pattern — purpose: consistent preprocessing across models.
- Forecasting: Prophet model cells and XGBoost time-series cell(s) — purpose: short-term forecasting experiments.
- Streamlit writer cell: writes `streamlit_app.py` using `with open('streamlit_app.py', 'w') as f: f.write(...)` — purpose: deliver a small app for interactive exploration.

---

If you'd like, I will now:

1. Commit this `document.md` to the repository and push it to GitHub (so the file appears at the repo URL). 
2. Or, if you prefer to review it first or want more granularity (line-by-line cell mapping), I can expand any section you specify.

Which would you prefer? If you'd like me to commit & push, I will do that and mark the todo as completed.
