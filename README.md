# DS Mini Project: Imports from African Countries

This repository contains an exploratory data analysis, modelling experiments, forecasting work and a small Streamlit demo built on an imports dataset for African countries.

Contents
- `Untitled5 (1).ipynb` — Main analysis notebook (EDA, feature engineering, modelling, forecasting, clustering, anomaly detection, and a cell that writes a Streamlit app).
- `imports-from-african-countries.csv` — Raw dataset used by the notebook (rows: ~139k). This file is large and was present in the repository at the time of committing.
- `requirements.txt` — Python package requirements for reproducing the analysis.
- `.gitignore` — Recommended ignores for Python/data projects.

Project summary
---------------
The notebook walks through:

- Data loading and cleaning (fill missing units, parse `date` column)
- Exploratory plots: time series, boxplots, histograms, bar charts of top countries/commodities, scatterplots and price calculations
- Feature engineering: creating `price_per_kg` (where `unit == 'Kgs'`), log transforms, and aggregation by time windows (daily/monthly/yearly)
- Supervised learning experiments: regression models (Ridge, Lasso, ElasticNet, Decision Tree, RandomForest, XGBoost, LightGBM, KNN, SVR) using ColumnTransformer pipelines with OneHot encoding for categorical features
- Classification experiments predicting `sub_region`
- Forecasting: Prophet and an XGBoost time-series approach
- Unsupervised analysis: KMeans clustering (elbow plot) and IsolationForest anomaly detection
- A Streamlit app builder cell that writes `streamlit_app.py` for quick interactive exploration and demo.

How to reproduce (local or virtual environment)
----------------------------------------------
1. Create a Python virtual environment and activate it. Example using venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook

- Open `Untitled5 (1).ipynb` in JupyterLab / Jupyter Notebook / VS Code and execute cells in order. The notebook was originally authored for Google Colab (it contains Drive mount cells). If running locally, update the CSV path or move the CSV into the repository root and re-run from the top.

4. Run the Streamlit demo (optional):

```bash
streamlit run streamlit_app.py
```

Notes and caveats
-----------------
- The CSV is large (~139k rows). If you prefer not to store large data in the repo, move `imports-from-african-countries.csv` to a separate `data/` folder and add it to `.gitignore`. If you want, I can help remove it from git history (that is a separate cleanup step).
- The notebook heavily one-hot encodes high-cardinality categorical fields (e.g., `commodity`) which produces many sparse features; tree-based models (RandomForest, XGBoost, LightGBM) are a good fit here — distance-based models like KNN/SVR may perform poorly unless features are reduced or embedded.
- The forecasting cells call `prophet` (the modern package is `prophet`; older code may reference `fbprophet`). If you run into import errors, install the package named `prophet`.

Suggested next steps
--------------------
- Clean and modularize the notebook into smaller scripts (data/ preprocessing/ models/) and add a reproducible pipeline.
- Add lightweight unit tests or smoke tests for data-loading and a small sample run of the modelling pipelines.
- Consider removing the large CSV from the repository and replacing it with a small sample CSV for CI/tests.

License & contact
-----------------
This repository contains analysis code — add a license file if you want to publish (e.g., MIT). If you'd like, I can add a `LICENSE` file.

If you want me to add a short `README` badge, a license, or to remove the large CSV from git history, tell me and I’ll proceed.
