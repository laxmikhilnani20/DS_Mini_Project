## Full explanation of `imports_analysis.ipynb` and extracted figures

This document is a polished, structured walkthrough of the notebook `imports_analysis.ipynb` and the visual outputs that were extracted to `docs/images/` (named `fig_001.png` … `fig_261.png`). A CSV map from the original notebook output names to these sequential names is available at `docs/images/image_mapping.csv`.

### Goals for this document
- Explain the dataset and preprocessing performed in the notebook.
- Walk through the notebook sections and the intent of each analysis block.
- Provide high-quality, human-friendly captions for each figure and deeper explanations for representative key visuals (including axis definitions, how to read the plot, and the main takeaways).

### Table of contents
1. Dataset summary
2. Notebook structure and workflow
3. Key EDA visuals (deep-dive)
4. Looped per-country and per-commodity visuals (how to interpret; mapping)
5. Modeling & diagnostics (key figures explained)
6. Forecasting and unsupervised analyses (explanations)
7. Full figure index (one-line captions) and usage notes
8. Next steps and options

### 1) Dataset summary
- Source file: `imports-from-african-countries.csv` (present in repository root)
- Rows: ~139,566 (varies slightly depending on cleaning). The file contains shipments/import rows with these important columns:
	- `date` (string/datetime) — shipping/transaction date
	- `country_name` — country importing the goods
	- `commodity` — HS-level commodity description
	- `unit` — units like 'Kgs' or other units
	- `value_qt` — quantity
	- `value_rs` — value in local currency
	- `value_dl` — value in USD (target used in many regressions)

### Preprocessing highlights in the notebook
- Parse `date` to datetime and derive `year`, `month` columns.
- Filter or normalize units; compute `price_per_kg` when `unit == 'Kgs'` as `value_dl / value_qt` (guarding divide-by-zero), and create `log1p` transforms for skewed numeric columns.
- Drop or impute rows with missing critical fields before modeling.

### 2) Notebook structure and workflow
- Setup & imports: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, prophet (some cells detect/assume Colab).
- EDA: initial boxplots, histograms, bar charts and scatterplots to understand distributions and identify outliers.
- Grouped analyses: top-N countries and top-N commodities by USD value; per-country commodity composition via pie charts.
- Feature engineering: price-per-kg (where applicable), categorical encodings (OneHot), year/month features for time-series.
- Modeling experiments: a number of regression and classification models with simple pipelines and metric reporting (R², RMSE, accuracy for classifiers). Models include RandomForest, XGBoost, LightGBM, Ridge, Lasso, KNN, SVR, and decision tree visualization.
- Forecasting: Prophet and XGBoost-based walk-forward forecasts for time-series (global and per-country).
- Unsupervised: KMeans elbow analysis and IsolationForest anomaly detection, with a printed top anomalies table.

### 3) Key EDA visuals (deep-dive)
Below are 6 representative figures with richer explanations (axis descriptions, what to look for, and interpretive notes). Tell me if you want more of these expanded.

- **fig_001.png** (cell 10): Boxplots of numeric fields (quantity, local-currency, USD)
	- X-axis: categorical grouping (field). Y-axis: numeric value (raw or log-scaled).
	- How to read: Box = IQR; whiskers show spread; outliers indicated as points. If log-scaled, central tendency is easier to read.
	- Why important: reveals skew and extreme shipments which strongly influence aggregates and model error.

- **fig_015.png** (cell 14): Monthly total import value (USD) — trend and seasonality
	- X-axis: month/time, Y-axis: total monthly USD.
	- How to read: trend shows growth/decline; spikes indicate unusual months. Can feed directly into forecasting.

- **fig_240.png** (cell 58): Scatter of quantity vs USD value
	- X-axis: `value_qt`, Y-axis: `value_dl` (often log-log scaled).
	- How to read: linear relationship indicates consistent per-unit pricing; bands/discontinuities reveal categorical pricing or unit issues.

- **fig_134.png** (cell 162): Top-10 commodity pie chart for a sample country
	- Shows share of total import value by commodity.
	- Why important: identifies dominant commodities for each country.

- **fig_193.png** (cell 169): Model diagnostics — Actual vs Predicted and residuals
	- Plots compare predicted vs actual USD values, with residual diagnostics.
	- Why important: evaluation of model fit and heteroscedasticity.

- **fig_225.png** (cell 194): IsolationForest anomaly-score visualization and top anomalies
	- Shows distribution of anomaly scores and likely anomalous rows printed in the notebook.
	- Why important: quick filter to inspect suspect shipments.

### 4) Looped per-country and per-commodity visuals
- Cells 155, 158, and 162 produce per-country visuals (yearly totals, YoY percent change, top-10 commodity pies). To find the exact country/commodity for any `fig_###`, use `docs/images/image_mapping.csv` which maps back to the original notebook output name (for example `plot_cell_158_71.png` → `fig_114.png`). The notebook prints the country name just before generating each plot in those loop cells.

Quick recipe to trace a figure to code & country:
1. Open `docs/images/image_mapping.csv` and find `fig_###` → `plot_cell_X_Y.png`.
2. Open `imports_analysis.ipynb` and search for code cell number `X` to inspect the plotting code; the loop typically has a `print(f"Processing: {country}")` which shows the country name.

### 5) Modeling & diagnostics (key figures explained)
- Typical model figures include:
	- Actual vs Predicted scatter (global and zoomed-in for lower values)
	- Residual histogram and residual vs predicted
	- Feature importance bars for tree models
	- Small decision-tree plot for interpretability

How to interpret model diagnostics
- High R² and low RMSE are desirable; residuals centered near zero indicate unbiased predictions. Look for patterns in residuals (e.g., rising variance with predicted value) which suggest heteroscedasticity.

### 6) Forecasting & unsupervised analyses
- Forecast plots (Prophet/XGBoost): show historical data, forecasted window, and intervals.
- KMeans elbow: choose k where inertia reduction plateaus.
- IsolationForest: anomaly-scores show which rows to inspect (not always errors — may be legitimate large trades).

### 7) Full figure index and usage notes
- The one-line index (all fig_001..fig_261 captions) exists in the previous draft. I kept this polished document focused on clarity and useful workflows. If you want the full one-line list re-inserted, say "append full index" and I will add it below verbatim.

### 8) Next steps (pick one)
- Expand selected figures into full paragraphs with code snippets and interpretation. Example: `fig_071, fig_114, fig_193`.
- Rename figures semantically (e.g., `158_Nigeria_yoy.png`); I can do this and update the document in-place.
- Revert names to original `plot_cell_X_Y.png` if you prefer.

If you'd like me to proceed, tell me which option and which figures to expand (or say "append full index"). I will then edit this file and commit the change.

---
Generated on 2025-10-25 — polished by scanning the notebook outputs and mapping file available in the repository.
