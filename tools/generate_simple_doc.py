#!/usr/bin/env python3
import csv, re, os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
mapfile = os.path.join(root, 'docs', 'images', 'image_mapping.csv')
out = os.path.join(root, 'document_simple.md')
cell_map = {
    10: ('Boxplots of numeric fields', 'Shows distributions and outliers that drive totals.'),
    14: ('Monthly total import value', 'Shows trend and seasonality over months.'),
    31: ('Daily total import value', 'Shows day-to-day variation; useful for spotting spikes.'),
    40: ('Top countries by import value', 'Shows which countries contribute most to total imports.'),
    46: ('Top commodities by import value', 'Shows which commodities drive import value.'),
    58: ('Quantity vs USD scatter', 'Reveals pricing patterns and outliers (cheap bulk vs expensive items).'),
    122: ('Yearly totals overview', 'Shows long-term totals by year.'),
    155: ('Per-country yearly series (loop)', 'Each image is a country-specific yearly totals or composition plot.'),
    156: ('YoY % change summary', 'Shows year-over-year percent change for totals.'),
    157: ('Related transform diagnostics', 'Small diagnostics for transformed features.'),
    158: ('Per-country yearly totals + YoY (loop)', 'Each image shows yearly totals and YoY% for a country.'),
    160: ('Country contribution barplots', 'Shows share contributions per country or sub-region.'),
    161: ('Country contribution pie charts', 'Pie charts of contribution by segment.'),
    162: ('Top-10 commodity pie charts per country (loop)', 'Shows commodity composition for each country.'),
    169: ('Model diagnostics (actual vs predicted)', 'Helps assess model fit and residual behavior.'),
    172: ('Decision tree visualization', 'Small tree for interpretability.'),
    174: ('Random Forest visuals', 'Feature importance and predictions.'),
    178: ('XGBoost visuals', 'Feature importance and predictions.'),
    179: ('LightGBM visuals', 'Feature importance and predictions.'),
    182: ('Forecast plots', 'Historical plus forecast and intervals.'),
    190: ('KMeans elbow & clusters', 'Shows elbow method and cluster visualization.'),
    194: ('Anomaly detection (IsolationForest)', 'Highlights unusual shipments to inspect.'),
    92: ('Per-commodity summaries', 'Multi-panel commodity-level plots.'),
    77: ('EDA grid plots', 'Small grid of related exploratory plots.'),
}

rows = []
with open(mapfile, newline='', encoding='utf-8') as fh:
    reader = csv.DictReader(fh)
    for r in reader:
        old = r['old'].strip()
        new = r['new'].strip()
        m = re.match(r'plot_cell_(\d+)_', old)
        cell = int(m.group(1)) if m else None
        rows.append((new, old, cell))

rows.sort()

lines = []
lines.append('# Simple figure guide — plain English')
lines.append('A very simple, beginner-friendly caption for every extracted figure. Images are in `docs/images/`.')
lines.append('')
lines.append('For each figure:')
lines.append('- What it is: very short description')
lines.append('- Why it matters: one-sentence practical takeaway')
lines.append('')

for new, old, cell in rows:
    desc, why = (f'Plot produced by notebook cell {cell}', 'Open the notebook at that cell to see the code and printed context.')
    if cell in cell_map:
        desc, why = cell_map[cell]
    lines.append('---')
    lines.append(f'## {new}')
    lines.append('')
    lines.append(f'![{new}](docs/images/{new})')
    lines.append('')
    lines.append(f'**What it is:** {desc}.')
    lines.append('')
    lines.append(f'**Why it matters:** {why}.')
    lines.append('')

with open(out, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines))

print('Wrote', out)
