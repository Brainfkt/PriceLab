# PriceLab

PriceLab is a local pricing intelligence simulator for a product catalogue. It turns sales, prices, stock, promotions, and context history into guarded product-level price recommendations with estimated impact on volume, revenue, margin, risk, and confidence.

The goal is portfolio-grade data science, not a descriptive dashboard. The app includes schema validation, data quality scanning, feature engineering, temporal backtesting, interpretable elasticity, a challenger ML model, reliability scoring, scenario simulation, optimization, explainability, and product report export.

## Links

- Cloudflare Pages redirect: `https://pricelab-71i.pages.dev`
- Streamlit app: `https://pricelab.streamlit.app`
- Deployment guide: [DEPLOYMENT.md](DEPLOYMENT.md)

The Cloudflare Pages URL publishes the static files in `docs/` and redirects visitors to the Streamlit app configured in `docs/config.js`. The GitHub repository URL itself remains `https://github.com/Brainfkt/PriceLab`.

## MVP Features

- Streamlit app with generated demo data when no CSV is uploaded.
- CSV import with column mapping and schema validation.
- Data quality scanner for missing values, invalid prices, cost inconsistencies, stockouts, promo contamination, temporal gaps, outliers, and weak price variation.
- Auto grain detection: daily inputs are aggregated to weekly modelling frames, while weekly inputs stay native.
- Portfolio and catalogue visual analytics with filters, Top N/All leaderboards, Pareto contribution, revenue-vs-margin scatterplots, category mix, and reliability health views.
- Product deep dive with enriched price/unit history, promotion and stock markers, adaptive temporal heatmaps, price performance matrix, best moments, and promotion lift charts.
- Log-log Ridge elasticity model per product.
- Segment elasticity by category and season when enough data is available.
- Random Forest challenger demand model with temporal backtesting against a baseline using wMAPE, SMAPE, MAE, RMSE, and R2.
- Reliability score with hard blocks, strengths, warnings, data-quality and demand-stability components.
- Contextual scenario simulator with price, discount, channel, region, month, stock, uncertainty bands, observed-range markers, and scenario curves for volume, revenue, and margin.
- Optimal price finder for volume, revenue, margin, or prudence.
- Catalogue opportunity scanner with action categories and a reliability-vs-impact opportunity matrix.
- Markdown and enriched HTML product report export with context metadata and embedded Plotly charts.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Run

```bash
streamlit run app.py
```

The app uses a generated synthetic dataset by default. Upload a CSV in the sidebar to use your own data.

## Validate

```bash
pytest -q
python -m pricelab.cli generate-demo --out data/demo/pricelab_demo.csv
python -m pricelab.cli validate data/demo/pricelab_demo.csv
```

## Deploy

PriceLab is a Python/Streamlit app, so the interactive application should be deployed on Streamlit Community Cloud or another Python host. Cloudflare Pages publishes the static portfolio page in `docs/` and redirects visitors to the Streamlit app.

Minimum Streamlit Cloud settings:

- Repository: `Brainfkt/PriceLab`
- Branch: `main`
- Main file path: `app.py`
- Dependencies: `requirements.txt`

Cloudflare Pages settings: project `pricelab`, branch `main`, root `/`, empty build command, output directory `docs`.

## Expected CSV Schema

Required columns:

- `date`
- `product_id`
- `product_name`
- `category`
- `units_sold`
- `price`
- `cost`
- `stock_available`
- `promotion_flag`

Optional columns:

- `discount_rate`
- `channel`
- `region`
- `competitor_price`
- `marketing_spend`
- `traffic`
- `holiday_flag`
- `customer_segment`
- `returns`
- `weather_index`

The model frame supports product x date x channel x region. Daily inputs are aggregated before modeling; weekly inputs stay at native grain.

## Reliability Rules

Recommendations are blocked or degraded when the history is short, price variation is too weak, price is almost constant, stockouts are too frequent, promotions dominate the sample, backtesting is poor, the scenario extrapolates too far, or margin optimization is requested without cost data.

Score levels:

- `>=75`: normal recommendation.
- `55-74`: cautious recommendation.
- `35-54`: simulation only.
- `<35`: blocked.

## Statistical Limits

PriceLab estimates conditional price response from observational data. It does not prove causal elasticity. Price endogeneity, promotional confounding, censored demand during stockouts, sparse histories, and channel or regional mix effects are surfaced as warnings in the reliability panel and product report.
