# Olist Brazilian E-Commerce Analysis

End-to-end data science project analysing **~100k real orders** from Brazil's largest e-commerce platform. Covers data cleaning, exploratory analysis, interactive dashboards, and ML classification to understand what drives bad customer reviews.

---

## Business Problems Solved

| Problem | Approach | Key Insight |
|---|---|---|
| Why do customers leave bad reviews? | Random Forest + Logistic Regression classification | Delivery delay is the #1 predictor |
| Which sellers underperform? | KPI dashboard with per-seller metrics | Bottom 10 sellers identified by complaint rate |
| Where do deliveries fail? | Geospatial EDA by customer state | RR and AP states have worst OTIF rates |
| What drives late deliveries? | Feature analysis with engineered variables | Freight cost and seller history most predictive |

---

## Key Findings

- **OTIF Rate: ~92%** — 8% of delivered orders arrived past the estimated date
- **Bad Review Rate: 12.8%** — strongly correlated with late delivery
- **Avg review score drops from 4.3 → 2.1** when an order is late
- **`delivery_delay_days`** is the #1 predictor of bad reviews (RF importance: 14.8%)
- **`seller_avg_score`** is #2 — a seller's history predicts future complaints
- **São Paulo** accounts for ~42% of all customers

---

## Project Structure

```
ecommerce_analysis/
├── data/
│   ├── raw/               ← Original CSVs (not committed)
│   ├── interim/            ← olist_master.parquet (cleaned & merged)
│   └── processed/          ← olist_features.parquet (38 engineered features)
├── notebooks/
│   ├── data-audit.ipynb
│   ├── eda-orders.ipynb
│   ├── eda-delivery.ipynb
│   ├── eda-reviews.ipynb
│   ├── eda-payments.ipynb
│   ├── eda-geolocation.ipynb
│   ├── cleaning-master.ipynb
│   ├── 3.0-feature-engineering.ipynb
│   ├── 4.0-kpi-dashboard.ipynb
│   └── 5.0-model-bad-review.ipynb
├── olist/                  ← Reusable Python package
│   ├── config.py           ← Paths and settings
│   ├── data.py             ← Data loading, cleaning, merging
│   ├── features.py         ← Feature engineering pipeline
│   ├── plots.py            ← Visualization helpers
│   ├── dataset.py          ← CLI dataset builder
│   └── modeling/
│       ├── train.py        ← Model training CLI
│       └── predict.py      ← Inference CLI
├── tests/
│   └── test_data.py        ← 10 unit tests
├── app.py                  ← Streamlit interactive dashboard
├── Makefile                <- make test, make lint, etc.
└── pyproject.toml
```

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/YOUR_USERNAME/ecommerce_analysis.git
cd ecommerce_analysis

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS / Linux

# 3. Install project (editable) + all dependencies
pip install -e .

# 4. Download dataset from Kaggle
kaggle datasets download olistbr/brazilian-ecommerce -p data/raw --unzip

# 5. Build the master dataset
python olist/dataset.py

# 6. Launch notebooks or streamlit
jupyter notebook
# — or —
streamlit run app.py
```

---

## Usage

| Command | Description |
|---|---|
| `make test` | Run unit tests |
| `make lint` | Lint with ruff |
| `make format` | Auto-format code |
| `streamlit run app.py` | Launch dashboard locally |
| `jupyter notebook` | Open analysis notebooks |
| `make dashboard` | Open deployed Streamlit dashboard |

---

## Deployed Dashboard

Access the live interactive dashboard:  
[ecommerce-analysis-n5iayas862de9z9lhkddls.streamlit.app](https://ecommerce-analysis-n5iayas862de9z9lhkddls.streamlit.app/)

---

## Model Results

| Model | ROC-AUC | Bad Review Recall |
|---|---|---|
| Logistic Regression | 0.786 | 58% |
| Random Forest | 0.774 | 28% |

**Top predictive features:**
1. `delivery_delay_days` — 14.8%
2. `seller_avg_score` — 12.7%
3. `total_freight` — 9.2%

---

## Dataset

[Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 9 relational tables, ~100k orders, 2016–2018. Contains orders, customers, sellers, products, payments, reviews, and geolocation data across all Brazilian states.

---

## Tech Stack

- **Python 3.12** — core language
- **Pandas / NumPy** — data manipulation
- **Scikit-learn** — ML classification
- **Streamlit** — interactive dashboard
- **Plotly** — visualizations
- **Jupyter** — exploratory notebooks
- **Ruff** — linting & formatting
- **pytest** — testing
- **Loguru** — logging
