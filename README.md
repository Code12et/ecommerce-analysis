# ecommerce_analysis

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>



--------

# Olist Brazilian E-Commerce Analysis

End-to-end data science project analysing 100k+ real orders from Brazil's 
largest e-commerce platform. Covers data cleaning, EDA, KPI dashboards, 
and machine learning to predict bad customer reviews.

---

## Business Problems Solved

| Problem | Approach | Result |
|---|---|---|
| Why do customers leave bad reviews? | ML classification | delivery delay is #1 predictor |
| Which sellers underperform? | KPI dashboard | bottom 10 sellers identified |
| Where do deliveries fail? | Geospatial EDA | RR and AP states worst OTIF |
| What drives late deliveries? | Feature analysis | freight cost and seller history |

---

## Key Findings

- **OTIF Rate: ~92%** — 8% of delivered orders arrived late
- **Bad Review Rate: 12.8%** — strongly correlated with late delivery
- **Avg review score drops from 4.3 → 2.1** when order is late
- **delivery_delay_days** is the #1 predictor of bad reviews (RF importance: 14.8%)
- **seller_avg_score** is #2 — a seller's history predicts future complaints
- **São Paulo** accounts for ~42% of all customers

---

## Project Structure

ecommerce_analysis/
├── data/
│   ├── raw/          ← original CSVs (not committed)
│   ├── interim/      ← olist_master.parquet (96,999 rows)
│   └── processed/    ← olist_features.parquet (38 features)
├── notebooks/
│   ├── 0.1-data-audit.ipynb
│   ├── 1.0-eda-orders.ipynb
│   ├── 1.1-eda-delivery.ipynb
│   ├── 1.2-eda-reviews.ipynb
│   ├── 1.3-eda-payments.ipynb
│   ├── 1.4-eda-geolocation.ipynb
│   ├── 2.0-cleaning-master.ipynb
│   ├── 3.0-feature-engineering.ipynb
│   ├── 4.0-kpi-dashboard.ipynb
│   └── 5.0-model-bad-review.ipynb
├── olist/
│   ├── data.py       ← cleaning and merging functions
│   ├── features.py   ← feature engineering
│   └── config.py     ← paths and settings
├── tests/
│   └── test_data.py  ← 10 unit tests (all passing)
└── reports/
└── figures/      ← all saved charts


---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ecommerce_analysis.git
cd ecommerce_analysis

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -e .
pip install pandas numpy matplotlib seaborn scikit-learn pyarrow jupyter loguru python-dotenv tqdm

# 4. Download dataset from Kaggle
kaggle datasets download olistbr/brazilian-ecommerce -p data/raw --unzip

# 5. Run notebooks in order
jupyter notebook
```

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

[Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) 
— 9 relational tables, 100k+ orders, 2016–2018.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.4-orange)