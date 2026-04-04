# 💳 Financial Fraud Detection

> End-to-end machine learning pipeline for detecting fraudulent financial transactions using XGBoost, LightGBM, and SHAP explainability.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Steps](#pipeline-steps)
- [Results](#results)
- [Key Features](#key-features)
- [Notebooks](#notebooks)
- [Usage](#usage)
- [Contributing](#contributing)

---

## Overview

This project builds a production-grade **fraud detection system** on 1 million synthetic financial transactions. It covers the full data science lifecycle:

1. **Exploratory Data Analysis** — distributions, temporal patterns, fraud rates by segment
2. **Feature Engineering** — ratio features, time flags, log transforms, behavioral signals
3. **Model Training** — Logistic Regression baseline, XGBoost, and LightGBM with class-imbalance handling
4. **Evaluation** — AUC-ROC, Average Precision, F1, Confusion Matrix, PR curves
5. **Explainability** — SHAP beeswarm, bar importance, and dependence plots
6. **Inference** — `FraudDetector` class for scoring new transactions at serving time

---

## Dataset

| Property | Value |
|---|---|
| Total transactions | 1,000,000 |
| Fraud rate | ~9.17% |
| Organizations | 200 |
| Users | 20,000 |
| Features | 24 raw + 8 engineered |
| Target | `is_fraud` (binary) |

**Key feature groups:**
- Transaction details: `transaction_amount`, `fee_amount`, `transaction_type`
- User profile: `user_account_age_days`, `kyc_verified`
- Device & channel: `device_type`, `operating_system`, `browser`, `payment_method`
- Temporal: `hour`, `day_of_week`, `is_night`, `time_diff`
- Security signals: `otp_used`
- Merchant: `merchant_category`

> ⚠️ This is a **synthetic dataset** for educational purposes. No real user or financial data is included.

---

## Project Structure

```
fraud_detection/
│
├── data/
│   └── improved_fraud_dataset.csv      ← Place dataset here
│
├── notebooks/
│   ├── 01_EDA.ipynb                    ← Exploratory Data Analysis
│   └── 02_Modeling.ipynb               ← Training, evaluation, SHAP
│
├── src/
│   ├── data_preprocessing.py           ← Load, clean, feature engineer
│   ├── eda.py                          ← All visualization functions
│   ├── model_training.py               ← Train + evaluate models
│   ├── explainability.py               ← SHAP analysis
│   └── inference.py                    ← FraudDetector inference class
│
├── models/                             ← Saved .pkl model files (git-ignored)
│
├── reports/
│   └── figures/                        ← Auto-generated plots (git-ignored)
│
├── tests/
│   └── test_preprocessing.py           ← Unit tests (pytest)
│
├── main.py                             ← Full pipeline CLI entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone & set up

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Add the dataset

```bash
# Place the CSV in the data directory
cp /path/to/improved_fraud_dataset.csv data/
```

### 3. Run the full pipeline

```bash
python main.py
```

This runs EDA → preprocessing → training → SHAP and saves all outputs.

**Optional flags:**

```bash
python main.py --skip-eda          # Skip EDA plots (faster iteration)
python main.py --skip-shap         # Skip SHAP (saves ~5 min on 1M rows)
python main.py --model XGBoost     # Choose model for SHAP analysis
python main.py --data /other.csv   # Use a different dataset
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Pipeline Steps

### Step 1 – EDA (`src/eda.py`)

Generates 7 publication-quality plots saved to `reports/figures/`:

| Plot | Description |
|---|---|
| `01_class_distribution` | Bar + pie chart of fraud vs. legitimate |
| `02_amount_distribution` | Amount histograms (raw + log scale) by label |
| `03_fraud_rate_by_category` | Fraud rate per category for 7 features |
| `04_temporal_patterns` | Fraud rate by hour of day and day of week |
| `05_correlation_heatmap` | Correlation matrix of numeric features |
| `06_security_signals` | KYC and OTP fraud rate comparison |
| `07_account_age_analysis` | Fraud rate bucketed by account age |

### Step 2 – Feature Engineering (`src/data_preprocessing.py`)

| Feature | Formula | Signal |
|---|---|---|
| `fee_ratio` | `fee / (amount + ε)` | Abnormal fee structures |
| `amount_per_acc_age` | `amount / (acc_age + 1)` | Large tx from new accounts |
| `is_weekend` | `day_of_week ∈ {Sat, Sun}` | Weekend fraud spike |
| `is_high_risk_hour` | `hour ∈ [0, 5]` | Late-night activity |
| `log_amount` | `log1p(amount)` | Normalise right skew |
| `log_time_diff` | `log1p(time_diff)` | Normalise time gaps |
| `no_otp_high_amount` | `otp=0 AND amount > 75th pct` | No auth on big tx |
| `unverified_new_user` | `kyc=0 AND age < 90d` | Unverified new accounts |

### Step 3 – Model Training (`src/model_training.py`)

All models use **5-fold stratified cross-validation** and `scale_pos_weight` for class imbalance.

| Model | Notes |
|---|---|
| `LogisticRegression` | Baseline, `class_weight='balanced'` |
| `XGBoost` | 300 estimators, depth=6, `scale_pos_weight` |
| `LightGBM` | 300 estimators, depth=6, fastest on 1M rows |

### Step 4 – Explainability (`src/explainability.py`)

SHAP TreeExplainer generates:
- **Beeswarm plot** — per-sample feature contributions
- **Bar chart** — mean absolute SHAP (global importance)
- **Dependence plot** — SHAP vs. raw feature value for `log_amount`

---

## Results

> Results will vary slightly due to random seeds. Expected approximate values:

| Model | CV AUC-ROC | Test AUC-ROC | Avg Precision | F1-Score |
|---|---|---|---|---|
| Logistic Regression | ~0.78 | ~0.79 | ~0.45 | ~0.52 |
| XGBoost | ~0.93 | ~0.94 | ~0.78 | ~0.73 |
| **LightGBM** | **~0.94** | **~0.95** | **~0.80** | **~0.74** |

LightGBM is the best-performing and fastest model, recommended for production.

---

## Key Features

- ✅ **Handles class imbalance** via `scale_pos_weight` and stratified splits
- ✅ **Reproducible** — fixed random seeds throughout
- ✅ **Production-ready inference** — `FraudDetector` class with `.predict()` and `.predict_single()`
- ✅ **Modular codebase** — each step is independently importable
- ✅ **SHAP explainability** — understand why each transaction is flagged
- ✅ **Unit-tested** — 15+ tests covering feature engineering and encoding
- ✅ **CLI pipeline** — single `python main.py` command runs everything

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_EDA.ipynb` | Interactive exploration of distributions, correlations, and fraud patterns |
| `02_Modeling.ipynb` | Step-by-step training, evaluation plots, SHAP, and inference demo |

To start Jupyter:
```bash
jupyter notebook notebooks/
```

---

## Usage

### Score new transactions programmatically

```python
from src.inference import FraudDetector
import pandas as pd

# Load pre-trained detector
detector = FraudDetector.load("fraud_detector", directory="models/")

# Score a batch
df = pd.read_csv("new_transactions.csv")
predictions = detector.predict(df, threshold=0.5)
print(predictions.head())
# transaction_id  fraud_score  is_fraud_predicted
#     TXN00001        0.0231               0
#     TXN00002        0.8743               1
```

### Score a single transaction

```python
result = detector.predict_single({
    "transaction_amount": 15000,
    "payment_method": "card",
    "card_type": "credit",
    "kyc_verified": 0,
    "otp_used": 0,
    "user_account_age_days": 30,
    "hour": 2,
    "day_of_week": "Sunday",
    "device_type": "mobile",
    "operating_system": "Android",
    "browser": "Chrome",
    "merchant_category": "electronics",
    "transaction_type": "purchase",
    "fee_amount": 300,
    "time_diff": 50000,
    "is_night": 1,
})
print(result)
# {'transaction_id': 0, 'fraud_score': 0.89, 'is_fraud': True}
```

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Commit changes: `git commit -m "Add my improvement"`
4. Push: `git push origin feature/my-improvement`
5. Open a Pull Request

Please run `pytest tests/ -v` before submitting.

---

## License

MIT License – see [LICENSE](LICENSE) for details.
