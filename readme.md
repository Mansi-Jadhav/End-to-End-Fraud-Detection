# End-to-End Fraud Detection
### IEEE-CIS Fraud Detection | Kaggle Competition

A complete machine learning pipeline for credit card fraud detection, built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset. Achieved **0.892 AUC-ROC** on the Kaggle public leaderboard.

---

## Project Structure
├── 1. EDA.ipynb # Exploratory data analysis
├── 2. Feature Engineering.ipynb # Feature pipeline and transformations
├── 3. Modelling.ipynb # Baseline models, tuning, evaluation
├── 4. SHAP Explainability.ipynb # Model explainability
├── 5. Test Predictions for Kaggle.ipynb # Submission generation
├── transformers.py # Custom sklearn transformer classes
└── README.md

---

## Dataset

The dataset is provided by Vesta Corporation via Kaggle and contains 590,540 e-commerce transactions with 434 features across two tables (transaction and identity). The fraud rate is 3.5%.

The dataset is not included in this repository. Download from:
https://www.kaggle.com/competitions/ieee-fraud-detection/data

---

## Approach

### 1. Exploratory Data Analysis
- Class imbalance analysis (3.5% fraud rate)
- Temporal pattern analysis — fraud peaks 1–6 AM
- Card identity analysis — fraud ring detection via card1/card2
- Two types of missingness identified: informative vs structural
- V feature cluster analysis (339 anonymised Vesta features)
- Full analysis across all categorical and numerical feature families

### 2. Feature Engineering
Custom sklearn-compatible pipeline with the following transformations:

| Transformation | Features |
|---|---|
| Temporal extraction | TransactionDT → Hour, Day, Week |
| log1p | TransactionAmt, C1–C14, dist1 |
| Missing indicators | D features, dist features, id_01–id_11 |
| Binary flag | addr2 → is_dominant_region |
| Category grouping | DeviceInfo, email domains, id_30/31/33 |
| Target encoding | card1, card2, card3, card5, addr1 |
| One-hot encoding | ProductCD, card4, card6, M1–M9, id_12–id_38 |
| UID aggregations | card1 + addr1 + email → txn_count, amt_mean, amt_std |
| V feature selection | Greedy correlation-based: 339 → 139 features |

All transformers implemented as `BaseEstimator + TransformerMixin` classes, assembled into a single sklearn `Pipeline`. Fitted on training split only.

### 3. Modelling

| Model | AUC-ROC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **LightGBM (+ UID features)** | **0.8912** | **0.5755** | **0.801** | **0.429** | **0.559** |

**Final model:** LightGBM with Optuna hyperparameter tuning (30 trials, Bayesian optimisation). Class imbalance handled via scale_pos_weight (~27x).

**Kaggle public leaderboard: 0.892 AUC-ROC**

### 4. Explainability (SHAP)
- Global feature importance — card1 dominant, UID aggregations highly ranked
- Beeswarm plot — direction of feature impact across all transactions
- Dependence plots — non-linear relationships confirmed for card1, C13, TransactionAmt
- Individual waterfall explanations — fraud vs legitimate transaction comparison

---

## Key Findings

- **card1** is the single most predictive feature — specific card identifiers 
  show 40–52% fraud rates, consistent with fraud rings
- **UID aggregation** (card + address + email) was the highest-impact 
  feature engineering step (+0.016 PR-AUC, +22% precision improvement)
- **Verification bypass** (missing M features) is more suspicious than failed 
  verification — fraudsters skip checks rather than fail them
- **Two types of missingness** require different treatment: informative 
  (M/D features — above baseline fraud) vs structural (id features — below baseline)
- **Hyperparameter tuning** via Optuna identified max_depth and subsample 
  as most impactful parameters; best solution found at trial 3 of 30

---

## Technical Stack
Python 3.11
LightGBM 4.x, XGBoost, scikit-learn
pandas, numpy
SHAP
Optuna
category_encoders
joblib

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Mansi-Jadhav/End-to-End-Fraud-Detection.git

# 2. Install dependencies
pip install lightgbm xgboost scikit-learn pandas numpy shap optuna category-encoders joblib

# 3. Download data from Kaggle
# Place train_transaction.csv, train_identity.csv, 
# test_transaction.csv, test_identity.csv in data/

# 4. Run notebooks in order:
# 1. EDA.ipynb
# 2. Feature Engineering.ipynb  
# 3. Modelling.ipynb
# 4. SHAP Explainability.ipynb
# 5. Test Predictions for Kaggle.ipynb
```

---

## Author

Mansi Jadhav