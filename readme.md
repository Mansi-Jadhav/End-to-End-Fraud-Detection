# End-to-End Fraud Detection System
### IEEE-CIS Fraud Detection | Kaggle Competition | 0.892 AUC-ROC

A production-grade machine learning pipeline for credit card fraud detection built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset. The project goes beyond model accuracy — it surfaces actionable business insights, quantifies operational impact, and delivers a fully serialised, deployable pipeline.

---

## Results at a Glance

| Metric | Value |
|---|---|
| Kaggle Public Leaderboard AUC-ROC | **0.892** |
| Final Model Precision | **80.1%** — 4 in 5 flagged transactions are real fraud |
| False Alarm Reduction | **69%** fewer false alarms vs baseline (4,855 → 1,516) |
| Fraud Cases Caught | **43%** recall at 0.5 threshold |
| Training Time | 62 seconds |
| Features (after selection) | 314 from an original 434 |

---

## Business Context

Credit card fraud costs the global economy billions annually. For every fraud case missed, money is lost. For every legitimate transaction incorrectly flagged, a customer is frustrated and an analyst wastes time investigating a false alarm. The core challenge is not just detection accuracy — it's striking the right balance between catching fraud and minimising disruption to legitimate customers.

This project treats fraud detection as a business problem first:
- **Precision matters** — a fraud team can only investigate so many cases per day. High precision means analysts spend time on real fraud, not noise.
- **Recall matters** — every missed fraud case is a financial loss and a potential customer liability.
- **Explainability matters** — a black-box model that nobody trusts will not be deployed, regardless of its AUC score.

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

Provided by Vesta Corporation via Kaggle. Contains 590,540 real-world e-commerce transactions with 434 features across two tables (transaction and identity). Fraud rate: **3.5%** — a severe class imbalance that makes standard accuracy meaningless as a metric.

**Not included in this repository.** Download from: https://www.kaggle.com/competitions/ieee-fraud-detection/data

---

## Key Findings

### Fraud Patterns Discovered in EDA

**Fraudsters operate at night.**
Fraud peaks between 1–6 AM when transaction volume is at its lowest. This is not coincidental — low-activity windows mean slower detection and fewer eyes on the system. Applying stricter verification or lower approval limits during these hours would reduce exposure with minimal impact on legitimate customers.

**Fraud rings reuse the same cards.**
Specific card identifiers (card1) showed fraud rates of 40–52% across hundreds of transactions. This is the signature of a fraud ring — a compromised card being cycled through many transactions before being blocked. Real-time velocity monitoring per card would intercept these rings far earlier than transaction-level rules alone.

**Fraudsters bypass verification, not fail it.**
Across all M-feature (identity match) columns, the fraud rate when verification was Missing (7.1%) was consistently higher than when it Failed (2.4%) — which in turn was higher than when it Passed (1.7%). Fraudsters skip verification checks entirely rather than attempting and failing them. This has a direct policy implication: making verification mandatory rather than optional removes a key evasion route.

**Geography is a strong signal.**
One address region (addr2=87) accounts for 99% of transactions and has a 2.3% fraud rate. All other regions show 11.7% fraud — a 5x difference. A single binary feature (is_dominant_region) captures this cleanly.

**Certain card ranges are systematically targeted.**
card2 likely represents BIN ranges (Bank Identification Numbers). Specific BIN ranges showed fraud rates of 35–41% — suggesting entire bank card portfolios are being targeted rather than individual cards. This is a network-level signal that issuers and card networks could act on directly.

---

## Approach

### 1. Exploratory Data Analysis
Full analysis across all 434 features including temporal patterns, card identity signals, verification bypass behaviour, two types of missingness, V-feature cluster structure, and geographic signals. Every feature engineering decision in notebook 2 is directly motivated by EDA findings.

### 2. Feature Engineering

Custom sklearn-compatible pipeline with 9 transformer classes:

| Transformation | Features | Rationale |
|---|---|---|
| Temporal extraction | TransactionDT → Hour, Day, Week | Fraud peaks at specific hours |
| log1p | TransactionAmt, C1–C14, dist1 | Right-skewed distributions |
| Missing indicators | D features, dist1/2, id_01–id_11 | Missingness is informative |
| Binary flag | addr2 → is_dominant_region | 5x fraud rate difference |
| Category grouping | DeviceInfo, email domains, id_30/31/33 | Reduce cardinality |
| Target encoding | card1, card2, card3, card5, addr1 | High cardinality — encode fraud history |
| One-hot encoding | ProductCD, card4, card6, M1–M9, id_12–id_38 | Low/moderate cardinality |
| UID aggregations | card1 + addr1 + email → velocity features | Largest single performance gain |
| V feature selection | Greedy correlation-based: 339 → 139 features | Remove redundant cluster members |

All transformers implemented as `BaseEstimator + TransformerMixin` classes, fitted on training data only. Pipeline serialised with joblib — applies to new data with a single `pipeline.transform()` call.

### 3. Modelling

Three baseline models evaluated, LightGBM selected as primary:

| Model | AUC-ROC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| XGBoost (baseline) | 0.8879 | 0.5154 | 0.248 | 0.695 | 0.366 |
| Random Forest (baseline) | 0.8921 | 0.4651 | 0.296 | 0.590 | 0.394 |
| LightGBM (baseline) | 0.8913 | 0.5409 | 0.346 | 0.632 | 0.447 |
| LightGBM (Optuna tuned) | 0.8921 | 0.5591 | 0.582 | 0.519 | 0.549 |
| **LightGBM (+ UID features)** | **0.8912** | **0.5755** | **0.801** | **0.429** | **0.559** |

**Why PR-AUC over AUC-ROC as the primary metric:**
With 3.5% fraud, a model predicting everything as legitimate scores 96.5% accuracy. AUC-ROC can still look strong even when the model barely detects fraud. PR-AUC specifically measures performance on the minority class — the fraud cases the business actually cares about.

**Why LightGBM:**
Handles missing values natively (no imputation needed for numeric features), efficient on high-dimensional mixed-type data, and class imbalance addressed via scale_pos_weight (~27x). Optuna identified max_depth and subsample as the most impactful parameters — best result found at trial 3 of 30.

**Operational impact of tuning:**
- False alarms fell from 4,855 to 1,516 (69% reduction)
- A fraud investigation team reviewing 1,516 cases instead of 4,855 per period is a meaningful operational difference — analyst capacity is a real constraint in fraud operations

### 4. SHAP Explainability

- **Global importance:** card1 dominates; UID aggregations confirm that user-level behaviour is more predictive than transaction-level signals alone
- **Beeswarm:** High card1 values push toward fraud; high C13 values (legitimacy count) push toward legitimate — directionally consistent with EDA
- **Dependence plots:** Non-linear relationships confirmed for card1, C13, and TransactionAmt — a model capable of capturing these is needed; linear models would fail here
- **Individual explanations:** A fraud transaction was explained by zero legitimacy counts and suspicious card identity despite having 769 uid transactions — consistent with fraud ring behaviour where history is built before executing high-value fraud

---

## Fraud Prevention Recommendations

Based on model signals and EDA findings:

**Card velocity monitoring** — Flag cards exceeding N transactions in T hours. card1 is the strongest predictor; real-time velocity checks intercept fraud rings before losses accumulate.

**Time-based risk adjustment** — Apply stricter verification 1–6 AM. Fraud peaks in low-volume windows; most legitimate customers are inactive.

**Mandatory verification completion** — Missing verification is more suspicious than failed verification. Making completion mandatory removes the primary evasion route identified in EDA.

**Device risk tiers** — Android: 3–5x baseline fraud rate. MacOS: below baseline. Step-up authentication for high-value Android transactions, reduced friction for MacOS/desktop.

**BIN-level issuer alerts** — Specific card2 BIN ranges show 35–41% fraud rates. Working with issuing banks on portfolio-level controls protects all cards in a targeted range simultaneously.

---

## Limitations and Future Work

**Limitations:**
- Static model — fraud patterns evolve; production deployment requires a retraining pipeline with drift monitoring
- 182-day window — seasonal patterns and emerging fraud techniques beyond this window are not captured
- Single model — ensembling LightGBM + XGBoost would likely push AUC above 0.92
- Threshold fixed at 0.5 — production threshold should be set against an explicit business cost function (missed fraud cost vs investigation cost)

**Future work:**
- Time-windowed velocity features (last 1hr, 6hr, 24hr per card)
- Additional UID combinations (card1 + DeviceInfo)
- Full 5-fold CV with feature engineering inside each fold
- Model stacking and blending
- Threshold optimisation against business cost function

---

## Technical Stack
Python 3.11 | LightGBM | XGBoost | scikit-learn
pandas | numpy | SHAP | Optuna | category-encoders | joblib

---

## How to Run

```bash
git clone https://github.com/Mansi-Jadhav/End-to-End-Fraud-Detection.git
pip install lightgbm xgboost scikit-learn pandas numpy shap optuna \
            category-encoders joblib matplotlib seaborn

# Download data from Kaggle and place in data/
# Run notebooks in order: 1 → 2 → 3 → 4 → 5
```

---

## Author
Mansi Jadhav