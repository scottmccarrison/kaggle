# Kaggle & Data Science Projects

End-to-end machine learning projects built to demonstrate the full 
data science workflow: EDA → modeling → evaluation → explainability.

Background: Data Engineer at an electric utility in Pittsburgh, PA.
Currently pursuing an MS in Data Science.
Goal: Become a full-stack Data Scientist — scope, build, deploy, and 
maintain ML solutions end-to-end.

---

## Completed Projects

### [GreenGrid: Household Energy Consumption Forecasting](./greengrid-energy-forecast/)
https://www.kaggle.com/competitions/green-grid-forecasting-daily-household-energy-consumption
<br>
Predicting daily household electricity consumption for 1,500 households 
over a 14-day window using XGBoost and SHAP explainability.  
**Validation RMSLE: 0.1559** | 59% improvement over baseline  
`XGBoost` `SHAP` `Time Series` `Panel Data` `Feature Engineering`

---

## In Progress

### [House Prices: Advanced Regression](./house-prices-advanced-regression/)
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
<br>
Cross-sectional tabular regression on the Ames, Iowa housing dataset 
(79 features, 1,460 rows). Predicting `SalePrice` with RMSLE metric.  
Planned approach: XGBoost baseline → LightGBM comparison → domain-informed 
feature engineering → SHAP explainability.  
`XGBoost` `LightGBM` `SHAP` `Cross-Sectional Regression` `Categorical Encoding`

---

## Planned — Problem-Type Tour

Building baseline competency across the major ML problem types before 
going deeper on any single one. Each project introduces one new skill 
dimension.

### [Titanic: Machine Learning from Disaster](./titanic/)
https://www.kaggle.com/competitions/titanic
<br>
Binary classification — predict passenger survival (1/0) from tabular 
features. First classification project.  
**New skills:** Classification metrics (precision, recall, F1, ROC-AUC), 
confusion matrix, `predict_proba`, threshold tuning.  
`Classification` `Binary` `Tabular` `Logistic Regression` `XGBoost`

### [Digit Recognizer](./digit-recognizer/)
https://www.kaggle.com/competitions/digit-recognizer
<br>
Multiclass classification — classify handwritten digits (0–9) from 28×28 
pixel images. First deep learning project.  
**New skills:** Image data handling, CNN/MLP in PyTorch or Keras, 
softmax, categorical cross-entropy.  
`Deep Learning` `CNN` `Image Classification` `Multiclass` `PyTorch`

### [NLP Getting Started: Disaster Tweets](./disaster-tweets/)
https://www.kaggle.com/competitions/nlp-getting-started
<br>
Binary text classification — predict whether a tweet is about a real 
disaster. First NLP project.  
**New skills:** Text preprocessing, TF-IDF, word embeddings, 
unstructured data handling.  
`NLP` `Text Classification` `TF-IDF` `Embeddings` `BERT`

### [Credit Card Fraud Detection](./credit-card-fraud/)
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
<br>
Anomaly detection / imbalanced classification — identify fraudulent 
transactions (~0.17% of data).  
**New skills:** Class imbalance handling, SMOTE/undersampling, 
precision-recall tradeoff, PR-AUC, anomaly detection algorithms.  
`Anomaly Detection` `Imbalanced Data` `SMOTE` `Isolation Forest`

---

## Planned — Applied Projects

Real-world datasets and end-to-end deployments that go beyond Kaggle 
competition structure.

### Pittsburgh Home Sale Propensity Model
Predict which houses in the Pittsburgh area are most likely to list 
for sale, using Allegheny County property assessment and sales records 
from the WPRDC.

Binary classification problem — for each property that hasn't recently 
sold, predict whether it will list within a given time window (e.g., 
6 months). Features would include:

- **Owner-level:** ownership duration (strongest known predictor), 
  owner type (absentee/primary/investor), equity position estimate
- **Property-level:** age, square footage, condition, recent renovations
- **Spatial:** number of homes sold within 0.25-mile radius in the last 
  6 months, median sale price of nearby recent sales, distance to 
  nearest recent sale
- **Market:** local inventory levels, median days-on-market, seasonality

This project combines several skills from the problem-type tour: 
binary classification, spatial feature engineering, and imbalanced 
data (most homes don't list in any given period). Requires acquiring 
and joining real-world data sources rather than using a pre-packaged 
Kaggle dataset.

**Prerequisite projects:** Titanic (classification metrics), 
Credit Card Fraud (class imbalance), House Prices (real estate 
domain knowledge).  
**Data source:** WPRDC (Western Pennsylvania Regional Data Center) — 
Allegheny County property assessment and sales records.  
**New skills:** Spatial feature engineering, real-world data acquisition 
and joining, survival analysis (time-to-event modeling), working with 
public government datasets.  
`Classification` `Binary` `Spatial Data` `Real Estate` `Survival Analysis` `WPRDC` `Pittsburgh`

### [Pittsburgh Housing Market Model](./pittsburgh-housing-model/)
End-to-end deployment project — apply regression skills to local 
Pittsburgh housing data with full MLOps pipeline.  
**New skills:** FastAPI, Docker, AWS deployment, MLflow, model monitoring.  
`FastAPI` `Docker` `AWS` `MLflow` `Deployment`

---

## Problem-Type Coverage Matrix

| Project | Data Type | Problem Type | Output | Key New Skill | Status |
|---|---|---|---|---|---|
| GreenGrid | Tabular, temporal | Panel time series forecasting | Continuous (kWh) | Temporal split, log target | ✅ |
| House Prices | Tabular | Cross-sectional regression | Continuous (price) | Categorical encoding, meaningful NaNs | 🔄 |
| Titanic | Tabular | Binary classification | Label (0/1) | Classification metrics | ⬜ |
| Digit Recognizer | Image | Multiclass classification | Label (0–9) | Deep learning, CNN | ⬜ |
| Disaster Tweets | Text | Binary text classification | Label (0/1) | NLP, TF-IDF, embeddings | ⬜ |
| Credit Card Fraud | Tabular | Imbalanced classification | Label (0/1) | Class imbalance, anomaly detection | ⬜ |
| Pittsburgh Sale Propensity | Tabular + spatial | Binary classification + survival | Label (0/1) + time | Spatial feature engineering, real-world data, survival analysis | ⬜ |
| Pittsburgh Housing Model | Tabular | Regression + deployment | Continuous (price) | MLOps, FastAPI, Docker, AWS | ⬜ |

---

## Stack

`Python` `XGBoost` `LightGBM` `scikit-learn` `SHAP` `pandas` `numpy`  
`PyTorch` `FastAPI` `Docker` `AWS` `MLflow`  
`Snowflake` `dbt` `Databricks` `SQL` `PySpark`
