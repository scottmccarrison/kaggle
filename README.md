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

## Planned — Deep Dive & Deployment

### [Pittsburgh Housing Market Model](./pittsburgh-housing-model/)
### Pittsburgh Home Sale Propensity Model
Predict which houses in the Pittsburgh area are most likely to list 
for sale within a given time window (e.g., 6 months).

Binary classification problem using Allegheny County property 
assessment and sales records from the WPRDC. The model scores every 
property that hasn't recently sold, flagging those most likely to 
list next. The strongest known predictor is ownership duration — 
how long the current owner has lived there. Other signals include 
owner type (primary resident vs. absentee/investor), equity position, 
property age and condition, and spatial features (density of nearby 
recent sales, median price of those sales).

This project combines skills from multiple tour projects: binary 
classification (Titanic), class imbalance (Credit Card Fraud), and 
real estate domain knowledge (House Prices). The key challenge is 
data acquisition — joining public county records to build owner 
histories and spatial features, rather than working with a 
pre-packaged Kaggle dataset.

**Prerequisites:** Titanic, Credit Card Fraud, House Prices  
**Data source:** WPRDC — Allegheny County property assessment and 
sales records  
**New skills:** Spatial feature engineering, real-world data 
acquisition and joining, survival analysis, public government datasets  
`Classification` `Binary` `Spatial Data` `Real Estate` `Survival Analysis` `WPRDC` `Pittsburgh`

**New skills:** FastAPI, Docker, AWS deployment, MLflow, model monitoring.  
`FastAPI` `Docker` `AWS` `MLflow` `Deployment`

---

## Problem-Type Coverage Matrix

| Project | Data Type | Problem Type | Output | Key New Skill |
|---|---|---|---|---|
| GreenGrid ✅ | Tabular, temporal | Panel time series forecasting |
```
