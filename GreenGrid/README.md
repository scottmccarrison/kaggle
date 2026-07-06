# GreenGrid: Household Energy Consumption Forecasting

Predicting daily household electricity consumption (kWh) for 1,500 households
over a 14-day window using XGBoost with SHAP explainability.

**Validation RMSLE: 0.1559** | Baseline RMSLE: 0.3866 | Improvement: 59%

---

## Problem Statement

Electric utilities need accurate short-term load forecasts to manage grid
operations, plan capacity, and identify anomalous consumption patterns.
This project builds a daily household-level energy forecasting model using
the GreenGrid Synthetic Smart-Meter Dataset — a panel dataset of 1,500
households over 74 days (60 train, 14 test).

The task is a **panel time series regression** problem: predict tomorrow's
kWh consumption for each household given historical usage, household
attributes, and weather data.

**Evaluation metric:** RMSLE (Root Mean Squared Log Error) — penalizes
errors proportionally, so a 10% error on a high-consumption household is
treated the same as a 10% error on a low-consumption household.

---

## Dataset

- **Source:** GreenGrid Synthetic Smart-Meter Dataset v1.0, 2026
- **Train:** 88,500 rows (1,500 households × 59 days)
- **Test:** 21,000 rows (1,500 households × 14 days)
- **Target:** Daily kWh consumption (range: 0.4–76.2 kWh, mean: 26.1 kWh)

### Feature Groups

| Group | Features |
|---|---|
| Household attributes (static) | `num_residents`, `home_sqft`, `has_ev`, `has_solar`, `has_pool`, `heating_type`, `hvac_age_years` |
| Weather (daily) | `temp_avg_c`, `temp_min_c`, `temp_max_c`, `humidity_pct`, `wind_kph`, `precip_mm`, `solar_index` |
| Calendar | `is_weekend`, `is_holiday` |
| Lag features | `prior_day_kwh`, `prior_week_avg_kwh` |

Data is fully synthetic with a transparent physical generation model:
`kwh ≈ base_load + heating_load + cooling_load + ev_charging + pool_pump
       + weekend/holiday bumps − solar_offset × lognormal noise`

---

## Exploratory Data Analysis

Key findings from EDA:

- **No missing values** — clean synthetic dataset, no imputation required
- **Winter-period data** — mean temperature -0.4°C (range: -18.7 to 19.4°C);
  heating load dominates, minimal cooling load
- **Lag features mirror target distribution** — `prior_week_avg_kwh` and
  `prior_day_kwh` have nearly identical mean/std to the target, indicating
  strong temporal autocorrelation
- **Target is approximately symmetric** — mean (26.1) ≈ median (25.3);
  log-transform applied for RMSLE alignment rather than skew correction
- **One string categorical** — `heating_type` (gas 39%, electric 35%,
  heat_pump 26%) requires one-hot encoding; all other features are numeric

---

## Modeling Approach

### Why XGBoost over LSTM

This dataset is structured tabular data with pre-engineered lag features.
Tree-based models consistently outperform deep learning on tabular data at
this scale (Grinsztajn et al., 2022). LSTMs require sequence reconstruction
per household; XGBoost uses the provided lag features directly.
An LSTM comparison is planned as a follow-on experiment.

### Validation Strategy

**Chronological split** — the last 14 days of training data are held out as
validation. Random splitting is inappropriate for time series because it
allows the model to learn from future data when predicting the past,
producing optimistic validation scores that do not reflect true test performance.

### Feature Engineering

Two engineered features were tested:
- `temp_range = temp_max_c − temp_min_c` — daily temperature swing
- `solar_interaction = temp_avg_c × has_solar` — explicit solar/temperature
  interaction

Both features preserved performance without degradation, confirming the
baseline tree structure already captured these relationships implicitly.
`temp_min_c` and `temp_max_c` were removed as redundant after
`temp_range` was added.

### Hyperparameter Tuning

Systematic tuning addressed overfitting identified in the baseline run
(train RMSE 0.122 vs val RMSE 0.159 at tree 300, with validation
degrading thereafter).

| Parameter | Baseline | Final | Rationale |
|---|---|---|---|
| `max_depth` | 6 | 4 | Shallower trees generalize better |
| `learning_rate` | 0.1 | 0.05 | Slower learning, more trees, better generalization |
| `n_estimators` | 500 | 2000 (ceiling) | Early stopping determines actual count |
| `reg_alpha` (L1) | 0 | 1.0 | Pushes weak feature weights toward zero |
| `reg_lambda` (L2) | 1.0 | 2.0 | Prevents single feature dominance |
| `early_stopping_rounds` | — | 25 | Auto-stops when val score plateaus |

Final model stopped at **tree 1,597** with validation score continuously
improving — no degradation pattern, confirming regularization resolved overfitting.

---

## Results

| Model | Validation RMSLE | Validation RMSE (kWh) |
|---|---|---|
| Baseline (median) | 0.3866 | — |
| XGBoost default params | 0.1592 | 4.23 |
| + depth tuning | 0.1575 | 4.20 |
| + regularization + early stopping | 0.1573 | 4.17 |
| **+ aggressive regularization (final)** | **0.1559** | **4.14** |

The final model reduces baseline error by **59%** — average prediction
error of 4.14 kWh on mean consumption of 26.1 kWh (~16% relative error).

---

## Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was used to explain model predictions
at both the population and individual household level.

### Key Finding: Gain Importance vs. SHAP Importance

Gain-based importance ranked `prior_week_avg_kwh` first at 49.5% — suggesting
the model was a near-persistence forecast. SHAP analysis revealed a different
picture after regularization:

| SHAP Rank | Feature | Interpretation |
|---|---|---|
| 1 | `has_solar` | Solar households have significantly lower net consumption |
| 2 | `home_sqft` | Larger homes drive higher heating/cooling load |
| 3 | `has_ev` | EV chargers add meaningful consumption |
| 4 | `num_residents` | More residents increase base load |
| 5 | `prior_week_avg_kwh` | Recent usage is predictive but not dominant |

All directional relationships are physically interpretable and consistent
with the known generation formula — confirming the model learned real
patterns rather than statistical artifacts.

### Individual Prediction Explanation (Waterfall Plot)

For a sample household (actual: 19.61 kWh, predicted: 21.54 kWh):

- Base value (population average): **24.86 kWh**
- Gas heating (not electric): **−0.04** (less electrical heating load)
- No EV charger: **−0.04** (no charger load)
- Below-average home size (2,256 sqft): **−0.02**
- New HVAC system (2 years): **−0.02** (high efficiency)
- 4 residents (above average): **+0.02**
- Final prediction: **21.54 kWh**

This level of explainability enables utility analysts to audit individual
household forecasts and communicate predictions to non-technical stakeholders.

---

## Repository Structure
