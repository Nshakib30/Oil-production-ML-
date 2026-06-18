# Volve Oil Recovery Predictor

**Live app:** [petroml.streamlit.app](https://petroml.streamlit.app/)

An interactive machine learning demo that predicts daily oil production from well operating parameters, using a tuned XGBoost model with SHAP-based explainability. Built as the applied companion to a research paper on explainable AI for petroleum reservoir engineering.

## What this app does

Enter a well's operating conditions (pressure, choke size, flow hours, water volume) and the app predicts expected daily oil output in barrels. Alongside the prediction, a SHAP breakdown shows which inputs pushed that specific prediction up or down, so the result isn't just a number, it comes with a reason.

## Background

Reservoir simulation is accurate but slow and data-hungry. Machine learning models can approximate recovery behavior in a fraction of the time, but most published ML studies in this space stop at reporting accuracy metrics, without explaining which parameters the model actually relied on or whether those parameters make physical sense. This project addresses that gap directly.

## Data

[Equinor Volve field dataset](https://www.equinor.com/energy/volve-data-sharing) — daily production data from a real, decommissioned North Sea oil field (2008–2016), released by Equinor for open research use.

## Methodology summary

- **Cleaning:** removed injector wells and shut-in (non-producing) days, fixed an impossible negative water volume reading, and identified and corrected a permanent downhole pressure gauge failure affecting two wells (one fully, one partially) by treating the affected readings as missing and filling them per well.
- **Feature engineering:** added `PRESSURE_DRAWDOWN` (downhole pressure minus wellhead pressure), the actual physical driver of inflow, after confirming raw downhole pressure alone gave a misleading correlation with output.
- **Leakage check:** an initially engineered `WATER_CUT` feature was found to algebraically contain the target variable and was removed before modeling.
- **Models compared:** Linear Regression, SVR, Random Forest, ANN, and XGBoost, evaluated with R², RMSE, and MAE on a held-out test set.
- **Tuning:** XGBoost hyperparameters tuned via 5-fold GridSearchCV.
- **Explainability:** SHAP (TreeExplainer) applied to the tuned model to produce a global feature importance ranking and per-prediction explanations, since native importance metrics (gain, weight, cover) gave inconsistent rankings on this dataset.

## Model performance (test set)

| Model | R² | RMSE (bbl) | MAE (bbl) |
|---|---|---|---|
| Linear Regression | 0.518 | 173.94 | 134.66 |
| SVR | 0.737 | 128.39 | 85.91 |
| ANN | 0.810 | 109.39 | 71.95 |
| Random Forest | 0.937 | 62.81 | 30.19 |
| **XGBoost (tuned)** | **0.945** | **58.78** | **30.94** |

## Key finding

`PRESSURE_DRAWDOWN` ranked as the single most important feature by SHAP, despite showing almost no linear correlation with oil output in standard EDA (Pearson r ≈ 0.04). The relationship is real but nonlinear and choke-gated, which is exactly the kind of pattern linear correlation analysis misses and tree-based ML with SHAP can recover.

## Limitation

Diagnostic analysis showed `PRESSURE_DRAWDOWN`'s importance is partially confounded with well identity: high-drawdown values cluster heavily in two specific wells, one of which had the documented gauge failure mentioned above. This is disclosed here and in the paper rather than glossed over; well-stratified modeling is a natural next step to disentangle the two effects.

## Tech stack

Python, scikit-learn, XGBoost, SHAP, Streamlit, Plotly.

## Run locally

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py
```

Requires `model.pkl`, `scaler.pkl`, and `feature_columns.pkl` in the same directory (included in this repo).

## Repository structure

```
.
├── app.py                  # Streamlit app
├── requirements.txt
├── model.pkl                # Tuned XGBoost model
├── scaler.pkl                # MinMaxScaler fit on training data
├── feature_columns.pkl       # Exact feature order expected by the model
├── petroleum_ml_pipeline.py  # Full training pipeline (data cleaning to SHAP)
└── figures/                  # EDA, model comparison, and SHAP figures from the paper
```

## Disclaimer

This is a research demo accompanying an academic paper and is not a production forecasting tool. Predictions are based on a single field's historical data and should not be used for operational decision-making.

## Acknowledgments

Built on the Equinor Volve open dataset, released for research and educational use.
