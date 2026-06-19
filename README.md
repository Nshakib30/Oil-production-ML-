# Volve Oil Recovery Predictor

**Live demo:** [petrum.streamlit.app](https://petrum.streamlit.app/)

## What is this?

A tool that predicts how much oil a well will produce each day, based on real-time operating conditions. You plug in pressure, choke size, flow hours, and water volume — it tells you the expected barrel rate, plus a breakdown of which factors pushed that prediction up or down.

## Why does it matter?

Most ML models stop at giving you a number. This one explains *why* it predicted that number using SHAP, so you can actually trust and understand the forecast, not just guess.

## How it works

1. **Trained on real data** — 8 years of daily production from the Equinor Volve North Sea field (publicly available data)
2. **Five models tested** — Linear Regression, SVR, Random Forest, ANN, XGBoost. XGBoost won with 94.5% accuracy
3. **Every prediction explained** — SHAP shows which inputs drove each forecast

## Key insight

Pressure difference (drawdown) turned out to be the most important factor, even though it barely correlated with output in basic stats. The relationship is real but hidden in nonlinear patterns that only ML can catch.

## Run it yourself

```bash
pip install -r requirements.txt
streamlit run app.py
```

Needs `model.pkl`, `scaler.pkl`, and `feature_columns.pkl` (all included).

## Note

This is a research demo, not a production tool. It's meant to show how ML + explainability can work together in reservoir engineering.
