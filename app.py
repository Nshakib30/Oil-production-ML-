import io
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

st.set_page_config(
    page_title="Volve Oil Recovery Predictor",
    layout="wide",
)

st.markdown(
    """
    <style>
    .header-title { font-size: 2.1rem; font-weight: 700; color: #9DC4F0; margin-bottom: 0.1rem; }
    .header-sub { font-size: 1rem; color: #B7C3D1; margin-top: 0; margin-bottom: 1.2rem; }
    .section-heading { font-size: 1.25rem; font-weight: 600; color: #EDF1F5; margin-bottom: 0.5rem; }
    .stButton>button {
        background-color: #5B9BD5; color: #0F1620; border-radius: 6px;
        font-weight: 600; border: none; padding: 0.6rem 1rem;
    }
    .stButton>button:hover { background-color: #7FB3E8; color: #0F1620; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


model, scaler, FEATURE_COLS = load_artifacts()
explainer = get_explainer(model)

WELLS = [
    "NO 15/9-F-1 C", "NO 15/9-F-11 H", "NO 15/9-F-12 H",
    "NO 15/9-F-14 H", "NO 15/9-F-15 D", "NO 15/9-F-5 AH",
]

st.markdown('<p class="header-title">Volve Oil Recovery Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="header-sub">XGBoost regression with SHAP explainability — '
    'Equinor Volve North Sea field dataset (open access)</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Well Operating Conditions")
    well = st.selectbox("Well", WELLS)

    st.subheader("Pressure")
    downhole_p = st.number_input("Downhole Pressure (psi)", 0.0, 400.0, 250.0, step=5.0)
    whp        = st.number_input("Wellhead Pressure, WHP (psi)", 0.0, 140.0, 40.0, step=1.0)
    wht        = st.number_input("Wellhead Temperature, WHT (°C)", 0.0, 100.0, 70.0, step=1.0)
    annulus    = st.number_input("Annulus Pressure (psi)", 0.0, 30.0, 15.0, step=0.5)
    gauge_valid = st.checkbox("Downhole gauge reading valid", value=True)

    st.subheader("Flow")
    on_stream = st.slider("On-Stream Hours", 0.0, 24.0, 24.0)
    choke     = st.number_input("Choke Size (%)", 0.0, 100.0, 50.0, step=1.0)
    wat_vol   = st.number_input("Water Volume (bbl)", 0.0, 1000.0, 50.0, step=10.0)

    predict_clicked = st.button("Predict Oil Recovery", use_container_width=True)

# ---------------------------------------------------------------
# Session state
# ---------------------------------------------------------------
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
    st.session_state.last_shap       = None
    st.session_state.last_X_scaled   = None
    st.session_state.last_explainer  = None

if predict_clicked:
    row = {c: 0 for c in FEATURE_COLS}
    row["ON_STREAM_HRS"]       = on_stream
    row["AVG_ANNULUS_PRESS"]   = annulus
    row["AVG_CHOKE_SIZE_P"]    = choke
    row["AVG_WHP_P"]           = whp
    row["AVG_WHT_P"]           = wht
    row["BORE_WAT_VOL"]        = wat_vol
    row["PRESSURE_DRAWDOWN"]   = downhole_p - whp
    row["DOWNHOLE_GAUGE_VALID"] = int(gauge_valid)
    row[f"WELL_{well}"]        = 1

    X_input  = pd.DataFrame([row])[FEATURE_COLS]
    X_scaled = scaler.transform(X_input)
    pred     = model.predict(X_scaled)[0]
    shap_row = explainer.shap_values(X_scaled)[0]

    st.session_state.last_prediction = pred
    st.session_state.last_shap       = pd.Series(shap_row, index=FEATURE_COLS)
    st.session_state.last_X_scaled   = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
    st.session_state.last_explainer  = explainer

# ---------------------------------------------------------------
# Top row — gauge + SHAP bar
# ---------------------------------------------------------------
col1, col2 = st.columns([1, 1.4])

with col1:
    st.markdown('<p class="section-heading">Predicted Oil Volume</p>', unsafe_allow_html=True)
    pred = st.session_state.last_prediction

    if pred is None:
        st.info("Set parameters in the sidebar and click Predict.")
        pred = 0

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        number={"suffix": " bbl/day", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 1000]},
            "bar": {"color": "#5B9BD5"},
            "steps": [
                {"range": [0,   250], "color": "#3A1A1A"},
                {"range": [250, 600], "color": "#2A2A1A"},
                {"range": [600, 1000], "color": "#1A2E1A"},
            ],
            "threshold": {
                "line": {"color": "#EDF1F5", "width": 2},
                "thickness": 0.75,
                "value": pred,
            },
        },
    ))
    gauge.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#EDF1F5",
        margin=dict(t=20, b=10, l=20, r=20),
    )
    st.plotly_chart(gauge, use_container_width=True)

with col2:
    st.markdown('<p class="section-heading">SHAP Analysis — Feature Impact</p>', unsafe_allow_html=True)
    if st.session_state.last_shap is not None:
        contrib = st.session_state.last_shap.sort_values()
        colors  = ["#BF616A" if v < 0 else "#5B9BD5" for v in contrib.values]
        shap_fig = go.Figure(go.Bar(
            x=contrib.values, y=contrib.index,
            orientation="h", marker_color=colors,
        ))
        shap_fig.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#EDF1F5",
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="Impact on predicted oil volume (bbl/day)",
            xaxis=dict(gridcolor="#2A3A4A"),
        )
        st.plotly_chart(shap_fig, use_container_width=True)
        st.caption("Blue = pushes prediction up. Red = pushes prediction down.")
    else:
        st.info("Run a prediction to see the SHAP breakdown.")

# ---------------------------------------------------------------
# SHAP Waterfall + Summary plots
# ---------------------------------------------------------------
if st.session_state.last_shap is not None:
    st.markdown("---")
    st.markdown('<p class="section-heading">SHAP Analysis — Detailed Plots</p>', unsafe_allow_html=True)
    wf_col, sum_col = st.columns(2)

    with wf_col:
        st.caption("Waterfall plot — how each feature moved this prediction from the baseline")
        try:
            exp = st.session_state.last_explainer(st.session_state.last_X_scaled)
            fig_wf, ax = plt.subplots(figsize=(6, 5))
            fig_wf.patch.set_facecolor("#0F1620")
            ax.set_facecolor("#0F1620")
            shap.plots.waterfall(exp[0], max_display=10, show=False)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                        facecolor="#0F1620")
            plt.close()
            buf.seek(0)
            st.image(buf, use_container_width=True)
        except Exception as e:
            st.warning(f"Waterfall plot unavailable: {e}")

    with sum_col:
        st.caption("Summary plot — average SHAP importance across all features")
        try:
            shap_vals = st.session_state.last_shap.sort_values(key=abs, ascending=True)
            fig_sum, ax = plt.subplots(figsize=(6, 5))
            fig_sum.patch.set_facecolor("#0F1620")
            ax.set_facecolor("#0F1620")
            bars = ax.barh(shap_vals.index, shap_vals.abs().values,
                           color=["#BF616A" if v < 0 else "#5B9BD5"
                                  for v in shap_vals.values])
            ax.set_xlabel("|SHAP value| — mean absolute impact", color="#EDF1F5")
            ax.tick_params(colors="#EDF1F5")
            ax.spines["bottom"].set_color("#3A4A5A")
            ax.spines["left"].set_color("#3A4A5A")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            buf2 = io.BytesIO()
            plt.savefig(buf2, format="png", dpi=150, bbox_inches="tight",
                        facecolor="#0F1620")
            plt.close()
            buf2.seek(0)
            st.image(buf2, use_container_width=True)
        except Exception as e:
            st.warning(f"Summary plot unavailable: {e}")

# ---------------------------------------------------------------
# Model performance footer
# ---------------------------------------------------------------
st.markdown("---")
st.markdown('<p class="section-heading">Model Performance (Held-Out Test Set)</p>',
            unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("R² (Accuracy)", "0.945")
m2.metric("RMSE (Avg. Error)", "58.78 bbl/day")
m3.metric("MAE (Median Error)", "30.94 bbl/day")

st.caption(
    "Tuned XGBoost model, trained on the Equinor Volve field dataset. "
    "This is a research demo — not a production forecasting tool."
)
