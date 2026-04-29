import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="EEG Stroke Auxiliary Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PARAM_PATH = Path(__file__).with_name("model_parameters.json")
with PARAM_PATH.open("r", encoding="utf-8") as f:
    PARAMS = json.load(f)

B0 = float(PARAMS["intercept"])
B = PARAMS["coefficients"]
THRESHOLD = float(PARAMS["threshold"])
REF = PARAMS["reference_values"]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def predict(age: float, sex_code: int, p_tbr: float, c_dtabr: float):
    logit = (
        B0
        + B["age"] * age
        + B["sex_female"] * sex_code
        + B["P-TBR"] * p_tbr
        + B["C-DTABR"] * c_dtabr
    )
    return logit, sigmoid(logit)


def make_force_plot(age: float, sex_code: int, p_tbr: float, c_dtabr: float):
    # This is a coefficient-based force-plot style visualization.
    # It is not a strict SHAP plot without the original training background data.
    base_logit, base_prob = predict(
        REF["age"], int(REF["sex_female"]), REF["P-TBR"], REF["C-DTABR"]
    )
    rows = [
        ("Age", age, B["age"] * (age - REF["age"])),
        ("Sex", "Female" if sex_code == 1 else "Male", B["sex_female"] * (sex_code - int(REF["sex_female"]))),
        ("P-TBR", p_tbr, B["P-TBR"] * (p_tbr - REF["P-TBR"])),
        ("C-DTABR", c_dtabr, B["C-DTABR"] * (c_dtabr - REF["C-DTABR"])),
    ]
    rows = sorted(rows, key=lambda x: abs(x[2]), reverse=True)
    current = base_logit
    shapes = []
    annotations = []
    y = 0.5
    height = 0.28

    for name, value, contribution in rows:
        x0 = current
        x1 = current + contribution
        color = "#ff0f5f" if contribution >= 0 else "#1e88ff"
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=min(x0, x1),
                x1=max(x0, x1),
                y0=y - height,
                y1=y + height,
                fillcolor=color,
                opacity=0.88,
                line=dict(width=0, color=color),
            )
        )
        label = f"{name} = {value}; {'+' if contribution >= 0 else ''}{contribution:.3f}"
        annotations.append(
            dict(
                x=(x0 + x1) / 2,
                y=0.12 if contribution >= 0 else 0.88,
                text=label,
                showarrow=False,
                font=dict(size=13, color=color),
                xanchor="center",
            )
        )
        current = x1

    final_logit = current
    final_prob = sigmoid(final_logit)
    x_min = min(base_logit, final_logit, *[s["x0"] for s in shapes], *[s["x1"] for s in shapes]) - 0.5
    x_max = max(base_logit, final_logit, *[s["x0"] for s in shapes], *[s["x1"] for s in shapes]) + 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[x_min, x_max], y=[0.5, 0.5], mode="lines", line=dict(color="#4b5563", width=1)))
    fig.add_vline(x=base_logit, line_width=1, line_dash="dot", line_color="#9ca3af")
    fig.add_vline(x=final_logit, line_width=2, line_color="#111827")
    fig.update_layout(
        title=dict(text=f"Probability of stroke status: {final_prob*100:.2f}%", x=0.5, font=dict(size=22, color="#111827")),
        shapes=shapes,
        annotations=annotations + [
            dict(x=base_logit, y=0.67, text="base value", showarrow=False, font=dict(size=12, color="#6b7280")),
            dict(x=final_logit, y=0.67, text=f"f(x)<br>{final_prob:.2f}", showarrow=False, font=dict(size=14, color="#111827")),
            dict(x=x_max, y=0.92, text="higher", showarrow=False, font=dict(size=13, color="#ff0f5f")),
            dict(x=x_max, y=0.82, text="lower", showarrow=False, font=dict(size=13, color="#1e88ff")),
        ],
        height=330,
        margin=dict(l=20, r=20, t=70, b=20),
        xaxis=dict(title="Logit contribution", showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1]),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig, rows, base_prob, final_prob


CUSTOM_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.6rem; max-width: 1550px;}
.main-title-box {
    background: #3178d4;
    border-radius: 6px;
    padding: 18px 24px;
    margin-bottom: 14px;
    text-align: center;
    color: white;
    font-weight: 800;
    font-size: 25px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.sub-note {
    font-size: 14px;
    color: #4b5563;
    line-height: 1.7;
    padding: 8px 2px 18px 2px;
}
.input-panel {
    border: 1px solid #d9dee8;
    border-radius: 8px;
    padding: 16px 16px 6px 16px;
    background: #ffffff;
    margin-bottom: 14px;
}
.result-note {
    background: #eaf4ff;
    border-radius: 8px;
    padding: 15px 16px;
    color: #075a9c;
    font-size: 15px;
    margin-bottom: 14px;
}
.warning-note {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 8px;
    padding: 12px 14px;
    color: #9a3412;
    font-size: 14px;
    line-height: 1.6;
    margin-top: 8px;
}
.stButton>button {
    background: #0d86df;
    color: white;
    border: 0;
    border-radius: 6px;
    height: 42px;
    font-weight: 700;
}
.stButton>button:hover {background: #0873c5; color:white; border:0;}
[data-testid="stMetricValue"] {font-size: 2.4rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="main-title-box">Clinical Decision Support for Stroke Auxiliary Recognition Based on EEG Power Ratio</div>',
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age", min_value=18.0, max_value=110.0, value=66.0, step=1.0, format="%.2f")
    with c2:
        sex_label = st.selectbox("Sex", options=["Male", "Female"], index=0)
    with c3:
        p_tbr = st.number_input("P-TBR", min_value=0.0, max_value=20.0, value=1.705, step=0.01, format="%.3f")
    with c4:
        c_dtabr = st.number_input("C-DTABR", min_value=0.0, max_value=20.0, value=1.170, step=0.01, format="%.3f")

    _, mid, _ = st.columns([1.3, 1, 1.3])
    with mid:
        start = st.button("Start prediction", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

sex_code = 1 if sex_label == "Female" else 0
logit, prob = predict(age, sex_code, p_tbr, c_dtabr)
result_text = "exceeds the stroke auxiliary recognition threshold" if prob >= THRESHOLD else "does not exceed the stroke auxiliary recognition threshold"

with st.expander("Current input", expanded=True):
    current_input = pd.DataFrame(
        [{"Age": age, "Sex": sex_label, "Sex code": sex_code, "P-TBR": p_tbr, "C-DTABR": c_dtabr}]
    )
    st.dataframe(current_input, hide_index=True, use_container_width=True)

with st.expander("Predict result", expanded=True):
    st.markdown(
        f'<div class="result-note">Predict probability segmentation threshold is <b>{THRESHOLD*100:.2f}%</b>.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<h3 style='text-align:center; font-family: Georgia, serif; font-weight:500;'>Probability of stroke status: {prob*100:.2f}%</h3>",
        unsafe_allow_html=True,
    )
    fig, contribution_rows, base_prob, final_prob = make_force_plot(age, sex_code, p_tbr, c_dtabr)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f"<div class='result-note'>Model decision: current probability {prob*100:.2f}% {result_text}.</div>",
        unsafe_allow_html=True,
    )

with st.expander("Model formula and contribution details", expanded=False):
    st.markdown(
        """
**Fixed Logistic model**

`logit(p) = -2.094 - 0.012 × Age + 0.332 × Sex code + 1.092 × P-TBR + 0.897 × C-DTABR`

`p = 1 / [1 + exp(-logit(p))]`

Sex coding: Male = 0, Female = 1.
        """
    )
    details = pd.DataFrame(
        [
            {"Feature": name, "Input value": value, "Logit contribution relative to reference": round(contribution, 4)}
            for name, value, contribution in contribution_rows
        ]
    )
    st.dataframe(details, hide_index=True, use_container_width=True)

st.markdown(
    """
<div class="warning-note">
<b>Clinical note:</b> This web tool is intended for research demonstration and clinical auxiliary reference only. 
It cannot replace head CT/MRI or specialist diagnosis. The force-plot style chart is generated from the fixed Logistic regression coefficients and reference values; strict SHAP analysis requires the original training dataset.
</div>
""",
    unsafe_allow_html=True,
)
