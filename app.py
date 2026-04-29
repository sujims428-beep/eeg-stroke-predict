import math
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="EEG Stroke Auxiliary Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

B0 = -2.094
B_AGE = -0.012
B_SEX = 0.332
B_PTBR = 1.092
B_CDTABR = 0.897
THRESHOLD = 0.6945

st.markdown(
    """
<style>
.block-container {
    padding-top: 2.0rem;
    padding-bottom: 1.5rem;
    max-width: 1500px;
}
.main-title {
    background: #2f78cf;
    color: white;
    text-align: center;
    border-radius: 6px;
    padding: 18px 18px 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.main-title h1 {
    font-size: 24px;
    line-height: 1.25;
    margin: 0;
    font-weight: 700;
}
.result-note {
    background: #eaf4ff;
    color: #075a9f;
    border-radius: 6px;
    padding: 12px 14px;
    font-size: 14px;
    margin-bottom: 6px;
}
.decision-line {
    background: #eaf4ff;
    color: #075a9f;
    border-radius: 6px;
    padding: 11px 14px;
    font-size: 14px;
    margin-top: 2px;
}
.footer-note {
    border: 1px solid #ffd8b0;
    background: #fffaf3;
    color: #9a4b00;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 13px;
    margin-top: 14px;
}
div[data-testid="stNumberInput"] input {
    background-color: #f3f5f8;
}
div[data-testid="stSelectbox"] div {
    background-color: #f3f5f8;
}
.stButton > button {
    background-color: #0b84d8;
    color: white;
    border-radius: 6px;
    border: 0;
    height: 38px;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #0574c4;
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def add_block(fig, x0, x1, label, color):
    """Draw one force-style block by shape, avoiding Plotly barmode compatibility issues."""
    y0, y1 = -0.13, 0.13
    fig.add_shape(
        type="rect",
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        line=dict(color="white", width=1.2),
        fillcolor=color,
        layer="below"
    )
    fig.add_annotation(
        x=(x0 + x1) / 2,
        y=0,
        text=label,
        showarrow=False,
        font=dict(size=12, color="white"),
        align="center"
    )

def make_force_plot(contribs, logit_value, prob_value):
    contribs = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)

    fig = go.Figure()

    pos_cursor = 0.0
    neg_cursor = 0.0

    for label, value in contribs:
        if value >= 0:
            x0 = pos_cursor
            x1 = pos_cursor + value
            pos_cursor = x1
            add_block(fig, x0, x1, f"{label} = {value:+.3f}", "#ff0a54")
        else:
            x0 = neg_cursor + value
            x1 = neg_cursor
            neg_cursor = x0
            add_block(fig, x0, x1, f"{label} = {value:+.3f}", "#1683e8")

    xmin = min(neg_cursor, -0.5) - 0.25
    xmax = max(pos_cursor, logit_value, 0.5) + 0.35

    fig.add_vline(x=0, line_width=1.1, line_dash="dot", line_color="#aab4c3")
    fig.add_vline(x=logit_value, line_width=1.8, line_color="#222222")

    fig.add_annotation(
        x=0,
        y=0.30,
        text="base value<br>0.00",
        showarrow=False,
        font=dict(size=12, color="#7c8797"),
        align="center",
    )
    fig.add_annotation(
        x=logit_value,
        y=0.30,
        text=f"f(x)<br>{logit_value:.2f}",
        showarrow=False,
        font=dict(size=12, color="#111111"),
        align="center",
    )
    fig.add_annotation(
        x=xmax,
        y=0.38,
        text="<span style='color:#ff0a54'>higher</span>  ↔  <span style='color:#1683e8'>lower</span>",
        showarrow=False,
        font=dict(size=13),
        align="right",
    )

    fig.update_layout(
        height=260,
        margin=dict(l=50, r=40, t=30, b=36),
        xaxis=dict(
            title="Logit contribution",
            range=[xmin, xmax],
            zeroline=False,
            showgrid=False,
            tickfont=dict(size=11, color="#6b7280"),
            titlefont=dict(size=12, color="#7c8797"),
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-0.32, 0.45],
        ),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(
            text=f"Probability of stroke status: {prob_value*100:.2f}%",
            x=0.5,
            y=0.98,
            xanchor="center",
            yanchor="top",
            font=dict(size=22, family="Georgia, Times New Roman, serif", color="#222"),
        ),
    )
    return fig

st.markdown(
    """
<div class="main-title">
  <h1>Clinical Decision Support for Stroke Auxiliary Recognition Based on EEG Power Ratio</h1>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:6px; border:1px solid #d9dee7; border-radius:6px; margin-bottom:10px;'></div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=66.0, step=1.0, format="%.2f")
with col2:
    sex_label = st.selectbox("Sex", ["Male", "Female"])
with col3:
    ptbr = st.number_input("P-TBR", min_value=0.0, max_value=50.0, value=2.00, step=0.01, format="%.3f")
with col4:
    cdtabr = st.number_input("C-DTABR", min_value=0.0, max_value=50.0, value=3.00, step=0.01, format="%.3f")

sex_code = 0 if sex_label == "Male" else 1

st.button("Start prediction", use_container_width=True)

logit = B0 + B_AGE * age + B_SEX * sex_code + B_PTBR * ptbr + B_CDTABR * cdtabr
prob = sigmoid(logit)

with st.expander("Current input", expanded=True):
    st.dataframe(
        [
            {
                "Age": f"{age:.2f}",
                "Sex": sex_label,
                "Sex code": sex_code,
                "P-TBR": f"{ptbr:.3f}",
                "C-DTABR": f"{cdtabr:.3f}",
            }
        ],
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Predict result", expanded=True):
    st.markdown(
        f"<div class='result-note'>Predict probability segmentation threshold is <b>{THRESHOLD*100:.2f}%</b>.</div>",
        unsafe_allow_html=True,
    )

    contrib_pairs = [
        ("P-TBR", B_PTBR * ptbr),
        ("C-DTABR", B_CDTABR * cdtabr),
        ("Age", B_AGE * age),
        ("Sex", B_SEX * sex_code),
    ]
    fig = make_force_plot(contrib_pairs, logit, prob)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if prob >= THRESHOLD:
        decision = "Model decision: current probability exceeds the stroke auxiliary recognition threshold."
    else:
        decision = "Model decision: current probability is below the stroke auxiliary recognition threshold."
    st.markdown(f"<div class='decision-line'>{decision}</div>", unsafe_allow_html=True)

with st.expander("Model formula and contribution details", expanded=False):
    st.markdown(
        f"""
**Fixed model formula**

`logit(p) = -2.094 - 0.012 × Age + 0.332 × Sex code + 1.092 × P-TBR + 0.897 × C-DTABR`

`p = 1 / [1 + exp(-logit(p))]`

**Current output**

- Sex code: `{sex_code}`；Male = 0，Female = 1  
- logit: `{logit:.4f}`  
- probability: `{prob:.4f}`  
- threshold: `{THRESHOLD:.4f}`
"""
    )

st.markdown(
    """
<div class="footer-note">
<b>Clinical note:</b> This web tool is for research demonstration and auxiliary reference only; it cannot replace head CT/MRI or specialist diagnosis.
</div>
""",
    unsafe_allow_html=True,
)
