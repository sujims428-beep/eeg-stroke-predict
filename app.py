import math
import streamlit as st

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
    margin-bottom: 10px;
}
.force-title {
    font-family: Georgia, 'Times New Roman', serif;
    text-align: center;
    font-size: 25px;
    font-weight: 700;
    color: #222;
    margin: 12px 0 6px 0;
}
.force-wrap {
    position: relative;
    height: 245px;
    background: white;
    border-radius: 6px;
    margin: 0 8px 10px 8px;
}
.axis-line {
    position: absolute;
    left: 8%;
    right: 8%;
    top: 118px;
    height: 1px;
    background: #4b5563;
}
.base-line {
    position: absolute;
    left: 23%;
    top: 70px;
    height: 120px;
    border-left: 1px dashed #aab4c3;
}
.fx-line {
    position: absolute;
    left: 62%;
    top: 70px;
    height: 120px;
    border-left: 2px solid #222;
}
.base-label {
    position: absolute;
    left: 20.2%;
    top: 55px;
    color: #7c8797;
    font-size: 12px;
    text-align: center;
}
.fx-label {
    position: absolute;
    left: 60.8%;
    top: 55px;
    color: #111;
    font-size: 12px;
    text-align: center;
}
.higher-lower {
    position: absolute;
    right: 6%;
    top: 52px;
    font-size: 13px;
}
.red-text { color: #ff0a54; }
.blue-text { color: #1683e8; }
.bar-zone {
    position: absolute;
    left: 12%;
    right: 10%;
    top: 107px;
    height: 38px;
    display: flex;
    align-items: center;
}
.neg-zone {
    width: 18%;
    height: 30px;
    display: flex;
    flex-direction: row-reverse;
}
.pos-zone {
    width: 82%;
    height: 30px;
    display: flex;
}
.red-block {
    height: 30px;
    background: #ff0a54;
    color: white;
    border-right: 1px solid white;
    font-size: 12px;
    display:flex;
    align-items:center;
    justify-content:center;
    white-space: nowrap;
    overflow: hidden;
}
.blue-block {
    height: 30px;
    background: #1683e8;
    color: white;
    border-left: 1px solid white;
    font-size: 12px;
    display:flex;
    align-items:center;
    justify-content:center;
    white-space: nowrap;
    overflow: hidden;
}
.tick-label {
    position:absolute;
    top:155px;
    color:#6b7280;
    font-size:11px;
}
.x-title {
    position:absolute;
    top:185px;
    left:46%;
    color:#7c8797;
    font-size:12px;
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

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def block_html(label, value, total_pos, total_neg):
    if value >= 0:
        width = max(8, abs(value) / max(total_pos, 0.001) * 100)
        return f"<div class='red-block' style='width:{width:.1f}%;'>{label} = {value:+.3f}</div>"
    width = max(8, abs(value) / max(total_neg, 0.001) * 100)
    return f"<div class='blue-block' style='width:{width:.1f}%;'>{label} = {value:+.3f}</div>"

def make_force_html(prob, logit, contribs):
    contribs = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)
    pos = [(k,v) for k,v in contribs if v >= 0]
    neg = [(k,v) for k,v in contribs if v < 0]
    total_pos = sum(v for _,v in pos)
    total_neg = sum(abs(v) for _,v in neg)

    neg_blocks = "".join(block_html(k, v, total_pos, total_neg) for k,v in neg)
    pos_blocks = "".join(block_html(k, v, total_pos, total_neg) for k,v in pos)

    return f"""
<div class="force-title">Probability of stroke status: {prob*100:.2f}%</div>
<div class="force-wrap">
  <div class="axis-line"></div>
  <div class="base-line"></div>
  <div class="fx-line"></div>
  <div class="base-label">base value<br>0.00</div>
  <div class="fx-label">f(x)<br>{logit:.2f}</div>
  <div class="higher-lower"><span class="red-text">higher</span> ↔ <span class="blue-text">lower</span></div>
  <div class="bar-zone">
    <div class="neg-zone">{neg_blocks}</div>
    <div class="pos-zone">{pos_blocks}</div>
  </div>
  <div class="tick-label" style="left:10%;">-1</div>
  <div class="tick-label" style="left:23%;">0</div>
  <div class="tick-label" style="left:41%;">1</div>
  <div class="tick-label" style="left:60%;">2</div>
  <div class="tick-label" style="left:78%;">3</div>
  <div class="x-title">Logit contribution</div>
</div>
"""

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
    st.markdown(make_force_html(prob, logit, contrib_pairs), unsafe_allow_html=True)

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
