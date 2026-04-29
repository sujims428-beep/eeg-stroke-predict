
import streamlit as st
import math
import plotly.graph_objects as go

st.set_page_config(page_title="卒中辅助识别工具", layout="wide")

B0 = -2.094
B_AGE = -0.012
B_SEX = 0.332
B_PTBR = 1.092
B_CDTABR = 0.897
THRESHOLD = 0.6945

st.markdown("""
<div style="background:#2f73c8;padding:18px;border-radius:8px;color:white;text-align:center;">
<h2>基于脑电功率比特征的卒中辅助识别工具</h2>
<p>输入患者特征后，系统自动计算卒中状态辅助识别概率，并生成个体化贡献解释图。</p>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
with c1:
    age = st.number_input("年龄", value=66.0)
with c2:
    sex = st.selectbox("性别", ["男","女"])
with c3:
    ptbr = st.number_input("P-TBR（顶区θ/β）", value=2.0)
with c4:
    cdtabr = st.number_input("C-DTABR（中央区δ/(θ+α+β)）", value=4.0)

sex_code = 0 if sex=="男" else 1

if st.button("开始预测", use_container_width=True):
    logit = B0 + B_AGE*age + B_SEX*sex_code + B_PTBR*ptbr + B_CDTABR*cdtabr
    p = 1/(1+math.exp(-logit))

    st.subheader("Current input")
    st.table({
        "年龄":[age],
        "性别":[sex],
        "P-TBR":[ptbr],
        "C-DTABR":[cdtabr]
    })

    st.subheader("Predict result")
    st.info(f"预测概率分割阈值：{THRESHOLD*100:.2f}%")
    st.markdown(f"## 卒中状态概率：{p*100:.2f}%")

    contrib = {
        "年龄": B_AGE*age,
        "性别": B_SEX*sex_code,
        "P-TBR": B_PTBR*ptbr,
        "C-DTABR": B_CDTABR*cdtabr
    }

    fig = go.Figure()
    for k,v in contrib.items():
        fig.add_trace(go.Bar(
            x=[abs(v)],
            y=["贡献路径"],
            orientation="h",
            name=f"{k}={round(v,3)}",
            marker_color=("crimson" if v>=0 else "dodgerblue")
        ))

    fig.update_layout(
        barmode="stack",
        title="个体化变量贡献图（Force-style）",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

    if p >= THRESHOLD:
        st.success("结果：高于阈值，支持卒中状态")
    else:
        st.warning("结果：低于阈值，不支持卒中状态")

st.markdown("---")
st.markdown("**临床说明：** 本工具仅用于科研展示和辅助参考，不能替代头颅CT/MRI及专科医师判断。")
