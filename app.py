# -*- coding: utf-8 -*-
"""
EEG-based stroke auxiliary recognition tool.
Model source: user's manuscript coefficients.
Run locally:
    streamlit run app.py
"""

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================
# 1. Fixed model parameters
# =========================
# Logistic regression model:
# logit(p) = beta0 + beta_age * age + beta_sex * sex_code + beta_ptbr * P_TBR + beta_c_dtabr * C_DTABR
# IMPORTANT: sex_code is set as 男=0, 女=1 according to the current manuscript-facing draft.
# If the original R model used a different reference level, change SEX_CODE below.
BETA0 = -2.094
BETA_AGE = -0.012
BETA_SEX = 0.332
BETA_PTBR = 1.092
BETA_C_DTABR = 0.897
THRESHOLD = 0.6945

SEX_CODE = {"男": 0, "女": 1}

# Reference values used only for the individual contribution plot.
# This is a linear contribution visualization, not a formal SHAP calculation without original training data.
REFERENCE = {
    "年龄": 66.0,
    "性别": 0.0,
    "P-TBR": 1.705,
    "C-DTABR": 1.170,
}


@dataclass
class PredictionResult:
    age: float
    sex_label: str
    sex_code: int
    p_tbr: float
    c_dtabr: float
    logit: float
    probability: float
    classification: str
    threshold: float


def sigmoid(x: float) -> float:
    """Numerically stable logistic transformation."""
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def predict(age: float, sex_label: str, p_tbr: float, c_dtabr: float) -> PredictionResult:
    """Calculate stroke auxiliary recognition probability."""
    sex_code = SEX_CODE[sex_label]
    logit = BETA0 + BETA_AGE * age + BETA_SEX * sex_code + BETA_PTBR * p_tbr + BETA_C_DTABR * c_dtabr
    probability = sigmoid(logit)
    classification = "高于阈值：倾向卒中状态" if probability >= THRESHOLD else "低于阈值：暂不支持卒中状态"
    return PredictionResult(age, sex_label, sex_code, p_tbr, c_dtabr, logit, probability, classification, THRESHOLD)


def make_gauge(probability: float, threshold: float) -> go.Figure:
    """Probability gauge."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 42}},
            delta={"reference": threshold * 100, "suffix": "%", "increasing": {"color": "#B42318"}, "decreasing": {"color": "#027A48"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#667085"},
                "bar": {"color": "#155EEF"},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#EAECF0",
                "steps": [
                    {"range": [0, 30], "color": "#ECFDF3"},
                    {"range": [30, 60], "color": "#FFFAEB"},
                    {"range": [60, 100], "color": "#FEF3F2"},
                ],
                "threshold": {
                    "line": {"color": "#B42318", "width": 4},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
            title={"text": "卒中辅助识别概率", "font": {"size": 18}},
        )
    )
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def make_contribution_plot(result: PredictionResult) -> go.Figure:
    """Linear contribution visualization relative to fixed reference values."""
    rows = [
        ("年龄", BETA_AGE * (result.age - REFERENCE["年龄"]), result.age, REFERENCE["年龄"]),
        ("性别", BETA_SEX * (result.sex_code - REFERENCE["性别"]), result.sex_label, "男"),
        ("P-TBR", BETA_PTBR * (result.p_tbr - REFERENCE["P-TBR"]), result.p_tbr, REFERENCE["P-TBR"]),
        ("C-DTABR", BETA_C_DTABR * (result.c_dtabr - REFERENCE["C-DTABR"]), result.c_dtabr, REFERENCE["C-DTABR"]),
    ]
    df = pd.DataFrame(rows, columns=["变量", "相对贡献", "当前值", "参考值"]).sort_values("相对贡献")
    colors = ["#1570EF" if x < 0 else "#E31B54" for x in df["相对贡献"]]
    text = [f"{v:+.3f}" for v in df["相对贡献"]]

    fig = go.Figure(
        go.Bar(
            x=df["相对贡献"],
            y=df["变量"],
            orientation="h",
            marker_color=colors,
            text=text,
            textposition="outside",
            hovertemplate="变量：%{y}<br>相对贡献：%{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#667085")
    fig.update_layout(
        title="个体化变量贡献图（基于Logit线性贡献）",
        xaxis_title="相对参考值的Logit贡献：β × (X − reference)",
        yaxis_title="",
        height=330,
        margin=dict(l=20, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_formula_card(result: PredictionResult) -> str:
    """HTML formula card."""
    return f"""
    <div class='formula-card'>
        <div class='formula-title'>固定模型公式</div>
        <div class='formula-body'>
            logit(p) = -2.094 − 0.012 × 年龄 + 0.332 × 性别编码 + 1.092 × P-TBR + 0.897 × C-DTABR<br>
            p = 1 / [1 + exp(−logit(p))]
        </div>
        <div class='formula-sub'>当前输入：性别编码={result.sex_code}；logit={result.logit:.4f}；p={result.probability:.4f}</div>
    </div>
    """


def build_download_record(result: PredictionResult) -> pd.DataFrame:
    """Create one-row CSV record for download."""
    return pd.DataFrame(
        [{
            "age": result.age,
            "sex": result.sex_label,
            "sex_code": result.sex_code,
            "P_TBR": result.p_tbr,
            "C_DTABR": result.c_dtabr,
            "logit": round(result.logit, 6),
            "probability": round(result.probability, 6),
            "probability_percent": round(result.probability * 100, 2),
            "threshold": result.threshold,
            "classification": result.classification,
        }]
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {max-width: 1180px; padding-top: 1.6rem; padding-bottom: 2rem;}
        h1, h2, h3 {letter-spacing: -0.02em;}
        .hero {
            padding: 22px 26px;
            border-radius: 22px;
            background: linear-gradient(135deg, #0B3B75 0%, #155EEF 52%, #6C63FF 100%);
            color: white;
            box-shadow: 0 18px 50px rgba(21, 94, 239, .20);
            margin-bottom: 18px;
        }
        .hero-title {font-size: 28px; font-weight: 800; margin-bottom: 8px;}
        .hero-sub {font-size: 15px; opacity: .92; line-height: 1.65;}
        .metric-card {
            padding: 18px 20px;
            border: 1px solid #EAECF0;
            border-radius: 18px;
            background: #FFFFFF;
            box-shadow: 0 10px 24px rgba(16, 24, 40, .06);
        }
        .metric-title {font-size: 14px; color: #667085; margin-bottom: 6px;}
        .metric-value {font-size: 32px; font-weight: 800; color: #101828; margin-bottom: 2px;}
        .metric-sub {font-size: 13px; color: #667085;}
        .risk-high {color: #B42318; font-weight: 800;}
        .risk-low {color: #027A48; font-weight: 800;}
        .formula-card {
            border: 1px solid #D0D5DD;
            border-radius: 18px;
            padding: 16px 18px;
            background: #F9FAFB;
            margin-top: 8px;
            margin-bottom: 12px;
        }
        .formula-title {font-weight: 800; color: #101828; margin-bottom: 8px;}
        .formula-body {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 14px; color: #344054; line-height: 1.7;}
        .formula-sub {font-size: 13px; color: #667085; margin-top: 8px;}
        .notice {
            border-left: 4px solid #F79009;
            background: #FFFAEB;
            padding: 12px 14px;
            border-radius: 12px;
            color: #7A2E0E;
            font-size: 14px;
            line-height: 1.7;
        }
        .small-note {font-size: 13px; color: #667085; line-height: 1.7;}
        [data-testid="stMetricValue"] {font-size: 30px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="EEG卒中辅助识别工具",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.markdown(
        """
        <div class='hero'>
            <div class='hero-title'>基于脑电功率比特征的卒中辅助识别工具</div>
            <div class='hero-sub'>
                本工具基于固定的多变量Logistic回归模型，输入年龄、性别、P-TBR和C-DTABR后，自动计算卒中状态辅助识别概率，
                并生成个体化变量贡献图。该工具仅用于科研展示和临床辅助参考，不能替代头颅CT/MRI及专科医师诊断。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("输入患者信息")
        st.caption("请使用与建模阶段一致的脑电预处理和功率比计算方法。")

        if "example_loaded" not in st.session_state:
            st.session_state.example_loaded = False

        if st.button("载入论文示例病例", use_container_width=True):
            st.session_state.age = 74.0
            st.session_state.sex_label = "女"
            st.session_state.p_tbr = 7.72
            st.session_state.c_dtabr = 4.04
            st.session_state.example_loaded = True

        age = st.number_input("年龄（岁）", min_value=18.0, max_value=110.0, value=st.session_state.get("age", 66.0), step=1.0, key="age")
        sex_label = st.selectbox("性别", options=["男", "女"], index=1 if st.session_state.get("sex_label", "男") == "女" else 0, key="sex_label")
        p_tbr = st.number_input("P-TBR：顶区 θ/β 功率比", min_value=0.0, max_value=20.0, value=st.session_state.get("p_tbr", 1.705), step=0.01, format="%.3f", key="p_tbr")
        c_dtabr = st.number_input("C-DTABR：中央区 δ/(θ+α+β) 功率比", min_value=0.0, max_value=20.0, value=st.session_state.get("c_dtabr", 1.170), step=0.01, format="%.3f", key="c_dtabr")

        st.divider()
        st.subheader("模型设置")
        st.write(f"判别阈值：**{THRESHOLD:.4f}**")
        st.write("性别编码：男=0，女=1")
        st.caption("若原始R模型的性别参考水平不同，请先修改代码中的 SEX_CODE。")

    result = predict(age, sex_label, p_tbr, c_dtabr)
    probability_percent = result.probability * 100
    risk_class = "risk-high" if result.probability >= result.threshold else "risk-low"

    col1, col2, col3 = st.columns([1.0, 1.0, 1.15])
    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>预测概率</div>
                <div class='metric-value'>{probability_percent:.2f}%</div>
                <div class='metric-sub'>阈值：{result.threshold * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>模型判定</div>
                <div class='metric-value {risk_class}'>{'高于阈值' if result.probability >= result.threshold else '低于阈值'}</div>
                <div class='metric-sub'>{result.classification}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(make_formula_card(result), unsafe_allow_html=True)

    left, right = st.columns([1.0, 1.15])
    with left:
        st.plotly_chart(make_gauge(result.probability, result.threshold), use_container_width=True)
    with right:
        st.plotly_chart(make_contribution_plot(result), use_container_width=True)

    st.subheader("当前输入与输出")
    record = build_download_record(result)
    st.dataframe(record, use_container_width=True, hide_index=True)
    st.download_button(
        label="下载本次预测结果 CSV",
        data=record.to_csv(index=False).encode("utf-8-sig"),
        file_name="eeg_stroke_prediction_result.csv",
        mime="text/csv",
        use_container_width=False,
    )

    st.markdown(
        """
        <div class='notice'>
        重要说明：本工具输出的是基于当前模型的“卒中状态辅助识别概率”，不是未来卒中发生风险，也不是影像学诊断结论。
        模型来自单中心回顾性数据，当前版本适合科研展示、方法复现和内部测试；正式临床使用前需进行独立外部验证和前瞻性评估。
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("查看变量定义与模型来源"):
        st.markdown(
            """
            - **P-TBR**：顶区 theta/beta 功率比，即顶区 θ/β 比值。
            - **C-DTABR**：中央区 delta/(theta+alpha+beta) 功率比，即中央区 δ/(θ+α+β) 比值。
            - **模型形式**：多变量Logistic回归。
            - **预测阈值**：0.6945，来自训练集Youden指数最大化原则。
            - **变量贡献图**：采用固定参考值下的线性Logit贡献展示，即 β × (X − reference)。没有原始训练集背景分布时，该图不应称为严格意义的SHAP图。
            """
        )


if __name__ == "__main__":
    main()
