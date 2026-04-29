# 基于脑电功率比特征的卒中辅助识别 Streamlit 网页

## 1. 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开本地地址即可查看网页。

## 2. 部署到 Streamlit Community Cloud

1. 新建一个 GitHub 仓库，例如：`eeg-stroke-predict`。
2. 将以下文件上传到仓库根目录：
   - `app.py`
   - `requirements.txt`
   - `README_DEPLOY.md`
3. 打开 Streamlit Community Cloud。
4. 选择 `New app`。
5. Repository 选择刚才的 GitHub 仓库。
6. Branch 选择 `main`。
7. Main file path 填写：`app.py`。
8. 点击 Deploy。
9. 部署完成后会生成一个公开访问网址，例如：`https://你的项目名.streamlit.app/`。

## 3. 需要重点核对的模型参数

当前模型公式：

```text
logit(p) = -2.094 - 0.012 × 年龄 + 0.332 × 性别编码 + 1.092 × P-TBR + 0.897 × C-DTABR
p = 1 / [1 + exp(-logit(p))]
```

当前性别编码：

```text
男 = 0
女 = 1
```

如果原始 R 模型中性别变量的参考水平不是“男=0、女=1”，必须在 `app.py` 中修改 `SEX_CODE`，否则预测概率会有偏差。

## 4. 学术表述建议

网页和论文中建议使用：

```text
卒中状态辅助识别概率
```

不建议写作：

```text
卒中发生概率
```

因为本模型来自回顾性健康人与卒中患者分类数据，不是前瞻性发病风险预测模型。

## 5. 关于变量贡献图

当前网页中的变量贡献图是基于 Logistic 回归系数计算的线性贡献图：

```text
β × (X − reference)
```

它用于展示某个变量相对参考值对 logit 的推高或降低作用。由于没有纳入原始训练集背景分布，当前图不应称为严格意义的 SHAP 图。若后续提供训练集数据，可进一步升级为真实 SHAP 可解释性图。
