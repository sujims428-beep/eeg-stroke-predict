# EEG Stroke Prediction Streamlit App

## 文件
- app.py：网页主程序
- requirements.txt：依赖包
- model_parameters.json：固定模型参数

## 在线部署
1. 新建 GitHub 仓库，建议 Public。
2. 上传 app.py、requirements.txt、model_parameters.json 到仓库根目录。
3. 打开 Streamlit Community Cloud。
4. Create app。
5. Repository 选择该仓库；Branch 选择 main；Main file path 填 app.py。
6. Deploy。

## 重要说明
当前网页中的个体化贡献图为基于 Logistic 回归系数的线性贡献可视化，用于模拟 SHAP force plot 的展示风格；若论文中要严格写作 SHAP 图，需要提供原始训练集数据并重新计算 SHAP 值。
