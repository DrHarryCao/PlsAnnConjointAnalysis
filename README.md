# Produced by Dr. Junwei Cao, 2026
> ⚠️ **Citation Format:** Cao, JW. [DrHarryCao]. (2026). PlsAnnConjointAnalysis [Computer software]. GitHub. https://github.com/DrHarryCao/PlsAnnConjointAnalysis
# PLS-ANN Analytical Tool V1.8 **JW❤QX @Xiaohongshu drharry**

## Introduction

**English:**
This is a Graphical User Interface (GUI) deep learning analysis tool based on Python `sklearn`. It is primarily developed for the subsequent ANN (Artificial Neural Network) analysis following PLS-SEM (Partial Least Squares Structural Equation Modeling). This tool aims to compensate for the limitations of linear models (such as PLS-SEM/CB-SEM) in capturing non-linear relationships, providing deeper insights through a "Multi-Analytical Approach."

**中文原文:**
这是一个基于 Python `sklearn` 的图形化 (GUI) 深度学习分析工具。它主要为 **PLS-SEM (偏最小二乘结构方程模型)** 的后续 **ANN (人工神经网络)** 分析而开发。本工具旨在弥补线性模型（如 PLS-SEM/CB-SEM）无法捕捉非线性关系的局限，通过**多分析方法 (Multi-Analytical Approach)** 提供更深度的见解。

> ⚠️ **Academic Statement**: The analysis logic and default parameters of this program strictly follow the guidelines published by **Leong et al. (2024)** in the *Journal of Computer Information Systems*: **"An SEM-ANN Approach - Guidelines in Information Systems Research"**
>
> ⚠️ **Citation Format:** Lai-Ying Leong, Teck-Soon Hew, Keng-Boon Ooi, Garry Wei-Han Tan & Alex Koohang (29 Mar 2024): An SEM-ANN Approach - Guidelines in Information Systems Research, Journal of Computer Information Systems, DOI: 10.1080/08874417.2024.2329128

## 1. Guidelines Compliance

### 1.1 Why use PLS-ANN?

**English:**

* **Complementarity**: PLS-SEM excels at verifying causal relationships and hypothesis testing (linear/compensatory models); ANN excels at capturing complex non-linear and non-compensatory relationships with higher predictive accuracy.

* **Two-Stage Strategy**: First, use PLS to filter significant variables, then use ANN for in-depth verification.

**中文原文:**

* **互补性**: PLS-SEM 擅长验证因果关系和假设检验（线性/补偿性模型）；ANN 擅长捕捉复杂的**非线性**和**非补偿性**关系，且预测精度更高。

* **两阶段策略**: 先用 PLS 筛选出显著的变量，再用 ANN 进行深度验证。

### 1.2 Analysis Standards (Based on Leong et al., 2024)

**English:**

1. **Input Selection**: Only include antecedent variables that are proven significant in PLS-SEM as input neurons for the ANN.

2. **Deep Learning Architecture**: It is recommended to use a Multi-Layer Perceptron (MLP) with 2 Hidden Layers, rather than traditional single-layer networks, to achieve deeper learning capacity.

3. **Data Split (90/10)**: Use 90% of the data for training and 10% for testing to maximize training performance and prevent overfitting.

4. **Cross-Validation (10-Fold)**: 10-fold cross-validation is mandatory rather than single-run training, ensuring robustness.

5. **Sample Size Rule**: The sample size should be at least 50 times the number of input variables (50-times rule) to ensure ANN estimation precision.

**中文原文:**

1. **输入变量选择**: 仅将 PLS-SEM 中被证明具有**显著性 (Significant)** 的前因变量作为 ANN 的输入神经元。

2. **深度学习架构**: 默认推荐使用 **多层感知机 (MLP)** 配合 **2个隐藏层 (Hidden Layers)**，而非传统的单层网络，以实现更深度的学习能力。

3. **数据划分 (90/10 Split)**: 推荐使用 **90%** 的数据用于训练 (Training)，**10%** 的数据用于测试 (Testing)，以最大化训练效能并防止过拟合。

4. **交叉验证 (10-Fold CV)**: 强制采用 **10折交叉验证**，而非单纯的一次性训练，以确保结果的稳健性。

5. **样本量法则**: 建议样本量至少为输入变量数量的 **50倍 (50-times rule)**，以保证 ANN 的估计精度（比传统的10倍法则更严谨）。

## 2. Quick Start and Data Preparation

### 2.1 Environment Setup

**English:**

1. **Install Anaconda**: Download and install the version suitable for your system. It is a package manager for Python scientific computing.

2. **Install PyCharm Community**: Download and install. It is the recommended code editor.

3. **Create Project**: Open PyCharm -> New Project -> Choose Conda (recommended) or venv.

4. **Dependencies**: Install the required libraries via terminal:
   `pip install pandas numpy matplotlib seaborn scikit-learn ttkbootstrap joblib plotly openpyxl statsmodels shap`

**中文原文:**

1. **下载 Anaconda**: 下载并安装适合您系统的版本。

2. **下载 PyCharm Community**: 下载安装。

3. **创建项目**: 打开 PyCharm -> New Project -> 选择 Conda (推荐) 或 venv。

4. **安装依赖库**: 在终端运行：
   `pip install pandas numpy matplotlib seaborn scikit-learn ttkbootstrap joblib plotly openpyxl statsmodels shap`

### 2.2 Data Preparation

**English:**

1. Run your model in PLS software (e.g., SmartPLS).

2. Export Latent Variable Scores.

3. Check the P-values of path coefficients and remove non-significant paths.

4. Organize the remaining significant latent variable scores into a new Excel table:

   * **Columns**: All significant independent variables (IVs) and dependent variables (DV).

   * **Rows**: Data for each sample.

**中文原文:**

1. 在 SmartPLS 等PLS软件中运行您的模型。

2. 导出 **潜变量得分 (Latent Variable Scores)**。

3. 检查路径系数的 P 值，剔除**不显著**的路径。

4. 将剩余**显著**的潜变量得分整理为一个新的 Excel 表格：

   * **列 (Columns)**: 所有显著的自变量 (IVs) 和因变量 (DV)。

   * **行 (Rows)**: 每一个样本的数据。

## 3. Operation Manual (Key Sections)

### Tab 1: Modeling

**English:**

* **Input/X**: Significant antecedent variables.

* **Target/Y**: Outcome variables.

* **Parameters**: 2 Hidden Layers, Sigmoid Activation, 10-Fold Cross-Validation, 90% Training Split.

**中文原文:**

* **自变量 (Input / X)**: 选择 PLS 分析中**显著**的前因变量。

* **因变量 (Target / Y)**: 选择您要预测的结果变量。

* **基础参数**: 隐藏层: 2, 激活函数: Sigmoid, 交叉验证: 10, 训练集占比: 90%.

### Tab 2: Advanced Configuration

**English:**

* **Ramsey RESET**: A critical test; if P < 0.05, it statistically justifies using ANN over linear regression.

* **SHAP/PDP/3D Surface**: Advanced visualizations for explaining non-linear relationships.

**中文原文:**

* **非线性检验 (Ramsey RESET)**: **[关键]** 如果 P < 0.05，证明数据存在显著的非线性关系，这为使用 ANN 提供了统计学依据。

* **SHAP/PDP/3D 曲面**: 高级可视化功能，用于解释非线性关系。

## 4. Output Files

**English:**

* **01_Reports**: Contains `Table8_Metrics.xlsx` (RMSE/R²), `Table9_Importance.xlsx` (Sensitivity Analysis), `Table_Linearity_Test.xlsx`.

* **02_Figures**: Includes network architecture, sensitivity plots, and PDP curves.

* **03_HTML_Interactive**: Interactive charts for web viewing.

**中文原文:**

* **01_Reports**: 包含 `Table8_Metrics.xlsx` (RMSE/R²), `Table9_Importance.xlsx` (敏感性分析), `Table_Linearity_Test.xlsx`。

* **02_Figures**: 包含神经网络拓扑图、敏感性分析图等。

* **03_HTML_Interactive**: 可交互的网页图表。

## 5. Author Statement & Copyright

**English:**

* **Non-Commercial**: This project is for academic research and personal learning only. Commercial use is strictly prohibited.

* **Attribution**: Please attribute as: **Source: drharry@Xiaohongshu**

**中文原文:**

* **非商用**: 本项目仅供学术研究与个人学习，**严禁商用**。

* **引用**: 修改、分发或在论文中使用本工具生成的结果，请在致谢或适当位置带上标签：**Source: drharry@小红书**
