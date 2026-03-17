# 🏦 Loan Default Risk Engine

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/Model-TensorFlow%2FKeras-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

🔗 **Live App:** [Launch Loan Default Risk Engine](https://loan-default-risk-engine-arup-roy.streamlit.app)

An end-to-end deep learning pipeline that predicts loan default 
risk on 395,000+ LendingClub applications — built around the 
Precision-Recall tradeoff that actually matters in lending.

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Approach](#-approach)
- [Key Findings](#-key-findings)
- [Model Performance](#-model-performance)
- [Business Impact](#-business-impact)
- [Business Recommendations](#-business-recommendations)
- [Known Limitations](#-known-limitations)
- [How to Run](#-how-to-run)

---

## ❓ Problem Statement

> **Can we predict loan default risk at the time of application 
> to protect lenders from capital loss?**

Standard loan models optimize for accuracy and look impressive 
on paper while catching almost no actual defaults. This project 
fixes that by optimizing around **Default Recall** — the metric 
that directly protects lending capital — using a neural network 
with computed class weights.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | LendingClub via Kaggle |
| Records | 395,190 loan applications |
| Original Features | 27 |
| Final Features | 67 (after encoding) |
| Target | Binary: Fully Paid (1) / Charged Off (0) |
| Class Balance | 80.4% Fully Paid / 19.6% Charged Off |
| Outlier Treatment | Capped at 99th percentile |

> **Note:** Dataset not included due to size.
> Download from [Kaggle — LendingClub Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
> and place at `data/raw/lending_club_loan_two.csv`

---

## 📁 Project Structure
```
loan-default-risk-engine/
│
├── data/
│   ├── raw/                      ← Original dataset (gitignored)
│   └── processed/                ← Cleaned data (gitignored)
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb    ← Audit, cleaning, outlier treatment
│   ├── 02_eda.ipynb              ← 9 charts answering 6 business questions
│   ├── 03_preprocessing.ipynb   ← Encoding, scaling, class weights
│   └── 04_model_training.ipynb  ← Baseline, NN, threshold optimization
│
├── models/
│   ├── loan_model.keras          ← Trained neural network
│   ├── scaler.pkl                ← Fitted MinMaxScaler
│   ├── best_threshold.pkl        ← Optimized decision threshold
│   ├── feature_names.json        ← Feature names for deployment
│   └── class_weights.json        ← Computed class weights
│
├── reports/figures/              ← All EDA and evaluation charts
├── app.py                        ← Streamlit deployment app
├── requirements.txt
└── README.md
```

---

## 🔬 Approach

### 1. Data Cleaning
- Audited 27 columns for missing values, disguised nulls and outliers
- Capped `dti`, `revol_util` and `annual_inc` at 99th percentile
- Dropped 6 columns: high cardinality text, data leakage (`issue_d`),
  and redundant features (`grade` captured by `sub_grade`)
- Imputed `mort_acc` using `total_acc` group means to avoid selection bias

### 2. Exploratory Data Analysis
- Answered 6 business questions visually
- Found `sub_grade` as strongest categorical predictor
  (A1=3% default rate → G5=51% default rate)
- Identified `int_rate` as strongest numerical predictor (-0.25 correlation)
- Confirmed `small_business` loans have highest default rate (29%)
- Documented multicollinearity between `loan_amnt`/`installment` (0.95)

### 3. Preprocessing
- Dropped 3 redundant features based on correlation analysis
- Ordinal encoded `emp_length` preserving natural order
- One hot encoded 6 categorical columns → 67 total features
- Stratified 80/20 train/test split
- MinMaxScaler fitted only on training data (no leakage)
- Computed class weights from data: `{0: 2.55, 1: 0.62}`

### 4. Model Training
- Established Logistic Regression baseline
- Built dynamic Neural Network (input_dim → input_dim//2 → input_dim//4 → 1)
- Used EarlyStopping and ReduceLROnPlateau callbacks
- Applied computed class weights during training
- Optimized decision threshold using Precision-Recall analysis

### 5. Deployment
- Built Streamlit app with full risk dashboard
- Shows default probability, risk zone and business impact in dollars
- Deployed live on Streamlit Cloud

---

## 🔍 Key Findings

### Interest Rate is the Strongest Predictor
```
Sub Grade A1 → 3% default rate
Sub Grade G5 → 51% default rate
Clear staircase pattern across all 35 sub-grades
```

### Income Inversely Correlates with Default
```
Q1 income (lowest)  → 24% default rate
Q4 income (highest) → 15% default rate
Lower income = higher default risk
```

### Loan Purpose Matters Significantly
```
small_business  → 29% default  ← Highest risk
moving          → 23% default
wedding         → 12% default  ← Lowest risk
car             → 14% default
```

### DTI Separates Defaulters from Repayers
```
Charged Off median DTI → ~19
Fully Paid median DTI  → ~16
Higher debt burden = higher default risk
```

---

## 📈 Model Performance

### Baseline vs Neural Network

| Model | AUC-ROC | Default Recall | Default F1 |
|---|---|---|---|
| Logistic Regression | 0.714 | 5.2% | 0.095 |
| **Neural Network** ✅ | **0.712** | **67.2%** | **0.428** |

> The neural network catches **13x more defaults** than logistic 
> regression despite similar AUC-ROC — demonstrating why 
> class-specific metrics matter more than overall accuracy 
> for imbalanced financial data.

### Threshold Optimization
```
Default threshold (0.50) → Default Recall: 67.2%
Optimized threshold (0.49) → Best F1 for default class
```

---

## 💰 Business Impact

Based on 79,038 test set loans with average loan amount of $14,113:

| Metric | Value |
|---|---|
| Defaults correctly blocked | 10,180 loans |
| Capital protected | **$143,670,340** |
| Defaults missed | 5,323 loans |
| Remaining exposure | $75,123,499 |
| Good loans wrongly rejected | 21,822 loans |
| Opportunity cost | $307,973,886 |

> Without the model: $218.8M total default exposure
> With the model: $75.1M remaining exposure
> **Net protection: $143.7M in loan capital**

---

## 💼 Business Recommendations

**1. Automate Rejection for F and G Grade Applicants**
> Sub grades F and G show 45-51% default rates. Automatic 
> rejection or mandatory collateral for these applicants 
> would significantly reduce default exposure.

**2. Apply Stricter DTI Limits**
> Applicants with DTI above 25 show consistently higher 
> default rates. Cap loan amounts at 3x monthly income 
> for high DTI applicants.

**3. Price Small Business Loans Higher**
> Small business loans default at 29% vs 12% for wedding 
> loans. Risk-based pricing should reflect this — charge 
> 3-5% higher interest rates for business loan applications.

**4. Flag Low Income Applicants**
> Q1 income bracket shows 24% default rate vs 15% for Q4. 
> Require additional documentation or co-signers for 
> applicants in the bottom income quartile.

**5. Use Threshold as a Portfolio Management Tool**
> The decision threshold (0.49) is configurable. Risk teams 
> should adjust based on current portfolio exposure and 
> economic conditions each quarter.

---

## ⚠️ Known Limitations

| Limitation | Detail |
|---|---|
| No temporal validation | Model not tested on time-based split — critical for financial models where economic cycles matter |
| Static threshold | Optimal threshold shifts with economic conditions |
| No explainability | Neural networks lack feature importance — SHAP values would improve clinical interpretability |
| Opportunity cost | 21,822 good loans wrongly rejected — precision improvement needed |
| Data vintage | LendingClub data reflects 2007-2018 lending conditions |

---

## 🚀 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/analyst-ted/loan-default-risk-engine.git
cd loan-default-risk-engine
```

### 2. Create Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
and place at `data/raw/lending_club_loan_two.csv`

### 4. Run Notebooks in Order
```
notebooks/01_data_cleaning.ipynb
notebooks/02_eda.ipynb
notebooks/03_preprocessing.ipynb
notebooks/04_model_training.ipynb
```

### 5. Launch Web App
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualization |
| Scikit-learn | Preprocessing and baseline model |
| TensorFlow 2.21 / Keras | Neural network |
| Streamlit | Web application |
| Joblib | Model serialization |
| Git & GitHub | Version control |

---

*For educational and research purposes only. 
Not intended for actual lending decisions without 
proper regulatory compliance and model validation.*
