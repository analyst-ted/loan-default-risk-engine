import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import json

# ─── Page Config ──────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Risk Engine",
    page_icon="🏦",
    layout="wide"
)

# ─── Load Assets ──────────────────────────────────────
@st.cache_resource
def load_assets():
    try:
        model     = tf.keras.models.load_model('models/loan_model.keras')
        scaler    = joblib.load('models/scaler.pkl')
        threshold = joblib.load('models/best_threshold.pkl')
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        with open('models/class_weights.json', 'r') as f:
            class_weights = json.load(f)
        return model, scaler, threshold, feature_names, class_weights
    except Exception as e:
        st.error(f"Failed to load model assets: {e}")
        st.stop()

model, scaler, threshold, feature_names, class_weights = load_assets()

# ─── Header ───────────────────────────────────────────
st.title("🏦 Loan Default Risk Engine")
st.markdown("### AI-powered loan default prediction for credit risk teams")
st.markdown("---")

# ─── Sidebar Inputs ───────────────────────────────────
st.sidebar.header("📋 Applicant Profile")

# Financial Details
st.sidebar.subheader("💰 Financial Information")
loan_amnt = st.sidebar.number_input(
    "Loan Amount ($)", min_value=500,
    max_value=40000, value=10000, step=500)

annual_inc = st.sidebar.number_input(
    "Annual Income ($)", min_value=0,
    max_value=250000, value=65000, step=1000)

dti = st.sidebar.slider(
    "Debt-to-Income Ratio", min_value=0.0,
    max_value=36.0, value=15.0, step=0.1)

int_rate = st.sidebar.slider(
    "Interest Rate (%)", min_value=5.0,
    max_value=31.0, value=12.5, step=0.1)

# Loan Details
st.sidebar.subheader("📄 Loan Details")

all_subgrades = [
    'A1','A2','A3','A4','A5',
    'B1','B2','B3','B4','B5',
    'C1','C2','C3','C4','C5',
    'D1','D2','D3','D4','D5',
    'E1','E2','E3','E4','E5',
    'F1','F2','F3','F4','F5',
    'G1','G2','G3','G4','G5'
]

sub_grade = st.sidebar.selectbox(
    "Credit Sub-Grade", options=all_subgrades, index=7)

term = st.sidebar.selectbox(
    "Loan Term", options=[36, 60])

purpose = st.sidebar.selectbox(
    "Loan Purpose", options=[
        'debt_consolidation', 'credit_card',
        'home_improvement', 'other', 'major_purchase',
        'small_business', 'car', 'medical', 'moving',
        'vacation', 'house', 'wedding', 'renewable_energy',
        'educational'
    ])

# Personal Details
st.sidebar.subheader("👤 Personal Information")

home_ownership = st.sidebar.selectbox(
    "Home Ownership", options=['RENT', 'MORTGAGE', 'OWN', 'OTHER'])

emp_length = st.sidebar.selectbox(
    "Employment Length", options=[
        '< 1 year', '1 year', '2 years', '3 years',
        '4 years', '5 years', '6 years', '7 years',
        '8 years', '9 years', '10+ years'
    ])

verification_status = st.sidebar.selectbox(
    "Income Verification", options=[
        'Verified', 'Source Verified', 'Not Verified'])

# Credit History
st.sidebar.subheader("📊 Credit History")

revol_util = st.sidebar.slider(
    "Revolving Credit Utilization (%)",
    min_value=0.0, max_value=98.0, value=50.0)

revol_bal = st.sidebar.number_input(
    "Revolving Balance ($)",
    min_value=0, max_value=500000, value=10000)

total_acc = st.sidebar.slider(
    "Total Credit Accounts",
    min_value=2, max_value=151, value=25)

mort_acc = st.sidebar.slider(
    "Mortgage Accounts",
    min_value=0, max_value=10, value=1)

pub_rec = st.sidebar.slider(
    "Public Records",
    min_value=0, max_value=10, value=0)

open_acc_proxy = st.sidebar.slider(
    "Open Credit Accounts",
    min_value=0, max_value=50, value=10)

initial_list_status = st.sidebar.selectbox(
    "Initial List Status", options=['f', 'w'])

application_type = st.sidebar.selectbox(
    "Application Type",
    options=['INDIVIDUAL', 'JOINT', 'DIRECT_PAY'])

# ─── Feature Engineering ──────────────────────────────
emp_map = {
    '< 1 year': 0, '1 year': 1, '2 years': 2,
    '3 years': 3,  '4 years': 4, '5 years': 5,
    '6 years': 6,  '7 years': 7, '8 years': 8,
    '9 years': 9,  '10+ years': 10
}

def build_input():
    """
    Builds input dataframe matching exact training features.
    Starts with zeros then fills matching feature values.
    """
    try:
        input_dict = {feat: 0 for feat in feature_names}

        # Numerical features
        input_dict['loan_amnt']   = loan_amnt
        input_dict['term']        = term
        input_dict['int_rate']    = int_rate
        input_dict['emp_length']  = emp_map[emp_length]
        input_dict['annual_inc']  = annual_inc
        input_dict['dti']         = dti
        input_dict['pub_rec']     = pub_rec
        input_dict['revol_bal']   = revol_bal
        input_dict['revol_util']  = revol_util
        input_dict['total_acc']   = total_acc
        input_dict['mort_acc']    = mort_acc

        # One hot encoded features
        sg_col = f'sub_grade_{sub_grade}'
        if sg_col in input_dict:
            input_dict[sg_col] = 1

        ho_col = f'home_ownership_{home_ownership}'
        if ho_col in input_dict:
            input_dict[ho_col] = 1

        vs_col = f'verification_status_{verification_status}'
        if vs_col in input_dict:
            input_dict[vs_col] = 1

        pu_col = f'purpose_{purpose}'
        if pu_col in input_dict:
            input_dict[pu_col] = 1

        il_col = f'initial_list_status_{initial_list_status}'
        if il_col in input_dict:
            input_dict[il_col] = 1

        at_col = f'application_type_{application_type}'
        if at_col in input_dict:
            input_dict[at_col] = 1

        return pd.DataFrame([input_dict])

    except Exception as e:
        st.error(f"Error building input: {e}")
        return None

# ─── Prediction Section ───────────────────────────────
st.markdown("## 🔍 Risk Assessment")

if st.button("🚀 Assess Default Risk", use_container_width=True):
    try:
        # Build and scale input
        input_df = build_input()
        if input_df is None:
            st.stop()

        scaled_input = scaler.transform(input_df)

        # Predict
        default_prob = 1 - model.predict(
            scaled_input, verbose=0)[0][0]
        repay_prob = 1 - default_prob
        prediction = int(default_prob >= threshold)

        # ─── Results Display ──────────────────────────
        col1, col2, col3 = st.columns(3)

        with col1:
            if prediction == 1:
                st.error("## 🔴 HIGH RISK\nLikely to Default")
            else:
                st.success("## 🟢 LOW RISK\nLikely to Repay")

        with col2:
            st.metric("Default Probability",
                      f"{default_prob:.1%}")
            st.metric("Repayment Probability",
                      f"{repay_prob:.1%}")

        with col3:
            st.metric("Decision Threshold",
                      f"{threshold:.2f}")
            st.metric("Interest Rate",
                      f"{int_rate}%")

        # ─── Risk Gauge ───────────────────────────────
        st.markdown("### 📊 Risk Level")
        st.progress(float(default_prob))

        if default_prob < 0.2:
            st.success("🟢 Low Risk Zone (< 20%)")
        elif default_prob < 0.4:
            st.warning("🟡 Moderate Risk Zone (20-40%)")
        else:
            st.error("🔴 High Risk Zone (> 40%)")

        # ─── Business Impact ──────────────────────────
        st.markdown("### 💰 Business Impact Estimate")
        avg_loan = 14113
        impact_col1, impact_col2 = st.columns(2)

        with impact_col1:
            if prediction == 1:
                st.error(
                    f"**Potential Loss if Approved:**\n\n"
                    f"${loan_amnt:,} at risk"
                )
            else:
                st.success(
                    f"**Expected Revenue if Approved:**\n\n"
                    f"${loan_amnt * (int_rate/100):,.0f} "
                    f"estimated interest income"
                )

        with impact_col2:
            st.info(
                f"**Model Confidence:**\n\n"
                f"This prediction is based on a model that "
                f"protects an estimated $143M in loan capital "
                f"by catching 65.7% of defaults."
            )

        # ─── Applicant Summary ────────────────────────
        st.markdown("### 📋 Applicant Summary")
        sum_col1, sum_col2, sum_col3 = st.columns(3)

        with sum_col1:
            st.write(f"**Loan Amount:** ${loan_amnt:,}")
            st.write(f"**Annual Income:** ${annual_inc:,}")
            st.write(f"**DTI Ratio:** {dti}%")
            st.write(f"**Interest Rate:** {int_rate}%")

        with sum_col2:
            st.write(f"**Sub Grade:** {sub_grade}")
            st.write(f"**Purpose:** {purpose}")
            st.write(f"**Term:** {term} months")
            st.write(f"**Home:** {home_ownership}")

        with sum_col3:
            st.write(f"**Employment:** {emp_length}")
            st.write(f"**Total Accounts:** {total_acc}")
            st.write(f"**Revolving Util:** {revol_util}%")
            st.write(f"**Public Records:** {pub_rec}")

        # ─── Disclaimer ───────────────────────────────
        st.markdown("---")
        st.caption(
            "⚠️ This tool is for research and educational "
            "purposes only. Not intended for actual lending "
            "decisions without proper regulatory compliance "
            "and model validation."
        )

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

else:
    # Default state — show model info
    st.info("👈 Fill in applicant details and click "
            "**Assess Default Risk** to get started.")

    st.markdown("### 📈 Model Performance Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", "Neural Network")
    m2.metric("AUC-ROC", "0.712")
    m3.metric("Default Recall", "67.2%")
    m4.metric("Capital Protected", "$143.7M")

    st.markdown("### 🎯 How It Works")
    st.markdown("""
    1. **Enter applicant details** in the sidebar
    2. **Click Assess** to run the AI model
    3. **Review the risk score** and probability
    4. **See business impact** in dollar terms
    5. **Make informed decisions** with full transparency
    """)