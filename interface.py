import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Configuration
st.set_page_config(page_title="MoCredit - Automated Credit Approval System", layout="centered")

# =========================================================================
# AUTOMATED AI MODEL LOADER (Standalone Cloud Execution)
# =========================================================================
@st.cache_resource # Caching to prevent re-loading the model on every user click
def load_ai_model():
    with open("lgb_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_ai_model()
except Exception as e:
    st.error(f"❌ Model file 'lgb_model.pkl' not found in your repository! Error: {e}")

# =========================================================================
# CREDIT LIMIT ALLOCATION ALGORITHM (STAGE 3 POLICY)
# =========================================================================
def allocate_credit_limit(prob_default, income_bracket, existing_debt_usd, financial_apps):
    # 1. Hard Risk Boundary Check
    if prob_default > 0.50:
        return "Rejected", 0

    # 2. Base Credit Limit Assignment based on Income Tier
    if income_bracket == 1:
        base_limit = 2000000
    elif income_bracket == 2:
        base_limit = 5000000
    elif income_bracket == 3:
        base_limit = 10000000
    else:
        base_limit = 2000000

    # 3. Risk Mitigation: Existing Debt Brake
    debt_threshold_usd = 25000  # Equivalent to 625,000,000 VND
    if existing_debt_usd > debt_threshold_usd:
        base_limit = base_limit * 0.5

    # 4. Behavioral Reward Incentive
    if financial_apps <= 2:
        base_limit = base_limit + 500000

    return "Approved", int(base_limit)

# =========================================================================
# USER INTERFACE DESIGN (FRONTEND)
# =========================================================================
st.title("🏦 MoCredit AI - Real-time Credit Underwriting System")
st.write("Enter the applicant's alternative behavioral data to trigger automated AI risk scoring and credit allocation.")
st.markdown("---")

# Two-column input fields layout
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Behavioral Metrics")
    financial_apps = st.number_input("Financial apps installed", min_value=0, max_value=50, value=2)
    screen_time = st.number_input("Avg daily screen time (hours)", min_value=0.0, max_value=24.0, value=4.5)
    unique_apps = st.number_input("Unique apps opened per day", min_value=0, max_value=200, value=25)
    bank_sms = st.number_input("Bank SMS count received", min_value=0, value=10)

with col2:
    st.subheader("💰 Financial & Personal Profile")
    online_txns = st.number_input("Online transactions (Last 30 days)", min_value=0, value=15)
    avg_txn_amt = st.number_input("Avg transaction amount (VND)", min_value=0.0, value=150000.0, step=10000.0)
    existing_debt = st.number_input("Existing debt at financial institutions (VND)", min_value=0.0, value=50000000.0, step=1000000.0)
    
    income_bracket = st.selectbox("Income Bracket Tier", options=[1, 2, 3], format_func=lambda x: f"Tier {x}")
    employment_type = st.selectbox("Employment Type", options=[1, 2, 3], format_func=lambda x: f"Type {x}")
    education_level = st.selectbox("Education Level", options=[1, 2, 3], format_func=lambda x: f"Level {x}")
    cluster_id = st.selectbox("Customer Behavior Cluster ID", options=[0, 1, 2])

st.markdown("---")

# Execution Button
if st.button("🚀 RUN AUTOMATED UNDERWRITING PIPELINE", type="primary", use_container_width=True):
    # Currency conversion to baseline system metrics (USD)
    debt_value = float(existing_debt / 25000)

    # Re-engineering the 24-feature static matrix layout for native LightGBM alignment
    trained_features = [
        "financial_apps_installed", "avg_daily_screen_time_hrs", "unique_apps_per_day",
        "bank_sms_count", "online_txn_count_last_30d", "avg_txn_amount", "existing_debt_amount",
        "income_bracket_1.0", "income_bracket_2.0", "income_bracket_3.0",
        "employment_type_1.0", "employment_type_2.0", "employment_type_3.0",
        "education_level_1.0", "education_level_2.0", "education_level_3.0",
        "cluster_id_0.0", "cluster_id_1.0", "cluster_id_2.0",
        "income_bracket", "employment_type", "education_level", "cluster_id"
    ]

    # Initialize a zero-filled DataFrame with exact 24-column shape
    input_data = pd.DataFrame(np.zeros((1, len(trained_features))), columns=trained_features)
    
    # Mapping numeric inputs
    input_data["financial_apps_installed"] = float(financial_apps)
    input_data["avg_daily_screen_time_hrs"] = float(screen_time)
    input_data["unique_apps_per_day"] = float(unique_apps)
    input_data["bank_sms_count"] = float(bank_sms)
    input_data["online_txn_count_last_30d"] = float(online_txns)
    input_data["avg_txn_amount"] = float(avg_txn_amt)
    input_data["existing_debt_amount"] = float(debt_value)

    # Simulating One-Hot Dummy Activation
    inc, emp, edu, clu = float(income_bracket), float(employment_type), float(education_level), float(cluster_id)
    if f"income_bracket_{inc}" in trained_features: input_data[f"income_bracket_{inc}"] = 1.0
    if f"employment_type_{emp}" in trained_features: input_data[f"employment_type_{emp}"] = 1.0
    if f"education_level_{edu}" in trained_features: input_data[f"education_level_{edu}"] = 1.0
    if f"cluster_id_{clu}" in trained_features: input_data[f"cluster_id_{clu}"] = 1.0

    # Populating original raw features for tree splits
    input_data["income_bracket"] = inc
    input_data["employment_type"] = emp
    input_data["education_level"] = edu
    input_data["cluster_id"] = clu

    # Machine Learning Inference Process
    with st.spinner("AI Engine is executing risk prediction calculations..."):
        input_data_final = input_data[trained_features]
        prob_default = float(model.predict_proba(input_data_final)[:, 1][0])

    # Execute business rule limit engine
    status, allocate_limit = allocate_credit_limit(
        prob_default=prob_default,
        income_bracket=int(income_bracket),
        existing_debt_usd=debt_value,
        financial_apps=int(financial_apps)
    )

    # Display Automated Underwriting Results
    st.subheader("🎯 System Underwriting Results:")
    if status == "Approved":
        st.success(f"🎉 PROFILE APPROVED (DECISION: {status})")
        st.metric(label="Allocated Credit Limit", value=f"{allocate_limit:,} VND")
    else:
        st.error(f"❌ PROFILE REJECTED (DECISION: {status})")
        st.metric(label="Allocated Credit Limit", value="0 VND")
        
    st.info(f"🤖 AI-Predicted Probability of Default (PD): {prob_default * 100:.2f}%")