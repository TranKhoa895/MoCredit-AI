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

# Global model initialization
model = load_ai_model()

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

    # 1. EXTRACT PRECISE FEATURES DIRECTLY FROM LIGHTGBM MODEL (Prevent manual mismatches)
    try:
        trained_features = model.feature_name_
    except AttributeError:
        trained_features = [
            "financial_apps_installed", "avg_daily_screen_time_hrs", "unique_apps_per_day",
            "bank_sms_count", "online_txn_count_last_30d", "avg_txn_amount", "existing_debt_amount",
            "income_bracket_1.0", "income_bracket_2.0", "income_bracket_3.0",
            "employment_type_1.0", "employment_type_2.0", "employment_type_3.0",
            "education_level_1.0", "education_level_2.0", "education_level_3.0",
            "cluster_id_0.0", "cluster_id_1.0", "cluster_id_2.0",
            "income_bracket", "employment_type", "education_level", "cluster_id"
        ]

    # 2. Initialize a zero-filled DataFrame with exact columns shape
    input_data = pd.DataFrame(np.zeros((1, len(trained_features))), columns=trained_features)
    
    # 3. Mapping continuous numeric inputs
    for col in input_data.columns:
        if col == "financial_apps_installed": input_data[col] = float(financial_apps)
        elif col == "avg_daily_screen_time_hrs": input_data[col] = float(screen_time)
        elif col == "unique_apps_per_day": input_data[col] = float(unique_apps)
        elif col == "bank_sms_count": input_data[col] = float(bank_sms)
        elif col == "online_txn_count_last_30d": input_data[col] = float(online_txns)
        elif col == "avg_txn_amount": input_data[col] = float(avg_txn_amt)
        elif col == "existing_debt_amount": input_data[col] = float(debt_value)

    # 4. Simulating One-Hot Dummy Activation (Supports both integer and .0 string formats)
    inc_val, emp_val, edu_val, clu_val = int(income_bracket), int(employment_type), int(education_level), int(cluster_id)
    
    for col in [f"income_bracket_{inc_val}.0", f"income_bracket_{inc_val}"]:
        if col in input_data.columns: input_data[col] = 1.0
        
    for col in [f"employment_type_{emp_val}.0", f"employment_type_{emp_val}"]:
        if col in input_data.columns: input_data[col] = 1.0
        
    for col in [f"education_level_{edu_val}.0", f"education_level_{edu_val}"]:
        if col in input_data.columns: input_data[col] = 1.0
        
    for col in [f"cluster_id_{clu_val}.0", f"cluster_id_{clu_val}"]:
        if col in input_data.columns: input_data[col] = 1.0

    # 5. Populating original raw features for tree splits if required
    if "income_bracket" in input_data.columns: input_data["income_bracket"] = float(inc_val)
    if "employment_type" in input_data.columns: input_data["employment_type"] = float(emp_val)
    if "education_level" in input_data.columns: input_data["education_level"] = float(edu_val)
    if "cluster_id" in input_data.columns: input_data["cluster_id"] = float(clu_val)

    # 6. Align structure and force float64 datatype conversion
    input_data_final = input_data[trained_features].astype(float)

    # Machine Learning Inference Process
    with st.spinner("AI Engine is executing risk prediction calculations..."):
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