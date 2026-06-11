# STAGE 4: PRODUCTION API BACKEND WITH FASTAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import uvicorn

# 1. Initialize FastAPI Application 
app = FastAPI(
    title = "Automated Credit Scoring API",
    description = "Real-time automated risk assessment & credit limit allocation",
    version = "1.0.0"
)

# 2. Load the serialized AI Model onto the Server
MODEL_PATH = "lgb_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        print("AI model loaded successfully onto the Server!")
except FileNotFoundError:
    print(f"Could not find the file {MODEL_PATH} in the current directory. Please double-check!")

# 3. Define the incoming customer data payload schema 
# (Ensures all columns required by the LightGBM model are declared properly)
class CustomerData(BaseModel):
    financial_apps_installed: int
    avg_daily_screen_time_hrs: float
    unique_apps_per_day: int
    bank_sms_count: int
    online_txn_count_last_30d: int
    avg_txn_amount: float
    existing_debt_amount_vnd: float  
    income_bracket: int          
    employment_type: int         
    education_level: int         
    cluster_id: int              

# 4. Define the credit limit allocation logic function 
def allocate_credit_limit(prob_default, income_bracket, existing_debt_usd, financial_apps):
    if prob_default > 0.5:
        return "Rejected", 0
    status = "Approved"

    if income_bracket == 1:
        base_limit = 2000000
    elif income_bracket == 2:
        base_limit = 5000000
    elif income_bracket == 3:
        base_limit = 10000000
    else:
        base_limit = 2000000
    
    existing_debt_vnd = existing_debt_usd * 25000
    if existing_debt_vnd > 625000000:
        allocate_limit = base_limit * 0.5
    else:
        if financial_apps < 2:
            allocate_limit = base_limit + 1000000
        else:
            allocate_limit = base_limit
    
    return status, int(allocate_limit)

# 5. API Endpoint to receive profiles for real-time credit underwriting
@app.post("/api/v1/credit-approval", summary="Automated credit scoring and limit approval")
async def credit_approval(customer: CustomerData):
    try:
        data_dict = customer.dict()
        
        debt_value = float(data_dict['existing_debt_amount_vnd'] / 25000)

        user_df = pd.DataFrame([data_dict])
        mock_data = pd.DataFrame([
            {"income_bracket": 1, "employment_type": 1, "education_level": 1, "cluster_id": 0},
            {"income_bracket": 2, "employment_type": 2, "education_level": 2, "cluster_id": 1},
            {"income_bracket": 3, "employment_type": 3, "education_level": 3, "cluster_id": 2}
        ])
        combined_df = pd.concat([user_df, mock_data], ignore_index=True)
        combined_df.loc[0, 'existing_debt_amount'] = debt_value
        if 'existing_debt_amount_vnd' in combined_df.columns:
            combined_df = combined_df.drop(columns=["existing_debt_amount_vnd"])

        cat_cols = ["income_bracket", "employment_type", "education_level", "cluster_id"]
        for col in cat_cols:
            combined_df[col] = combined_df[col].astype(str)    

        encoded_df = pd.get_dummies(combined_df, columns=cat_cols)
        final_row = encoded_df.iloc[[0]].copy()
        
        trained_features = model.feature_name_
        for col in trained_features:
            if col not in final_row.columns:
                final_row[col] = 0.0

        input_data_final = final_row[trained_features]

        prob_default = float(model.predict_proba(input_data_final)[:, 1][0])
        status, allocate_limit = allocate_credit_limit(
            prob_default = prob_default,
            income_bracket = customer.income_bracket,
            existing_debt_usd = debt_value,
            financial_apps = customer.financial_apps_installed
        )

        return {
            "status": "Success",
            "risk_score_probability": round(prob_default, 4),
            "decision": status,
            "allocated_limit_vnd": allocate_limit,
            "currency": "VND"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# 6. Spin up the Uvicorn ASGI Server
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port = 8000, reload = True)