import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Healthcare Risk Prediction API", version="1.0")

# Load model from pickle file
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully")

# Define input schema
class MemberData(BaseModel):
    age: int
    gender: int
    claim_count_90days: int
    er_visits_6months: int
    total_claim_cost: float
    medication_count: int
    has_diabetes: int
    has_hypertension: int
    has_copd: int
    high_er_usage: int
    high_claim_frequency: int
    high_cost_member: int
    multiple_chronic: int
    high_medication_burden: int
    risk_indicator: int

@app.get("/")
def home():
    return {"message": "Healthcare Risk Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: MemberData):
    input_df = pd.DataFrame([data.dict()])
    risk_score = model.predict_proba(input_df)[0][1]
    risk_label = int(model.predict(input_df)[0])

    return {
        "risk_score": round(float(risk_score), 4),
        "risk_label": risk_label,
        "risk_category": "High Risk" if risk_label == 1 else "Low Risk",
        "recommendation": "Enroll in care management program" if risk_label == 1 else "Routine monitoring"
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)