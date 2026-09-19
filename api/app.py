import pandas as pd
import pickle
import os
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from fastapi.responses import PlainTextResponse
import time

app = FastAPI(title="Healthcare Risk Prediction API", version="1.0")

# Load model from pickle file
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).with_name("model.pkl"))))
with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully")

# Define Prometheus metrics safely
try:
    prediction_counter = Counter(
        'predictions_total',
        'Total number of predictions made',
        ['risk_category']
    )
except ValueError:
    prediction_counter = REGISTRY._names_to_collectors.get('predictions_total')

try:
    latency_histogram = Histogram(
        'prediction_latency_seconds',
        'Time taken to make a prediction'
    )
except ValueError:
    latency_histogram = REGISTRY._names_to_collectors.get('prediction_latency_seconds')

try:
    high_risk_gauge = Gauge(
        'high_risk_members_total',
        'Total number of high risk predictions'
    )
except ValueError:
    high_risk_gauge = REGISTRY._names_to_collectors.get('high_risk_members_total')

try:
    low_risk_gauge = Gauge(
        'low_risk_members_total',
        'Total number of low risk predictions'
    )
except ValueError:
    low_risk_gauge = REGISTRY._names_to_collectors.get('low_risk_members_total')

# Counters to track totals
high_risk_count = 0
low_risk_count = 0

# Define input schema
class MemberData(BaseModel):
    age: int = Field(ge=18, le=85)
    gender: int = Field(ge=0, le=1)
    claim_count_90days: int = Field(ge=0)
    er_visits_6months: int = Field(ge=0)
    total_claim_cost: float = Field(ge=0, allow_inf_nan=False)
    medication_count: int = Field(ge=0)
    has_diabetes: int = Field(ge=0, le=1)
    has_hypertension: int = Field(ge=0, le=1)
    has_copd: int = Field(ge=0, le=1)
    high_er_usage: int = Field(ge=0, le=1)
    high_claim_frequency: int = Field(ge=0, le=1)
    high_cost_member: int = Field(ge=0, le=1)
    multiple_chronic: int = Field(ge=0, le=1)
    high_medication_burden: int = Field(ge=0, le=1)
    risk_indicator: int = Field(ge=0)

@app.get("/")
def home():
    return {"message": "Healthcare Risk Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: MemberData):
    global high_risk_count, low_risk_count

    start_time = time.time()

    input_df = pd.DataFrame([data.model_dump()])
    risk_score = model.predict_proba(input_df)[0][1]
    risk_label = int(model.predict(input_df)[0])

    latency = time.time() - start_time

    # Update Prometheus metrics
    latency_histogram.observe(latency)

    if risk_label == 1:
        high_risk_count += 1
        prediction_counter.labels(risk_category="high_risk").inc()
        high_risk_gauge.set(high_risk_count)
    else:
        low_risk_count += 1
        prediction_counter.labels(risk_category="low_risk").inc()
        low_risk_gauge.set(low_risk_count)

    return {
        "risk_score": round(float(risk_score), 4),
        "risk_label": risk_label,
        "risk_category": "High Risk" if risk_label == 1 else "Low Risk",
        "scope": "Synthetic demonstration; not for clinical decisions",
        "latency_ms": round(latency * 1000, 2)
    }

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)