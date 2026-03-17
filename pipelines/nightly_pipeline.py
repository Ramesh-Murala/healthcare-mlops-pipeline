import pandas as pd
import pickle
import logging
import os
import json
from datetime import datetime

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/nightly_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

print("=" * 60)
print(f"  BCBS Nightly Risk Scoring Pipeline")
print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ─────────────────────────────────────────
# TASK 1 - Load Raw Data
# ─────────────────────────────────────────
def task_load_data():
    print("\n[Task 1/6] Loading member claims data...")
    logging.info("Task 1 started - Loading raw data")
    
    df = pd.read_csv("data/raw/member_claims.csv")
    logging.info(f"Loaded {len(df)} member records")
    print(f"  ✅ Loaded {len(df)} member records")
    return df

# ─────────────────────────────────────────
# TASK 2 - PHI Anonymization
# ─────────────────────────────────────────
def task_anonymize(df):
    print("\n[Task 2/6] Running PHI anonymization...")
    logging.info("Task 2 started - PHI anonymization")
    
    import hashlib
    df = df.copy()
    df["member_id"] = df["member_id"].apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()[:12]
    )
    df = df.drop(columns=["zip_code"])
    
    logging.info("PHI anonymization complete - zip_code removed, member IDs hashed")
    print(f"  ✅ PHI anonymization complete")
    return df

# ─────────────────────────────────────────
# TASK 3 - Data Validation
# ─────────────────────────────────────────
def task_validate(df):
    print("\n[Task 3/6] Running data validation checks...")
    logging.info("Task 3 started - Data validation")
    
    errors = []
    
    if df[(df["age"] < 18) | (df["age"] > 85)].shape[0] > 0:
        errors.append("Invalid age values found")
    if df[df["total_claim_cost"] < 0].shape[0] > 0:
        errors.append("Negative claim costs found")
    if df[df["er_visits_6months"] < 0].shape[0] > 0:
        errors.append("Negative ER visits found")
    if any(df[key].isnull().any() for key in ["member_id", "age", "total_claim_cost"]):
        errors.append("Missing values found")

    if errors:
        logging.error(f"Validation failed: {errors}")
        raise ValueError(f"Data validation failed: {errors}")
    
    logging.info("All validation checks passed")
    print(f"  ✅ All validation checks passed")
    return df

# ─────────────────────────────────────────
# TASK 4 - Feature Engineering
# ─────────────────────────────────────────
def task_feature_engineering(df):
    print("\n[Task 4/6] Running feature engineering...")
    logging.info("Task 4 started - Feature engineering")
    
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 35, 50, 65, 100],
        labels=["young", "middle", "senior", "elderly"]
    )
    df["high_er_usage"] = (df["er_visits_6months"] > 2).astype(int)
    df["high_claim_frequency"] = (df["claim_count_90days"] > 10).astype(int)
    df["high_cost_member"] = (df["total_claim_cost"] > 50000).astype(int)
    df["multiple_chronic"] = (
        df["has_diabetes"] + df["has_hypertension"] + df["has_copd"] > 1
    ).astype(int)
    df["high_medication_burden"] = (df["medication_count"] > 6).astype(int)
    df["risk_indicator"] = (
        df["high_er_usage"] + df["high_claim_frequency"] +
        df["high_cost_member"] + df["multiple_chronic"] +
        df["high_medication_burden"]
    )
    df["gender"] = df["gender"].map({"M": 1, "F": 0})
    df = df.drop(columns=["diagnosis_codes", "age_group"])
    
    logging.info(f"Feature engineering complete - {len(df.columns)} features")
    print(f"  ✅ Feature engineering complete - {len(df.columns)} features")
    return df

# ─────────────────────────────────────────
# TASK 5 - Run Batch Inference
# ─────────────────────────────────────────
def task_run_inference(df):
    print("\n[Task 5/6] Running batch inference on all members...")
    logging.info("Task 5 started - Batch inference")
    
    with open("api/model.pkl", "rb") as f:
        model = pickle.load(f)
    
    feature_cols = [
        "age", "gender", "claim_count_90days", "er_visits_6months",
        "total_claim_cost", "medication_count", "has_diabetes",
        "has_hypertension", "has_copd", "high_er_usage",
        "high_claim_frequency", "high_cost_member", "multiple_chronic",
        "high_medication_burden", "risk_indicator"
    ]
    
    X = df[feature_cols]
    df["risk_score"] = model.predict_proba(X)[:, 1].round(4)
    df["risk_label"] = model.predict(X)
    df["risk_category"] = df["risk_label"].map({1: "High Risk", 0: "Low Risk"})
    
    high_risk = df[df["risk_label"] == 1]
    low_risk = df[df["risk_label"] == 0]
    
    logging.info(f"Inference complete - High risk: {len(high_risk)}, Low risk: {len(low_risk)}")
    print(f"  ✅ Inference complete")
    print(f"     High Risk Members : {len(high_risk)}")
    print(f"     Low Risk Members  : {len(low_risk)}")
    return df

# ─────────────────────────────────────────
# TASK 6 - Save Results & Send Alert
# ─────────────────────────────────────────
def task_save_and_notify(df):
    print("\n[Task 6/6] Saving results and sending notification...")
    logging.info("Task 6 started - Save results")
    
    os.makedirs("data/processed", exist_ok=True)
    output_path = f"data/processed/risk_scores_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output_path, index=False)
    
    high_risk_count = df[df["risk_label"] == 1].shape[0]
    low_risk_count = df[df["risk_label"] == 0].shape[0]
    avg_score = df["risk_score"].mean().round(4)
    
    summary = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_members_scored": len(df),
        "high_risk_members": high_risk_count,
        "low_risk_members": low_risk_count,
        "average_risk_score": float(avg_score),
        "output_file": output_path,
        "status": "SUCCESS"
    }
    
    with open("logs/pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Pipeline complete - {summary}")
    
    print(f"  ✅ Results saved to {output_path}")
    print(f"  ✅ Pipeline summary saved to logs/pipeline_summary.json")
    
    return summary

# ─────────────────────────────────────────
# MAIN PIPELINE RUNNER
# ─────────────────────────────────────────
def run_pipeline():
    start_time = datetime.now()
    
    try:
        df = task_load_data()
        df = task_anonymize(df)
        df = task_validate(df)
        df = task_feature_engineering(df)
        df = task_run_inference(df)
        summary = task_save_and_notify(df)
        
        end_time = datetime.now()
        duration = (end_time - start_time).seconds
        
        print("\n" + "=" * 60)
        print(f"  ✅ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"  Total Members Scored : {summary['total_members_scored']}")
        print(f"  High Risk Members    : {summary['high_risk_members']}")
        print(f"  Low Risk Members     : {summary['low_risk_members']}")
        print(f"  Average Risk Score   : {summary['average_risk_score']}")
        print(f"  Duration             : {duration} seconds")
        print("=" * 60)
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        print(f"\n❌ PIPELINE FAILED: {str(e)}")

if __name__ == "__main__":
    run_pipeline()