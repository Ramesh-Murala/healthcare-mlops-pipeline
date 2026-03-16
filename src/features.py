import pandas as pd
import numpy as np
import os

def engineer_features(input_path, output_path):
    print("Starting feature engineering...")
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records")

    # Feature 1 - Age group buckets
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 35, 50, 65, 100],
        labels=["young", "middle", "senior", "elderly"]
    )

    # Feature 2 - High ER usage flag
    df["high_er_usage"] = (df["er_visits_6months"] > 2).astype(int)

    # Feature 3 - High claim frequency flag
    df["high_claim_frequency"] = (df["claim_count_90days"] > 10).astype(int)

    # Feature 4 - High cost flag
    df["high_cost_member"] = (df["total_claim_cost"] > 50000).astype(int)

    # Feature 5 - Multiple chronic conditions flag
    df["multiple_chronic"] = (
        df["has_diabetes"] + df["has_hypertension"] + df["has_copd"] > 1
    ).astype(int)

    # Feature 6 - Medication burden flag
    df["high_medication_burden"] = (df["medication_count"] > 6).astype(int)

    # Feature 7 - Overall risk indicator score
    df["risk_indicator"] = (
        df["high_er_usage"] +
        df["high_claim_frequency"] +
        df["high_cost_member"] +
        df["multiple_chronic"] +
        df["high_medication_burden"]
    )

    # Drop columns not needed for training
    df = df.drop(columns=["diagnosis_codes"])

    # Save featured dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Feature engineering complete: {len(df)} records")
    print(f"✅ New features added: age_group, high_er_usage, high_claim_frequency,")
    print(f"   high_cost_member, multiple_chronic, high_medication_burden, risk_indicator")
    print(f"✅ Saved to {output_path}")
    print(df.head())

if __name__ == "__main__":
    engineer_features(
        input_path="data/processed/member_claims_anonymized.csv",
        output_path="data/processed/member_claims_featured.csv"
    )