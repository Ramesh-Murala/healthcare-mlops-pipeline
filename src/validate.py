import pandas as pd
import os

def validate_data(input_path):
    print("Starting data validation...")

    df = pd.read_csv(input_path)
    errors = []

    # Check 1 - Age must be between 18 and 85
    invalid_age = df[(df["age"] < 18) | (df["age"] > 85)]
    if len(invalid_age) > 0:
        errors.append(f"❌ Invalid age values: {len(invalid_age)} records")
    else:
        print("✅ Age range check passed")

    # Check 2 - Claim cost cannot be negative
    invalid_cost = df[df["total_claim_cost"] < 0]
    if len(invalid_cost) > 0:
        errors.append(f"❌ Negative claim costs: {len(invalid_cost)} records")
    else:
        print("✅ Claim cost check passed")

    # Check 3 - ER visits cannot be negative
    invalid_er = df[df["er_visits_6months"] < 0]
    if len(invalid_er) > 0:
        errors.append(f"❌ Negative ER visits: {len(invalid_er)} records")
    else:
        print("✅ ER visits check passed")

    # Check 4 - Risk label must be 0 or 1
    invalid_label = df[~df["risk_label"].isin([0, 1])]
    if len(invalid_label) > 0:
        errors.append(f"❌ Invalid risk labels: {len(invalid_label)} records")
    else:
        print("✅ Risk label check passed")

    # Check 5 - No missing values in key columns
    key_columns = ["member_id", "age", "total_claim_cost", "risk_label"]
    missing = df[key_columns].isnull().sum()
    if missing.any():
        errors.append(f"❌ Missing values found: {missing[missing > 0].to_dict()}")
    else:
        print("✅ No missing values check passed")

    # Final result
    if errors:
        print("\n⚠️  Validation failed with errors:")
        for e in errors:
            print(e)
        raise ValueError("Data validation failed — pipeline stopped.")
    else:
        print(f"\n✅ All validation checks passed — {len(df)} records ready for training")

if __name__ == "__main__":
    validate_data("data/processed/member_claims_featured.csv")