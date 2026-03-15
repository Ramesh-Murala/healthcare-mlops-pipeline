import pandas as pd
import hashlib
import logging
import os
from datetime import datetime

# Setup audit logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/audit.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def hash_member_id(member_id):
    """Replace real member ID with a hashed version"""
    return hashlib.sha256(member_id.encode()).hexdigest()[:12]

def anonymize_data(input_path, output_path):
    logging.info(f"Starting PHI anonymization - Input: {input_path}")
    
    df = pd.read_csv(input_path)
    logging.info(f"Loaded {len(df)} member records")

    # Remove direct identifiers
    df = df.drop(columns=["zip_code"])
    logging.info("Removed zip_code (PHI)")

    # Hash member IDs
    df["member_id"] = df["member_id"].apply(hash_member_id)
    logging.info("Hashed all member IDs")

    # Save anonymized data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logging.info(f"Anonymized data saved to {output_path}")
    logging.info(f"Anonymization complete - {len(df)} records processed")
    
    print(f"✅ Anonymization complete: {len(df)} records processed")
    print(f"✅ Audit log saved to logs/audit.log")
    print(df.head())

if __name__ == "__main__":
    anonymize_data(
        input_path="data/raw/member_claims.csv",
        output_path="data/processed/member_claims_anonymized.csv"
    )