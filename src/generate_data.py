import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker()
np.random.seed(42)
random.seed(42)

NUM_MEMBERS = 1000

diagnosis_codes = [
    "E11.9",   # Type 2 Diabetes
    "I10",     # Hypertension
    "J44.1",   # COPD
    "I25.10",  # Coronary Artery Disease
    "E78.5",   # High Cholesterol
    "F32.9",   # Depression
    "M54.5",   # Back Pain
    "N18.3",   # Chronic Kidney Disease
]

records = []

for _ in range(NUM_MEMBERS):
    member_id = fake.uuid4()
    age = random.randint(18, 85)
    gender = random.choice(["M", "F"])
    zip_code = fake.zipcode()
    num_conditions = random.randint(0, 4)
    conditions = random.sample(diagnosis_codes, num_conditions)
    claim_count_90days = random.randint(0, 20)
    er_visits_6months = random.randint(0, 5)
    total_claim_cost = round(random.uniform(500, 80000), 2)
    medication_count = random.randint(0, 10)
    has_diabetes = int("E11.9" in conditions)
    has_hypertension = int("I10" in conditions)
    has_copd = int("J44.1" in conditions)

    risk_score = (
        (age > 60) * 1 +
        (er_visits_6months > 2) * 2 +
        (num_conditions > 2) * 2 +
        (claim_count_90days > 10) * 1 +
        (total_claim_cost > 50000) * 2
    )
    risk_label = 1 if risk_score >= 3 else 0

    records.append({
        "member_id": member_id,
        "age": age,
        "gender": gender,
        "zip_code": zip_code,
        "diagnosis_codes": "|".join(conditions),
        "claim_count_90days": claim_count_90days,
        "er_visits_6months": er_visits_6months,
        "total_claim_cost": total_claim_cost,
        "medication_count": medication_count,
        "has_diabetes": has_diabetes,
        "has_hypertension": has_hypertension,
        "has_copd": has_copd,
        "risk_label": risk_label
    })

df = pd.DataFrame(records)
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/member_claims.csv", index=False)
print(f"Dataset created: {len(df)} members")
print(f"High risk members: {df['risk_label'].sum()} ({df['risk_label'].mean()*100:.1f}%)")
print(df.head())