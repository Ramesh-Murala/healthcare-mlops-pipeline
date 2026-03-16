import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("data/processed/member_claims_featured.csv")

# Drop non-numeric columns
df = df.drop(columns=["member_id", "age_group"])

# Encode gender M/F to 1/0
df["gender"] = df["gender"].map({"M": 1, "F": 0})

# Features and target
X = df.drop(columns=["risk_label"])
y = df["risk_label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set MLflow experiment
mlflow.set_experiment("healthcare_risk_prediction")

# Define 6 use cases with different models
experiments = [
    {
        "use_case": "hospitalization_risk",
        "model": RandomForestClassifier(n_estimators=100, random_state=42),
        "model_name": "RandomForest"
    },
    {
        "use_case": "er_visit_likelihood",
        "model": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "model_name": "GradientBoosting"
    },
    {
        "use_case": "chronic_disease_onset",
        "model": RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42),
        "model_name": "RandomForest"
    },
    {
        "use_case": "medication_nonadherence",
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "model_name": "LogisticRegression"
    },
    {
        "use_case": "care_gap_identification",
        "model": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=42),
        "model_name": "GradientBoosting"
    },
    {
        "use_case": "high_cost_claimant",
        "model": RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42),
        "model_name": "RandomForest"
    },
]

print("Starting MLflow training runs...\n")

for exp in experiments:
    with mlflow.start_run(run_name=f"{exp['use_case']}_v1"):

        # Train model
        exp["model"].fit(X_train, y_train)
        y_pred = exp["model"].predict(X_test)
        y_prob = exp["model"].predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy  = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        auc       = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)

        # Log to MLflow
        mlflow.log_param("use_case", exp["use_case"])
        mlflow.log_param("model_type", exp["model_name"])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.set_tag("use_case", exp["use_case"])
        mlflow.set_tag("status", "production")

        # Register model
        mlflow.sklearn.log_model(
            exp["model"],
            artifact_path="model",
            registered_model_name=exp["use_case"]
        )

        print(f"✅ {exp['use_case']}")
        print(f"   Accuracy: {accuracy:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
        print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}\n")

print("🎉 All 6 models trained and logged to MLflow!")
print("Run 'mlflow ui' to view the experiment dashboard.")