"""Compare six configurations on one synthetic risk label; export a fixed baseline."""
import json
import os
import pickle
import platform
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train():
    df = pd.read_csv("data/processed/member_claims_featured.csv")
    df = df.drop(columns=["member_id", "age_group"])
    df["gender"] = df["gender"].map({"M": 1, "F": 0})
    X, y = df.drop(columns=["risk_label"]), df["risk_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("synthetic_risk_configuration_comparison")
    configs = {
        "random_forest_baseline": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting_baseline": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "random_forest_shallow": RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42),
        "scaled_logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "gradient_boosting_slow": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=42),
        "random_forest_depth7": RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42),
    }
    report = {"scope": "One rule-generated synthetic target; six model configurations, not six clinical outcomes.",
              "environment": {"python": platform.python_version(), "scikit-learn": version("scikit-learn"), "mlflow": version("mlflow")},
              "seed": 42, "train_rows": len(X_train), "test_rows": len(X_test),
              "exported_configuration": "random_forest_baseline", "models": {}}
    for name, model in configs.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            predicted = model.predict(X_test)
            metrics = dict(accuracy=accuracy_score(y_test, predicted),
                           f1=f1_score(y_test, predicted),
                           roc_auc=roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
            mlflow.log_params({"configuration": name, "target": "synthetic_risk_label", "seed": 42})
            mlflow.set_tag("status", "simulation")
            mlflow.log_metrics(metrics)
            # Only serialize models just trained in this process. Artifacts are
            # trusted local pickle data, not a format for untrusted uploads.
            mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(2),
                                    serialization_format="cloudpickle")
            report["models"][name] = metrics
            # Fixed before evaluation; do not select the winner on the test set.
            if name == report["exported_configuration"]:
                Path("api").mkdir(exist_ok=True)
                with Path("api/model.pkl").open("wb") as output:
                    pickle.dump(model, output)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/synthetic_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    train()
