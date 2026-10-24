import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_data_validation_rejects_negative_cost(tmp_path):
    validate = load("src/validate.py", "validation")
    path = tmp_path / "data.csv"
    pd.DataFrame([dict(member_id="synthetic", age=40, total_claim_cost=-1, er_visits_6months=0, risk_label=0)]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        validate.validate_data(path)


def test_batch_failure_propagates_to_scheduler(monkeypatch):
    pipeline = load("pipelines/nightly_pipeline.py", "nightly")
    def fail():
        raise FileNotFoundError("missing synthetic input")
    monkeypatch.setattr(pipeline, "task_load_data", fail)
    with pytest.raises(FileNotFoundError):
        pipeline.run_pipeline()


def batch_row():
    return dict(
        member_id="synthetic", age=40, gender="F", zip_code="00000",
        diagnosis_codes="", claim_count_90days=2, er_visits_6months=0,
        total_claim_cost=1_000, medication_count=1, has_diabetes=0,
        has_hypertension=0, has_copd=0, risk_label=0,
    )


@pytest.mark.parametrize("column", [
    "age", "claim_count_90days", "er_visits_6months", "total_claim_cost",
    "medication_count", "has_diabetes", "has_hypertension", "has_copd",
    "risk_label",
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "not-a-number"])
def test_batch_validation_rejects_invalid_numeric_model_inputs(column, value):
    pipeline = load("pipelines/nightly_pipeline.py", f"nightly_invalid_{column}")
    row = batch_row()
    row[column] = value

    with pytest.raises(ValueError, match="numeric model inputs must be present and finite"):
        pipeline.task_validate(pd.DataFrame([row]))


def test_batch_validation_coerces_finite_numeric_strings():
    pipeline = load("pipelines/nightly_pipeline.py", "nightly_numeric_strings")
    row = batch_row()
    row.update(age="40", claim_count_90days="2", total_claim_cost="1000.50")

    validated = pipeline.task_validate(pd.DataFrame([row]))

    assert validated.loc[0, "age"] == 40
    assert validated.loc[0, "claim_count_90days"] == 2
    assert validated.loc[0, "total_claim_cost"] == 1000.5


def test_batch_validation_rejects_missing_numeric_model_input():
    pipeline = load("pipelines/nightly_pipeline.py", "nightly_missing_column")
    row = batch_row()
    row.pop("medication_count")

    with pytest.raises(ValueError, match="numeric model inputs must be present and finite"):
        pipeline.task_validate(pd.DataFrame([row]))


@pytest.mark.parametrize("column", ["age", "total_claim_cost", "er_visits_6months", "risk_label"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "not-a-number"])
def test_data_validation_rejects_non_finite_or_nonnumeric_values(tmp_path, column, value):
    validate = load("src/validate.py", "validation_numeric")
    row = dict(member_id="synthetic", age=40, total_claim_cost=0,
               er_visits_6months=0, risk_label=0)
    row[column] = value
    path = tmp_path / "invalid.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="numeric values must be finite"):
        validate.validate_data(path)


def test_data_validation_accepts_finite_boundary_values(tmp_path):
    validate = load("src/validate.py", "validation_boundaries")
    path = tmp_path / "valid.csv"
    pd.DataFrame([
        dict(member_id="synthetic-1", age=18, total_claim_cost=0,
             er_visits_6months=0, risk_label=0),
        dict(member_id="synthetic-2", age=85, total_claim_cost=50_000,
             er_visits_6months=3, risk_label=1),
    ]).to_csv(path, index=False)
    validate.validate_data(path)


def test_api_predict_and_metrics():
    api = load("api/app.py", "risk_api")
    df = pd.read_csv("data/processed/member_claims_featured.csv")
    sample = df.iloc[0].drop(["member_id", "age_group", "risk_label"]).to_dict()
    sample["gender"] = {"M": 1, "F": 0}[sample["gender"]]
    with TestClient(api.app) as client:
        response = client.post("/predict", json=sample)
        assert response.status_code == 200
        assert 0 <= response.json()["risk_score"] <= 1
        assert "not for clinical" in response.json()["scope"]
        assert "predictions_total" in client.get("/metrics").text
        sample["age"] = -1
        assert client.post("/predict", json=sample).status_code == 422


def test_api_derives_risk_features_instead_of_trusting_client_values():
    api = load("api/app.py", "risk_api_features")
    raw = dict(age=70, gender=1, claim_count_90days=12,
               er_visits_6months=4, total_claim_cost=60_000,
               medication_count=8, has_diabetes=1,
               has_hypertension=1, has_copd=0)
    expected = api.model_input(api.MemberData(**raw)).iloc[0]
    assert expected["risk_indicator"] == 5

    # Legacy clients may still send derived fields. Pydantic ignores them,
    # and the service recomputes the authoritative values from raw inputs.
    tampered = {**raw, "high_er_usage": 0, "high_claim_frequency": 0,
                "high_cost_member": 0, "multiple_chronic": 0,
                "high_medication_burden": 0, "risk_indicator": 0}
    actual = api.model_input(api.MemberData(**tampered)).iloc[0]
    assert actual[api.MODEL_FEATURES].equals(expected[api.MODEL_FEATURES])
