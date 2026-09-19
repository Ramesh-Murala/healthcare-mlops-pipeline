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
