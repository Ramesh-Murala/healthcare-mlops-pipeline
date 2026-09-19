# Healthcare MLOps Simulation

A local MLOps demonstration using **synthetic insurance-claims data**: generate records, transform features, compare models, serve predictions, expose Prometheus metrics, and run batch scoring.

This is an independent portfolio simulation. It is not affiliated with an insurer, clinically validated, deployed to production, or certified as HIPAA compliant. Do not use it for patient-care or insurance decisions.

## What is implemented

| Stage | Implementation |
|---|---|
| Data | 1,000 synthetic records, seeded Python/NumPy/Faker generation |
| Transformation | Hash synthetic member IDs, remove ZIP codes, derive risk features |
| Training | Six model configurations evaluated against the same synthetic risk label |
| Tracking | Local MLflow experiment, metrics, model artifacts |
| Serving | FastAPI `/predict`, `/health`, `/metrics` |
| Monitoring | Prediction count and latency metrics; a Prometheus scrape configuration |
| Batch scoring | Sequential Python tasks, dated CSV output, JSON run summary |
| CI | Fresh data generation, training, batch run, tests, Docker build and health check |

The six configurations do not represent six independently validated clinical use cases. The target is generated from rules in `src/generate_data.py`; good held-out scores mainly show that models approximate those rules.

## Reproduce from a clean checkout

Python 3.11 or 3.12; run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/generate_data.py
python src/anonymize.py
python src/features.py
python src/validate.py
python src/train.py
python pipelines/nightly_pipeline.py
python -m pytest -q
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Training exports `api/model.pkl` for serving and writes `reports/synthetic_evaluation.json`. The random-forest baseline is selected in advance for export; test-set results are for comparison, not model selection. The other configurations are logged to MLflow. Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` to inspect runs.

## Docker

After running the data and training steps above:

```bash
docker build -t healthcare-risk-api -f api/Dockerfile .
docker run --rm -p 8000:8000 healthcare-risk-api
```

Open `http://localhost:8000/docs`. Generate the model and build the image with matching scikit-learn dependencies; pickle artifacts are version-dependent and must only come from trusted training runs. `MODEL_PATH` can override the model location.

## Evidence and limitations

- [Synthetic evaluation](reports/synthetic_evaluation.json) records the split, configurations, and scores. Dependency versions can affect exact results.
- Tests reject invalid data, check prediction/metrics endpoints, and ensure batch failures propagate to schedulers.
- The batch script is manually executable; there is no deployed Airflow scheduler or nightly schedule.
- Prometheus metrics are implemented. Grafana dashboards, drift detection, alerts, cloud deployment, and model promotion are not included.
- Hashing IDs and dropping a column demonstrate transformations; they do not establish anonymization or regulatory compliance.
- The API accepts precomputed features. A production service needs shared feature computation, consistency checks, authentication, model lineage, and robust multi-worker monitoring.

Generated databases, model binaries, logs, caches, and data are excluded from Git. Recreate them with the commands above; earlier versions remain in Git history.
