from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
DEST = Path(__file__).parent / "assets"
DEST.mkdir(exist_ok=True)
from fastapi.testclient import TestClient
from api.app import app
request = {"age": 32, "gender": 1, "claim_count_90days": 3, "er_visits_6months": 0, "total_claim_cost": 2500.0, "medication_count": 2, "has_diabetes": 0, "has_hypertension": 0, "has_copd": 0}
with TestClient(app) as client:
    result = client.post("/predict", json=request)
    assert result.status_code == 200
    response = result.json()

(DEST / "request.json").write_text(json.dumps(request, indent=2) + "\n")
(DEST / "response.json").write_text(json.dumps(response, indent=2) + "\n")
print(json.dumps(response, indent=2))
