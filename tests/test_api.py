import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MODEL = Path("models/best_model.pkl")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="model not pulled")

CLASSES = {"Cancelled", "Delivered", "Returned"}


@pytest.fixture(scope="module")
def client():
    from app import app
    return TestClient(app)


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert set(r.json()["classes"]) == CLASSES


def test_predict(client):
    payload = json.loads(Path("tests/sample_payload.json").read_text())
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_return_status"] in CLASSES
    assert set(body["class_probabilities"]) == CLASSES


def test_predict_rejects_bad_input(client):
    r = client.post("/predict", json={"order_year": "not-a-number"})
    assert r.status_code == 422
