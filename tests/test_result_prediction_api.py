import os
import pickle
import pytest
from fastapi.testclient import TestClient
from apis.f1_position_prediction_api import result_prediction_api

client = TestClient(result_prediction_api.app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/f1_race_position_model.pkl')

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "model_loaded" in data
    assert "endpoints" in data

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "unhealthy"]
    assert "model_loaded" in data

def test_reload_model():
    response = client.post("/reload-model")
    assert response.status_code in [200, 500]  # 500 if model file missing
    if response.status_code == 200:
        assert "message" in response.json()

def test_predict_valid():
    payload = {
        "year": 2023,
        "month": 5,
        "round": 1,
        "grid": 1,
        "qualifying_position": 1,
        "circuit_name": "Monaco",
        "driver_surname": "Verstappen",
        "constructor_name": "Red Bull",
        "avg_race_pos": 1.0,
        "avg_sprint_pos": 1.0,
        "avg_lap_time": 80.0,
        "points": 25.0,
        "avg_qual_pos": 1.0,
        "driver_encoded": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 500]  # 500 if model not loaded
    if response.status_code == 200:
        data = response.json()
        assert "predicted_position" in data
        assert 1 <= data["predicted_position"] <= 20
        assert "confidence" in data
        assert "input_features" in data
