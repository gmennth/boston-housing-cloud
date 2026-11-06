from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

SAMPLE = {"CRIM": 0.00632, "ZN": 18, "INDUS": 2.31, "CHAS": 0, "NOX": 0.538, "RM": 6.575, "AGE": 65.2,
          "DIS": 4.09, "RAD": 1, "TAX": 296, "PTRATIO": 15.3, "B": 396.9, "LSTAT": 4.98}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_ok():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    data = r.json()
    # On teste une plage plausible, jamais une valeur exacte : elle change avec le modèle
    assert 0 < data["prediction_k_usd"] < 60
    assert data["range_k_usd"][0] < data["prediction_k_usd"] < data["range_k_usd"][1]
    # Le prix en dollars est cohérent avec le prix en milliers (arrondis près)
    assert abs(data["prediction_usd"] - data["prediction_k_usd"] * 1000) < 10


def test_predict_invalid_value():
    assert client.post("/predict", json={**SAMPLE, "RM": -1}).status_code == 422


def test_predict_missing_field():
    incomplet = {k: v for k, v in SAMPLE.items() if k != "LSTAT"}
    assert client.post("/predict", json=incomplet).status_code == 422


def test_model_info():
    info = client.get("/model-info").json()
    assert "metrics" in info
    assert "feature_ranges" in info
    assert set(info["feature_ranges"]) == set(SAMPLE)
