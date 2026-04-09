import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["calibration_loaded"] is True
    assert body["model_version"] == "test-v1"


def test_predict_default_aspects(client):
    r = client.post("/predict", json={"text": "the food was great"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 5
    assert body["model_version"] == "test-v1"
    assert body["inference_time_ms"] >= 0


def test_predict_specific_aspects(client):
    r = client.post("/predict", json={"text": "x", "aspects": ["food", "service"]})
    assert r.status_code == 200
    body = r.json()
    assert [p["aspect"] for p in body["results"]] == ["food", "service"]
    food = body["results"][0]
    assert food["sentiment"] == "positive"
    assert food["probabilities"]["positive"] == pytest.approx(0.7)


def test_predict_empty_text_rejected(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_text_too_long_rejected(client):
    r = client.post("/predict", json={"text": "x" * 2000})
    assert r.status_code == 422


def test_predict_unknown_aspect_rejected(client):
    r = client.post("/predict", json={"text": "x", "aspects": ["totally_made_up"]})
    assert r.status_code == 422


def test_predict_empty_aspect_list_rejected(client):
    r = client.post("/predict", json={"text": "x", "aspects": []})
    assert r.status_code == 422


def test_batch_predict(client):
    r = client.post("/batch_predict", json={
        "reviews": [
            {"text": "great food"},
            {"text": "ok service", "aspects": ["service"]},
        ],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["total_reviews"] == 2
    assert len(body["results"]) == 2
    assert len(body["results"][0]) == 5
    assert len(body["results"][1]) == 1


def test_batch_predict_too_many_rejected(client):
    r = client.post("/batch_predict", json={"reviews": [{"text": "x"}] * 101})
    assert r.status_code == 422


def test_metrics_empty(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["total_requests"] == 0
    assert body["drift_alert"] is False


def test_metrics_after_predict(client):
    client.post("/predict", json={"text": "great food", "aspects": ["food"]})
    client.post("/predict", json={"text": "more food", "aspects": ["food"]})
    body = client.get("/metrics").json()
    assert body["total_requests"] == 2
    assert body["avg_confidence"] > 0
    assert body["overall_negative_ratio"] == 0.0
