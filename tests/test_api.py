def test_predict_returns_200(client):
    r = client.post("/predict", json={"text": "the food was great"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 5
    assert body["model_version"] == "test-v0"
    assert body["inference_time_ms"] >= 0


def test_predict_rejects_invalid_aspect(client):
    r = client.post("/predict", json={"text": "x", "aspects": ["totally_made_up"]})
    assert r.status_code == 422


def test_batch_predict_multiple_reviews(client):
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


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["calibration_loaded"] is True
    assert body["model_version"] == "test-v0"


def test_metrics_reflects_predicts(client):
    client.post("/predict", json={"text": "good food", "aspects": ["food"]})
    client.post("/predict", json={"text": "bad service", "aspects": ["service"]})
    metrics = client.get("/metrics").json()
    assert metrics["total_requests"] == 2
    assert metrics["avg_latency_ms"] > 0
