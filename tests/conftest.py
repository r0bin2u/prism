import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.monitoring import PredictionMonitor
from src.api.routes import router
from src.api.schemas import AspectPrediction


class FakeBundle:
    def __init__(self):
        self.model_version = "test-v1"
        self.calibration_loaded = True
        self.default_aspects = ["food", "service", "ambience", "price", "anecdotes/miscellaneous"]

    def _fake_pred(self, aspect):
        if "food" in aspect:
            probs = {"positive": 0.7, "negative": 0.1, "neutral": 0.2}
            sentiment = "positive"
        else:
            probs = {"positive": 0.2, "negative": 0.1, "neutral": 0.7}
            sentiment = "neutral"
        return AspectPrediction(
            aspect=aspect,
            sentiment=sentiment,
            confidence=max(probs.values()),
            probabilities=probs,
        )

    def predict_aspects(self, text, aspects=None):
        asps = list(aspects) if aspects else list(self.default_aspects)
        return [self._fake_pred(a) for a in asps]

    def predict_batch(self, reviews):
        return [self.predict_aspects(t, a) for t, a in reviews]


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    app.state.model_bundle = FakeBundle()
    app.state.monitor = PredictionMonitor(maxlen=100, recent_window=10, drift_threshold=0.15)
    with TestClient(app) as c:
        yield c
