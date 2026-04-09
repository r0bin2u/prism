import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_model, get_monitor
from src.api.main import app
from src.api.monitoring import PredictionMonitor
from src.api.schemas import AspectPrediction


SENTIMENTS = ("positive", "negative", "neutral")
DEFAULT_ASPECTS = ["food", "service", "ambience", "price", "anecdotes/miscellaneous"]


class FakeBundle:
    model_version = "test-v0"
    calibration_loaded = True
    calibration_T = 1.5
    default_aspects = DEFAULT_ASPECTS

    def _pred(self, aspect):
        if "food" in aspect:
            probs = {"positive": 0.7, "negative": 0.1, "neutral": 0.2}
        else:
            probs = {"positive": 0.2, "negative": 0.7, "neutral": 0.1}
        sentiment = max(probs, key=probs.get)
        return AspectPrediction(
            aspect=aspect,
            sentiment=sentiment,
            confidence=probs[sentiment],
            probabilities=probs,
        )

    def predict_aspects(self, text, aspects=None):
        asps = list(aspects) if aspects else list(self.default_aspects)
        return [self._pred(a) for a in asps]

    def predict_batch(self, reviews):
        return [self.predict_aspects(t, a) for t, a in reviews]


@pytest.fixture
def fake_bundle():
    return FakeBundle()


@pytest.fixture
def fake_monitor():
    return PredictionMonitor(maxlen=100, recent_window=10, drift_threshold=0.15)


@pytest.fixture
def client(fake_bundle, fake_monitor):
    app.dependency_overrides[get_model] = lambda: fake_bundle
    app.dependency_overrides[get_monitor] = lambda: fake_monitor
    yield TestClient(app)
    app.dependency_overrides.clear()
