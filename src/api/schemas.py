from typing import Literal
from pydantic import BaseModel, Field, field_validator


VALID_ASPECTS = {"food", "service", "ambience", "price", "anecdotes/miscellaneous"}
VALID_SENTIMENTS = ("positive", "negative", "neutral")

Sentiment = Literal["positive", "negative", "neutral"]

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1024)
    aspects: list[str] | None = None

    @field_validator("aspects")
    @classmethod
    def _check_aspects(cls, v):
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("aspects cannot be empty list, pass null to use defaults")
        unknown = set(v) - VALID_ASPECTS
        if unknown:
            raise ValueError(f"unknown aspects: {sorted(unknown)}; valid: {sorted(VALID_ASPECTS)}")
        return v


class AspectPrediction(BaseModel):
    aspect: str
    sentiment: Sentiment
    confidence: float
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    results: list[AspectPrediction]
    model_version: str
    inference_time_ms: float


class BatchPredictRequest(BaseModel):
    reviews: list[PredictRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    results: list[list[AspectPrediction]]
    model_version: str
    inference_time_ms: float
    total_reviews: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    calibration_loaded: bool
    model_version: str


class MetricsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    avg_confidence: float
    recent_negative_ratio: float
    overall_negative_ratio: float
    drift_alert: bool
    alert_reason: str | None
