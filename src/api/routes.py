from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import ModelBundle, get_model, get_monitor
from src.api.monitoring import PredictionMonitor
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    model: ModelBundle = Depends(get_model),
    monitor: PredictionMonitor = Depends(get_monitor),
):
    t0 = time.perf_counter()
    try:
        results = model.predict_aspects(req.text, req.aspects)
    except Exception as e:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=f"inference error: {type(e).__name__}")

    latency_ms = (time.perf_counter() - t0) * 1000
    monitor.record(latency_ms, results)

    return PredictResponse(
        results=results,
        model_version=model.model_version,
        inference_time_ms=round(latency_ms, 2),
    )


@router.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(
    req: BatchPredictRequest,
    model: ModelBundle = Depends(get_model),
    monitor: PredictionMonitor = Depends(get_monitor),
):
    reviews = [(r.text, r.aspects) for r in req.reviews]

    t0 = time.perf_counter()
    try:
        results = model.predict_batch(reviews)
    except Exception as e:
        logger.exception("batch_predict failed")
        raise HTTPException(status_code=500, detail=f"inference error: {type(e).__name__}")

    latency_ms = (time.perf_counter() - t0) * 1000

    flat = [p for sublist in results for p in sublist]
    monitor.record(latency_ms, flat)

    return BatchPredictResponse(
        results=results,
        model_version=model.model_version,
        inference_time_ms=round(latency_ms, 2),
        total_reviews=len(reviews),
    )


@router.get("/health", response_model=HealthResponse)
def health(model: ModelBundle = Depends(get_model)):
    return HealthResponse(
        status="ok",
        model_loaded=True,
        calibration_loaded=model.calibration_loaded,
        model_version=model.model_version,
    )


@router.get("/metrics", response_model=MetricsResponse)
def metrics(monitor: PredictionMonitor = Depends(get_monitor)):
    return monitor.snapshot()
