from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import load_model_bundle
from src.api.monitoring import PredictionMonitor
from src.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint_dir = PROJECT_ROOT / "models" / "best_model"
    logger.info("loading model bundle from %s", checkpoint_dir)

    bundle = load_model_bundle(checkpoint_dir, CONFIG)
    app.state.model_bundle = bundle

    api_cfg = CONFIG["api"]
    monitor = PredictionMonitor(
        maxlen=api_cfg["monitor_maxlen"],
        recent_window=api_cfg["monitor_recent_window"],
        drift_threshold=api_cfg["monitor_drift_threshold"],
    )
    app.state.monitor = monitor

    logger.info("startup done, model_version=%s", bundle.model_version)
    yield
    logger.info("shutting down")


app = FastAPI(
    title="Prism ABSA API",
    description="Aspect-Based Sentiment Analysis served by a distilled DeBERTa model",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "%s %s %d %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    api_cfg = CONFIG["api"]
    uvicorn.run(
        "src.api.main:app",
        host=api_cfg["host"],
        port=api_cfg["port"],
        reload=False,
    )
