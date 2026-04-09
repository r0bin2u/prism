from __future__ import annotations

import time
from collections import deque

from src.api.schemas import AspectPrediction, MetricsResponse


# No lock here on purpose. FastAPI runs sync endpoints in a threadpool so
# multiple workers may call record/snapshot concurrently, but:
#   1. deque.append is GIL-atomic in CPython, the structure can't corrupt.
#   2. snapshot() copies history to a local list once, then computes outside
#      any shared state. Worst case: a single /metrics call sees a count that
#      lags by one request. The next call corrects it.
#   3. Monitoring stats are not transactional data. A 0.001 drift in a
#      reported ratio has zero impact on business decisions.
# Adding a Lock would buy strong consistency we don't need at the cost of
# code complexity and lock contention on a hot path. If this monitor ever
# fed a payment counter or alert SLA, we'd revisit.
class PredictionMonitor:
    def __init__(
        self,
        maxlen: int = 1000,
        recent_window: int = 100,
        drift_threshold: float = 0.15,
        baseline_negative_ratio: float | None = None,
    ):
        self.history: deque[dict] = deque(maxlen=maxlen)
        self.recent_window = recent_window
        self.drift_threshold = drift_threshold
        self.baseline_negative_ratio = baseline_negative_ratio

    def record(self, latency_ms: float, predictions: list[AspectPrediction]) -> None:
        if not predictions:
            return
        avg_conf = sum(p.confidence for p in predictions) / len(predictions)
        neg_count = sum(1 for p in predictions if p.sentiment == "negative")
        self.history.append({
            "ts": time.time(),
            "latency_ms": float(latency_ms),
            "avg_confidence": avg_conf,
            "negative_count": neg_count,
            "total_count": len(predictions),
        })

    def snapshot(self) -> MetricsResponse:
        history = list(self.history)

        if not history:
            return MetricsResponse(
                total_requests=0,
                avg_latency_ms=0.0,
                avg_confidence=0.0,
                recent_negative_ratio=0.0,
                overall_negative_ratio=0.0,
                drift_alert=False,
                alert_reason=None,
            )

        total = len(history)
        avg_latency = sum(r["latency_ms"] for r in history) / total
        avg_conf = sum(r["avg_confidence"] for r in history) / total

        overall_neg = sum(r["negative_count"] for r in history)
        overall_total = sum(r["total_count"] for r in history)
        overall_neg_ratio = overall_neg / overall_total if overall_total else 0.0

        recent = history[-self.recent_window:]
        recent_neg = sum(r["negative_count"] for r in recent)
        recent_total = sum(r["total_count"] for r in recent)
        recent_neg_ratio = recent_neg / recent_total if recent_total else 0.0

        baseline = (
            self.baseline_negative_ratio
            if self.baseline_negative_ratio is not None
            else overall_neg_ratio
        )
        drift = abs(recent_neg_ratio - baseline)
        alert = drift > self.drift_threshold and len(recent) >= self.recent_window

        reason = None
        if alert:
            baseline_label = "fixed baseline" if self.baseline_negative_ratio is not None else "overall"
            reason = (
                f"recent_negative_ratio={recent_neg_ratio:.3f} differs from "
                f"{baseline_label}={baseline:.3f} by {drift:.3f} "
                f"(threshold={self.drift_threshold})"
            )

        return MetricsResponse(
            total_requests=total,
            avg_latency_ms=round(avg_latency, 2),
            avg_confidence=round(avg_conf, 4),
            recent_negative_ratio=round(recent_neg_ratio, 4),
            overall_negative_ratio=round(overall_neg_ratio, 4),
            drift_alert=alert,
            alert_reason=reason,
        )
