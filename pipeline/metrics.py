"""SiteSentry safety monitoring pipeline: timing metrics and stage timers."""

import time
from contextlib import contextmanager


class StageTimer:
    """Context manager that records the elapsed milliseconds of a stage into a PipelineMetrics object."""

    def __init__(self, metrics: "PipelineMetrics", stage: str):
        """Store the metrics sink and stage name to time."""
        self.metrics = metrics
        self.stage = stage
        self._start = 0.0

    def __enter__(self):
        """Start the timer and return self."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Record the elapsed milliseconds, even if the block raised."""
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self.metrics.record(self.stage, elapsed_ms)
        return False


class PipelineMetrics:
    """Accumulate per-stage timing stats across frames."""

    def __init__(self):
        """Initialize empty accumulators."""
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self.frames: int = 0
        self._last_frame_ms: float = 0.0

    @contextmanager
    def stage(self, name: str):
        """Yield a timer that records the block's duration in ms under `name`."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1000.0)

    def record(self, name: str, elapsed_ms: float):
        """Add one timing sample for a stage."""
        self._totals[name] = self._totals.get(name, 0.0) + elapsed_ms
        self._counts[name] = self._counts.get(name, 0) + 1

    def new_frame(self, total_ms: float):
        """Register a completed frame with its end-to-end latency in ms."""
        self.frames += 1
        self._last_frame_ms = total_ms

    def to_dict(self) -> dict:
        """Return {"stages": {name: avg_ms}, "total_ms", "frames", "fps"}."""
        stages = {
            name: (self._totals[name] / self._counts[name])
            for name in self._totals
            if self._counts.get(name, 0) > 0
        }
        fps = 1000.0 / self._last_frame_ms if self._last_frame_ms > 0 else 0.0
        return {
            "stages": stages,
            "total_ms": self._last_frame_ms,
            "frames": self.frames,
            "fps": fps,
        }
