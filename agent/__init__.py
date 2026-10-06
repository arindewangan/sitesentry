"""SiteSentry: OpenCV 5 vision pipeline for construction-site safety monitoring.

Composes person detection, Kalman tracking, PPE analysis, zone monitoring,
annotation, and timing metrics into a single per-frame ``process()`` call.
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .annotate import annotate_frame
from .detector import PersonDetector, select_dnn_engine
from .metrics import PipelineMetrics
from .ppe import PPEAnalyzer
from .tracker import MultiTracker
from .zones import ZoneMonitor

logger = logging.getLogger(__name__)

__all__ = ["SafetyPipeline", "engine_info", "annotate_frame"]

# Models live next to the code: <project>/models/*.onnx (git-ignored).
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_PERSON_CANDIDATES = ["yolov8n.onnx"]
_PPE_CANDIDATES = ["ppe_yolov8s.onnx", "safetyvision-v1/best.onnx"]


def _autodiscover(candidates: list[str]) -> str | None:
    """Return the first existing model path from the candidates, else None."""
    for name in candidates:
        p = _DEFAULT_MODEL_DIR / name
        if p.is_file():
            return str(p)
    return None


def engine_info() -> dict:
    """Describe the OpenCV/DNN runtime this pipeline runs on.

    Returns:
        {"opencv_version", "dnn_engine" (name from select_dnn_engine),
         "fp16" (whether an FP16 input blob can be built)}.
    """
    engine_name, _ = select_dnn_engine()
    fp16 = False
    try:
        probe = np.zeros((8, 8, 3), dtype=np.uint8)
        cv2.dnn.blobFromImage(probe, size=(8, 8), ddepth=cv2.CV_16F)
        fp16 = True
    except cv2.error:
        fp16 = False
    return {
        "opencv_version": cv2.__version__,
        "dnn_engine": engine_name,
        "fp16": fp16,
    }


class SafetyPipeline:
    """End-to-end safety pipeline: detect -> track -> PPE -> zones -> events."""

    def __init__(
        self,
        model_dir: str = "models",
        person_model: str | None = None,
        ppe_model: str | None = None,
    ):
        """Wire up the pipeline stages.

        Args:
            model_dir: Directory models are expected in (informational default).
            person_model: Path to a YOLOv8 person ONNX file. None (default)
                auto-discovers ``models/yolov8n.onnx``; when absent, person
                detections come from the PPE model's own Person class.
            ppe_model: Path to the safetyvision PPE ONNX file. None (default)
                auto-discovers ``models/ppe_yolov8s.onnx``; when absent the
                HSV heuristic fallback is used.
        """
        if person_model is None:
            person_model = _autodiscover(_PERSON_CANDIDATES)
        if ppe_model is None:
            ppe_model = _autodiscover(_PPE_CANDIDATES)
        self.detector = PersonDetector(model_path=person_model)
        self.tracker = MultiTracker()
        self.ppe = PPEAnalyzer(model_path=ppe_model)
        self.zones = ZoneMonitor()
        self.metrics = PipelineMetrics()
        self.blur_faces = True  # responsible-use default: blur faces in output
        self._engine_name, self._engine_applied = select_dnn_engine()
        self._engine_info = engine_info()
        logger.info(
            "SafetyPipeline ready: person_detector=%s ppe_model=%s engine=%s(applied=%s)",
            self.detector.enabled,
            self.ppe.enabled,
            self._engine_name,
            self._engine_applied,
        )

    def add_zone(self, name: str, polygon_points: list, restricted: bool = True):
        """Register a restricted (or non-restricted) zone on the monitor."""
        self.zones.add_zone(name, polygon_points, restricted=restricted)

    def process(self, frame: np.ndarray) -> dict:
        """Run one frame through the pipeline.

        Returns:
            {
              "tracks": [{"id", "bbox": (x1,y1,x2,y2), "conf", "ppe": {...}}],
              "events": [{"type", "track_id", "conf", "bbox", "ts"}],
              "metrics": PipelineMetrics.to_dict(),
              "engine": engine_info(),
            }
        """
        start = time.perf_counter()
        with self.metrics.stage("detect"):
            detections = self.detector.detect(frame)
            if not detections and self.ppe.enabled:
                # No standalone person model: the PPE model's Person class
                # is the person detector (one forward pass, shared cache).
                detections = self.ppe.detect_persons(frame)
        with self.metrics.stage("track"):
            tracks = self.tracker.update(detections)
        with self.metrics.stage("ppe"):
            ppe_results = self.ppe.analyze(frame, tracks)
        with self.metrics.stage("zones"):
            self.zones.update_background(frame)
            zone_events = self.zones.check(tracks, frame)

        ts = time.time()
        events = []
        for ev in zone_events:
            events.append({**ev, "ts": ts})

        track_out = []
        for track in tracks:
            ppe = ppe_results.get(track.id, {})
            track_out.append(
                {
                    "id": track.id,
                    "bbox": tuple(float(v) for v in track.bbox),
                    "conf": float(track.conf),
                    "ppe": ppe,
                }
            )
            if not ppe.get("helmet", False):
                no_helmet_conf = ppe.get("no_helmet_conf", 0.0) or 0.0
                conf = float(no_helmet_conf if no_helmet_conf > 0 else 1.0 - ppe.get("helmet_conf", 0.0))
                events.append(
                    {
                        "type": "NO_HELMET",
                        "track_id": track.id,
                        "conf": conf,
                        "bbox": tuple(float(v) for v in track.bbox),
                        "ts": ts,
                        "evidence": (
                            f"track {track.id}: no hardhat (conf {conf:.2f}, "
                            f"method={ppe.get('method', '?')})"
                        ),
                    }
                )
            if not ppe.get("vest", False):
                no_vest_conf = ppe.get("no_vest_conf", 0.0) or 0.0
                conf = float(no_vest_conf if no_vest_conf > 0 else 1.0 - ppe.get("vest_conf", 0.0))
                events.append(
                    {
                        "type": "NO_VEST",
                        "track_id": track.id,
                        "conf": conf,
                        "bbox": tuple(float(v) for v in track.bbox),
                        "ts": ts,
                        "evidence": (
                            f"track {track.id}: no safety vest (conf {conf:.2f}, "
                            f"method={ppe.get('method', '?')})"
                        ),
                    }
                )
            if ppe.get("fall_detected", False) or track.fall_candidate:
                events.append(
                    {
                        "type": "FALL_SUSPECTED",
                        "track_id": track.id,
                        "conf": float(ppe.get("fall_conf", 0.9)),
                        "bbox": tuple(float(v) for v in track.bbox),
                        "ts": ts,
                        "evidence": (
                            f"track {track.id}: fall signature "
                            f"(model={ppe.get('fall_detected', False)}, "
                            f"motion={track.fall_candidate})"
                        ),
                    }
                )

        total_ms = (time.perf_counter() - start) * 1000.0
        self.metrics.new_frame(total_ms)
        return {
            "tracks": track_out,
            "events": events,
            "metrics": self.metrics.to_dict(),
            "engine": self._engine_info,
        }
