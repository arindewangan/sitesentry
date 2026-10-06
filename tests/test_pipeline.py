"""Offline tests for the SiteSentry OpenCV 5 vision pipeline.

All tests run with synthetic data only: no model files, no network.
"""

import numpy as np
import pytest

from pipeline import SafetyPipeline, engine_info
from pipeline.annotate import annotate_frame
from pipeline.detector import PersonDetector, select_dnn_engine
from pipeline.metrics import PipelineMetrics, StageTimer
from pipeline.ppe import PPE_CLASSES, PPEAnalyzer
from pipeline.tracker import MultiTracker, Track
from pipeline.zones import ZoneMonitor


class FakeNet:
    """Stand-in for cv2.dnn.Net with a scripted forward() output."""

    def __init__(self, output):
        self.output = output
        self.input_blob = None

    def setInput(self, blob):
        self.input_blob = blob

    def forward(self):
        return self.output


def _yolo_person_output():
    """(1, 84, 8400) array with one strong person detection at anchor 100."""
    out = np.zeros((1, 84, 8400), dtype=np.float32)
    out[0, 0, 100] = 320.0  # cx
    out[0, 1, 100] = 320.0  # cy
    out[0, 2, 100] = 128.0  # w
    out[0, 3, 100] = 256.0  # h
    out[0, 4, 100] = 0.95  # person score
    return out


def _yolo_ppe_output():
    """(1, 17, 8400) array with one Hardhat detection at anchor 7."""
    out = np.zeros((1, 17, 8400), dtype=np.float32)
    out[0, 0, 7] = 320.0  # cx
    out[0, 1, 7] = 300.0  # cy
    out[0, 2, 7] = 100.0  # w
    out[0, 3, 7] = 100.0  # h
    out[0, 4 + 3, 7] = 0.92  # Hardhat (class 3) score
    return out


# ------------------------------------------------------------------
# (a) dnn engine selection
# ------------------------------------------------------------------
def test_select_dnn_engine_returns_name_and_bool():
    name, applied = select_dnn_engine()
    assert name in {"ENGINE_NEW", "ENGINE_OPENCV", "ENGINE_CLASSIC", "UNKNOWN"}
    assert isinstance(applied, bool)


def test_engine_info_shape():
    info = engine_info()
    assert info["opencv_version"] == "5.0.0"
    assert isinstance(info["dnn_engine"], str)
    assert isinstance(info["fp16"], bool)


# ------------------------------------------------------------------
# (b) missing model -> graceful degradation
# ------------------------------------------------------------------
def test_person_detector_missing_model_disabled():
    det = PersonDetector(model_path="/nonexistent/x.onnx")
    assert det.enabled is False
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert det.detect(frame) == []


def test_ppe_analyzer_missing_model_uses_heuristic():
    analyzer = PPEAnalyzer(model_path="/nonexistent/ppe.onnx")
    assert analyzer.enabled is False


# ------------------------------------------------------------------
# (c) YOLOv8 person parse via monkeypatched net
# ------------------------------------------------------------------
def test_yolo_parse_single_person_detection():
    det = PersonDetector(model_path="/nonexistent/x.onnx")
    det.net = FakeNet(_yolo_person_output())
    det.enabled = True
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    dets = det.detect(frame)
    assert len(dets) == 1
    x1, y1, x2, y2 = dets[0]["bbox"]
    assert (x1, y1, x2, y2) == pytest.approx((256.0, 192.0, 384.0, 448.0))
    assert dets[0]["conf"] == pytest.approx(0.95)
    assert dets[0]["class_id"] == 0


def test_yolo_parse_scales_to_frame_size():
    det = PersonDetector(model_path="/nonexistent/x.onnx")
    det.net = FakeNet(_yolo_person_output())
    det.enabled = True
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    dets = det.detect(frame)
    assert len(dets) == 1
    assert dets[0]["bbox"] == pytest.approx((128.0, 96.0, 192.0, 224.0))


# ------------------------------------------------------------------
# (d) tracker stability
# ------------------------------------------------------------------
def test_multitracker_stable_ids():
    mt = MultiTracker()
    frame1 = [{"bbox": (100, 100, 200, 300), "conf": 0.9, "class_id": 0}]
    tracks1 = mt.update(frame1)
    assert len(tracks1) == 1
    first_id = tracks1[0].id
    frame2 = [{"bbox": (105, 102, 205, 302), "conf": 0.9, "class_id": 0}]
    tracks2 = mt.update(frame2)
    assert len(tracks2) == 1
    assert tracks2[0].id == first_id


def test_multitracker_two_tracks_kept_apart():
    mt = MultiTracker()
    dets = [
        {"bbox": (10, 10, 60, 160), "conf": 0.9, "class_id": 0},
        {"bbox": (400, 10, 450, 160), "conf": 0.9, "class_id": 0},
    ]
    tracks = mt.update(dets)
    assert len(tracks) == 2
    assert tracks[0].id != tracks[1].id


# ------------------------------------------------------------------
# (e) PPE class map + heuristic classify
# ------------------------------------------------------------------
def test_ppe_classes_has_13_entries():
    assert len(PPE_CLASSES) == 13
    assert PPE_CLASSES[3] == "Hardhat"
    assert PPE_CLASSES[7] == "NO-Hardhat"
    assert PPE_CLASSES[12] == "Safety Vest"
    assert PPE_CLASSES[9] == "NO-Safety Vest"
    assert PPE_CLASSES[0] == "Fall-Detected"


def _yellow_head_crop():
    crop = np.full((200, 100, 3), 128, dtype=np.uint8)  # gray body
    # Yellow rectangle in the head region (top 30%): BGR yellow.
    cv2_yellow = (0, 255, 255)
    crop[10:50, 30:70] = cv2_yellow
    return crop


def test_heuristic_helmet_detected():
    analyzer = PPEAnalyzer(model_path="/nonexistent/ppe.onnx")
    result = analyzer.classify(_yellow_head_crop())
    assert result["method"] == "heuristic"
    assert result["helmet"] is True
    assert result["helmet_conf"] > 0


def test_heuristic_plain_gray_no_helmet():
    analyzer = PPEAnalyzer(model_path="/nonexistent/ppe.onnx")
    crop = np.full((200, 100, 3), 128, dtype=np.uint8)
    result = analyzer.classify(crop)
    assert result["helmet"] is False
    assert result["vest"] is False


def test_heuristic_empty_crop_safe():
    analyzer = PPEAnalyzer(model_path="/nonexistent/ppe.onnx")
    result = analyzer.classify(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result["helmet"] is False


# ------------------------------------------------------------------
# PPE YOLO analyze() association
# ------------------------------------------------------------------
def test_analyze_hardhat_associates_to_track():
    analyzer = PPEAnalyzer(model_path="/nonexistent/ppe.onnx")
    analyzer.net = FakeNet(_yolo_ppe_output())
    analyzer.enabled = True
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    track = Track(1, (250, 250, 390, 390))
    results = analyzer.analyze(frame, [track])
    ppe = results[1]
    assert ppe["helmet"] is True
    assert ppe["helmet_conf"] == pytest.approx(0.92)
    assert ppe["method"] == "ppe-yolo"


def test_analyze_conservative_default_no_detections():
    analyzer = PPEAnalyzer(model_path="/nonexistent/ppe.onnx")
    analyzer.net = FakeNet(np.zeros((1, 17, 8400), dtype=np.float32))
    analyzer.enabled = True
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    track = Track(1, (250, 250, 390, 390))
    results = analyzer.analyze(frame, [track])
    ppe = results[1]
    assert ppe["helmet"] is False
    assert ppe["helmet_conf"] == pytest.approx(0.25)
    assert ppe["vest"] is False


def test_analyze_no_hardhat_forces_violation():
    analyzer = PPEAnalyzer(model_path="/nonexistent/ppe.onnx")
    out = np.zeros((1, 17, 8400), dtype=np.float32)
    out[0, 0, 7] = 320.0
    out[0, 1, 7] = 300.0
    out[0, 2, 7] = 100.0
    out[0, 3, 7] = 100.0
    out[0, 4 + 7, 7] = 0.88  # NO-Hardhat (class 7)
    analyzer.net = FakeNet(out)
    analyzer.enabled = True
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    track = Track(1, (250, 250, 390, 390))
    results = analyzer.analyze(frame, [track])
    assert results[1]["helmet"] is False
    assert results[1]["no_helmet_conf"] == pytest.approx(0.88)


# ------------------------------------------------------------------
# (f) zone intrusion
# ------------------------------------------------------------------
def test_zone_intrusion_event():
    zm = ZoneMonitor()
    zm.add_zone("crane", [(0, 0), (200, 0), (100, 200)], restricted=True)
    track = Track(1, (80, 80, 120, 180))  # center (100, 130) inside triangle
    events = zm.check([track], np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(events) == 1
    assert events[0]["type"] == "ZONE_INTRUSION"
    assert events[0]["zone"] == "crane"
    assert events[0]["track_id"] == 1


def test_zone_no_event_when_outside():
    zm = ZoneMonitor()
    zm.add_zone("crane", [(0, 0), (200, 0), (100, 200)], restricted=True)
    track = Track(1, (400, 400, 450, 550))
    events = zm.check([track], np.zeros((480, 640, 3), dtype=np.uint8))
    assert events == []


# ------------------------------------------------------------------
# (g) annotation
# ------------------------------------------------------------------
def test_annotate_frame_shape_and_dtype():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    track = Track(1, (100, 100, 200, 300))
    ppe_results = {1: {"helmet": True, "vest": False}}
    out = annotate_frame(frame, [track], ppe_results, [], {"total_ms": 12.5, "fps": 80.0})
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    # input must not be modified in place
    assert frame.sum() == 0


def test_annotate_frame_no_tracks():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = annotate_frame(frame, [], {}, [], {"total_ms": 1.0, "fps": 999.0}, blur_faces=False)
    assert out.shape == frame.shape


# ------------------------------------------------------------------
# (h) metrics
# ------------------------------------------------------------------
def test_pipeline_metrics_to_dict_keys():
    m = PipelineMetrics()
    with m.stage("detect"):
        pass
    m.new_frame(16.7)
    d = m.to_dict()
    assert set(d.keys()) == {"stages", "total_ms", "frames", "fps"}
    assert d["frames"] == 1
    assert d["total_ms"] == pytest.approx(16.7)
    assert "detect" in d["stages"]
    assert d["fps"] == pytest.approx(1000.0 / 16.7)


def test_stage_timer_records():
    m = PipelineMetrics()
    with StageTimer(m, "ppe"):
        pass
    assert m.to_dict()["stages"]["ppe"] >= 0.0


# ------------------------------------------------------------------
# (i) end-to-end pipeline with no models
# ------------------------------------------------------------------
def test_safety_pipeline_process_no_models():
    pipe = SafetyPipeline(model_dir="models")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipe.process(frame)
    assert set(result.keys()) == {"tracks", "events", "metrics", "engine"}
    assert result["tracks"] == []
    assert result["events"] == []
    assert set(result["metrics"].keys()) == {"stages", "total_ms", "frames", "fps"}
    assert result["engine"]["opencv_version"] == "5.0.0"


def test_safety_pipeline_process_with_zone():
    pipe = SafetyPipeline(model_dir="models")
    pipe.add_zone("pit", [(0, 0), (640, 0), (320, 480)], restricted=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipe.process(frame)
    assert set(result.keys()) == {"tracks", "events", "metrics", "engine"}


def test_ppe_detect_persons_graceful_without_model():
    """detect_persons returns [] (not an exception) when no model is loaded."""
    from pipeline.ppe import PPEAnalyzer

    analyzer = PPEAnalyzer(model_path="/nonexistent/model.onnx")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert analyzer.detect_persons(frame) == []


def test_pipeline_autodiscovers_models_dir():
    """SafetyPipeline() with no args discovers models/ and stays functional."""
    from pipeline import SafetyPipeline

    pipe = SafetyPipeline()
    # ppe model file exists in this environment; person model does not.
    assert isinstance(pipe.detector.enabled, bool)
    assert isinstance(pipe.ppe.enabled, bool)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    res = pipe.process(frame)
    assert set(res) == {"tracks", "events", "metrics", "engine"}
