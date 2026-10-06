"""SiteSentry safety monitoring pipeline: PPE (helmet/vest) analysis via YOLO detection.

Primary path: the Hugging Face model ayushgupta7777/safetyvision-yolov8
(file v2/best_640.onnx, YOLOv8s, 13 classes, ~43MB) is a YOLO *detector*,
not a crop classifier. Its item detections (hardhat / NO-hardhat / safety
vest / NO-safety-vest / fall) are associated to person tracks by overlap.
Note on weights: Ultralytics YOLOv8-lineage weights are AGPL-3.0 licensed.

Fallback path: when no model file is available, ``classify(crop)`` applies a
pure-OpenCV HSV heuristic to a person crop.
"""

import logging
import os

import cv2
import numpy as np

from .detector import _make_blob, _parse_yolo_detections
from .tracker import Track, iou

logger = logging.getLogger(__name__)

# Class index map for ayushgupta7777/safetyvision-yolov8 v2/best_640.onnx.
# IMPORTANT: verify against the Hugging Face model card before swapping in a
# different checkpoint -- the map below is the one the analyzer assumes.
PPE_CLASSES = {
    0: "Fall-Detected",
    1: "Gloves",
    2: "Goggles",
    3: "Hardhat",
    4: "Mask",
    5: "NO-Gloves",
    6: "NO-Goggles",
    7: "NO-Hardhat",
    8: "NO-Mask",
    9: "NO-Safety Vest",
    10: "No_Harness",
    11: "Person",
    12: "Safety Vest",
}

CLS_FALL = 0
CLS_HARDHAT = 3
CLS_NO_HARDHAT = 7
CLS_NO_VEST = 9
CLS_PERSON = 11
CLS_VEST = 12

# Module-level HSV ranges (OpenCV H in [0,179]) for the heuristic fallback.
# Helmet colors: yellow / white / orange / red hard hats.
HELMET_HSV_RANGES = [
    (np.array([20, 100, 100]), np.array([35, 255, 255])),  # yellow
    (np.array([0, 0, 180]), np.array([179, 40, 255])),  # white (low saturation, bright)
    (np.array([8, 100, 100]), np.array([20, 255, 255])),  # orange
    (np.array([0, 100, 100]), np.array([8, 255, 255])),  # red (low-hue half)
    (np.array([160, 100, 100]), np.array([179, 255, 255])),  # red (high-hue half)
]
# Safety-vest colors: neon yellow / orange hi-vis fabric.
VEST_HSV_RANGES = [
    (np.array([20, 120, 120]), np.array([35, 255, 255])),  # neon yellow
    (np.array([8, 120, 120]), np.array([20, 255, 255])),  # orange
]

HELMET_FRACTION_THRESHOLD = 0.08
VEST_FRACTION_THRESHOLD = 0.12

IOU_ASSOC_THRESHOLD = 0.05


def _mask_fraction(hsv: np.ndarray, ranges: list) -> float:
    """Return the fraction of pixels in `hsv` covered by any of the HSV ranges."""
    combined = None
    for lo, hi in ranges:
        mask = cv2.inRange(hsv, lo, hi)
        combined = mask if combined is None else cv2.bitwise_or(combined, mask)
    if combined is None:
        return 0.0
    return float(cv2.countNonZero(combined)) / (hsv.shape[0] * hsv.shape[1])


class PPEAnalyzer:
    """Analyze helmet/vest compliance per person track.

    With a model file, runs the safetyvision YOLOv8 detector on the frame and
    associates item detections to tracks. Without one, falls back to the
    ``classify()`` HSV heuristic applied to each track's crop.
    """

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        helmet_threshold: float = 0.6,
        vest_threshold: float = 0.6,
    ):
        """Load the PPE YOLO model if present; otherwise use the heuristic."""
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.helmet_threshold = helmet_threshold
        self.vest_threshold = vest_threshold
        self.net = None
        self.enabled = False
        self.precision = "fp32"
        self._last_frame = None  # strong ref: identity cache, no id() reuse
        self._cache_items: list[dict] = []

        if model_path and os.path.isfile(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                self.enabled = True
                logger.info("Loaded PPE model: %s", model_path)
            except cv2.error as exc:
                logger.warning("Failed to load PPE ONNX model %s: %s", model_path, exc)

    # ------------------------------------------------------------------
    # YOLO path
    # ------------------------------------------------------------------
    def _detect_items(self, frame: np.ndarray) -> list[dict]:
        """Run the PPE detector and return item detections in frame coords."""
        h, w = frame.shape[:2]
        blob, self.precision = _make_blob(frame, self.input_size)
        self.net.setInput(blob)
        output = self.net.forward()
        sx, sy = w / self.input_size, h / self.input_size
        dets = _parse_yolo_detections(
            output, self.input_size, self.conf_threshold, self.nms_threshold
        )
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            det["bbox"] = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
            det["label"] = PPE_CLASSES.get(det["class_id"], f"class_{det['class_id']}")
        return dets

    def _detect_items_cached(self, frame: np.ndarray) -> list[dict]:
        """Run the forward pass at most once per frame object.

        detect_persons() and analyze() share one forward pass per process()
        call. Identity (``is``) plus a strong reference avoids the classic
        id()-reuse-after-GC bug.
        """
        if self._last_frame is frame:
            return self._cache_items
        try:
            items = self._detect_items(frame)
        except cv2.error as exc:
            logger.warning("PPE forward failed: %s", exc)
            items = []
        self._last_frame = frame
        self._cache_items = items
        return items

    def detect_persons(self, frame: np.ndarray) -> list[dict]:
        """Return person detections from the PPE model's own Person class.

        Used as the person-detection source when no standalone YOLOv8n model
        is configured: one forward pass yields persons, PPE items and falls.
        Boxes are in frame coordinates: [{"bbox", "conf", "class_id": 11}].
        """
        if not self.enabled or self.net is None:
            return []
        items = self._detect_items_cached(frame)
        h, w = frame.shape[:2]
        persons = [
            {"bbox": it["bbox"], "conf": it["conf"], "class_id": CLS_PERSON,
             "synthesized": False}
            for it in items
            if it["class_id"] == CLS_PERSON and it["conf"] >= 0.30
        ]
        synth = self._synthesize_person_boxes(items, w, h)
        # Two-stage merge: head-derived boxes win; vest-derived boxes that
        # overlap a kept head box (IoU>0.25) are duplicates of the same worker.
        head = self._nms(
            [s for s in synth if s["kind"] == "head"], 0.30, 0.5)
        vest = [s for s in synth if s["kind"] == "vest"
                and max((iou(s["bbox"], hb["bbox"]) for hb in head),
                        default=0.0) < 0.25]
        persons.extend(head)
        persons.extend(self._nms(vest, 0.30, 0.5))
        if not persons:
            return []
        return self._nms(persons, 0.30, 0.5)

    @staticmethod
    def _nms(cands, score_thr, nms_thr):
        """NMS over candidate dicts with 'bbox'/'conf' keys; returns survivors."""
        if not cands:
            return []
        boxes = [list(c["bbox"]) for c in cands]
        confs = [c["conf"] for c in cands]
        kept = cv2.dnn.NMSBoxes(boxes, confs, score_thr, nms_thr)
        if kept is None or len(kept) == 0:
            return []
        return [cands[int(i)] for i in np.asarray(kept).flatten()]

    @staticmethod
    def _synthesize_person_boxes(items, w, h):
        """Estimate person boxes from PPE item detections (head/vest -> body).

        Body proportions from a head box (Hardhat/NO-Hardhat): width 2.4x the
        head width centered on the head, top slightly above the head, height
        4.8x the head height. From a vest box (Vest/NO-Vest): expand upward
        for the head and downward for the legs. Boxes are clamped to the
        frame. Heuristic -- flagged honestly via ``synthesized=True``.
        """
        synth = []
        for it in items:
            x1, y1, x2, y2 = it["bbox"]
            conf = it["conf"] * 0.9  # synthesized boxes inherit discounted conf
            if it["class_id"] in (CLS_HARDHAT, CLS_NO_HARDHAT):
                hw, hh = x2 - x1, y2 - y1
                cx = (x1 + x2) / 2.0
                # Width from head HEIGHT (stable); loose wide head boxes
                # must not explode the person box.
                eff_hw = min(hw, 1.4 * hh)
                pw = 2.2 * eff_hw
                top = y1 - 0.15 * hh
                ph = 4.6 * hh
                box = (cx - pw / 2.0, top, cx + pw / 2.0, top + ph)
            elif it["class_id"] in (CLS_VEST, CLS_NO_VEST):
                vw, vh = x2 - x1, y2 - y1
                box = (x1 - 0.35 * vw, y1 - 0.90 * vh,
                       x2 + 0.35 * vw, y2 + 0.55 * vh)
            else:
                continue
            x1c = min(max(box[0], 0.0), float(w))
            y1c = min(max(box[1], 0.0), float(h))
            x2c = min(max(box[2], 0.0), float(w))
            y2c = min(max(box[3], 0.0), float(h))
            if x2c - x1c < 8 or y2c - y1c < 8:
                continue
            synth.append({"bbox": (x1c, y1c, x2c, y2c), "conf": float(conf),
                          "class_id": CLS_PERSON, "synthesized": True,
                          "kind": ("head" if it["class_id"] in
                                   (CLS_HARDHAT, CLS_NO_HARDHAT) else "vest"),
                          "from": PPE_CLASSES.get(it["class_id"])})
        return synth

    @staticmethod
    def _overlaps(item_bbox: tuple, track_bbox: tuple) -> bool:
        """True when the item overlaps the track: IoU > 0.05 or item center inside the track box."""
        if iou(item_bbox, track_bbox) > IOU_ASSOC_THRESHOLD:
            return True
        cx = (item_bbox[0] + item_bbox[2]) / 2.0
        cy = (item_bbox[1] + item_bbox[3]) / 2.0
        return track_bbox[0] <= cx <= track_bbox[2] and track_bbox[1] <= cy <= track_bbox[3]

    def _decide(self, track_bbox: tuple, items: list[dict]) -> dict:
        """Combine overlapping item detections into one PPE verdict.

        Conservative default: unless there is positive evidence (Hardhat /
        Safety Vest) the track is flagged non-compliant with a low confidence
        (0.25) -- missing evidence counts as a violation rather than a pass.
        Negative evidence ("NO-Hardhat" / "NO-Safety Vest") forces the
        corresponding flag off and overrides any positive hit.
        """
        overlapping = [it for it in items if self._overlaps(it["bbox"], track_bbox)]

        def _best(cid):
            cands = [it for it in overlapping if it["class_id"] == cid]
            return max(cands, key=lambda it: it["conf"]) if cands else None

        hardhat_hit = _best(CLS_HARDHAT)
        no_hardhat_hit = _best(CLS_NO_HARDHAT)
        hardhat_conf = hardhat_hit["conf"] if hardhat_hit else 0.0
        no_hardhat_conf = no_hardhat_hit["conf"] if no_hardhat_hit else 0.0
        head_box = (no_hardhat_hit or hardhat_hit or {}).get("bbox")
        vest_conf = max((it["conf"] for it in overlapping if it["class_id"] == CLS_VEST), default=0.0)
        no_vest_conf = max((it["conf"] for it in overlapping if it["class_id"] == CLS_NO_VEST), default=0.0)
        fall_conf = max((it["conf"] for it in overlapping if it["class_id"] == CLS_FALL), default=0.0)

        if no_hardhat_conf >= self.conf_threshold:
            helmet, helmet_conf = False, no_hardhat_conf
        elif hardhat_conf >= self.conf_threshold:
            helmet, helmet_conf = True, hardhat_conf
        else:
            helmet, helmet_conf = False, 0.25  # conservative default: no evidence -> violation

        if no_vest_conf >= self.conf_threshold:
            vest, vest_conf_out = False, no_vest_conf
        elif vest_conf >= self.conf_threshold:
            vest, vest_conf_out = True, vest_conf
        else:
            vest, vest_conf_out = False, 0.25

        return {
            "helmet": helmet,
            "helmet_conf": float(helmet_conf),
            "vest": vest,
            "vest_conf": float(vest_conf_out),
            "no_helmet_conf": float(no_hardhat_conf),
            "no_vest_conf": float(no_vest_conf),
            "fall_detected": fall_conf >= self.conf_threshold,
            "fall_conf": float(fall_conf),
            "head_box": (tuple(float(v) for v in head_box)
                         if head_box is not None else None),
            "method": "ppe-yolo",
        }

    def analyze(self, frame: np.ndarray, tracks: list[Track]) -> dict[int, dict]:
        """Return {track_id: ppe dict} for every track.

        Uses the YOLO detector when the model is loaded; otherwise runs the
        ``classify()`` heuristic on each track's crop (method="heuristic").
        """
        if not tracks:
            return {}
        if self.enabled and self.net is not None:
            items = self._detect_items_cached(frame)
            h, w = frame.shape[:2]
            results = {}
            for track in tracks:
                results[track.id] = self._decide(track.bbox, items)
            return results
        # Heuristic fallback: one crop per track.
        h, w = frame.shape[:2]
        results = {}
        for track in tracks:
            x1, y1, x2, y2 = (int(v) for v in track.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            ppe = self.classify(frame[y1:y2, x1:x2])
            ppe["method"] = "heuristic"
            ppe["fall_detected"] = False
            ppe["head_box"] = None
            results[track.id] = ppe
        return results

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------
    def classify(self, crop: np.ndarray) -> dict:
        """Classify helmet/vest on a person crop using HSV color heuristics.

        Head = top 30% of the crop, torso = 25-75% of the crop height.
        helmet is True when the fraction of yellow/white/orange/red pixels in
        the head region exceeds 0.08; vest is True when the fraction of
        neon-yellow/orange pixels in the torso exceeds 0.12.
        """
        result = {
            "helmet": False,
            "helmet_conf": 0.0,
            "vest": False,
            "vest_conf": 0.0,
            "method": "heuristic",
        }
        if crop is None or crop.size == 0 or crop.ndim != 3:
            return result
        h = crop.shape[0]
        if h < 10:
            return result
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        head = hsv[: int(h * 0.30)]
        torso = hsv[int(h * 0.25): int(h * 0.75)]
        helmet_frac = _mask_fraction(head, HELMET_HSV_RANGES)
        vest_frac = _mask_fraction(torso, VEST_HSV_RANGES)
        result["helmet"] = helmet_frac > HELMET_FRACTION_THRESHOLD
        result["helmet_conf"] = min(1.0, helmet_frac / HELMET_FRACTION_THRESHOLD * 0.5)
        result["vest"] = vest_frac > VEST_FRACTION_THRESHOLD
        result["vest_conf"] = min(1.0, vest_frac / VEST_FRACTION_THRESHOLD * 0.5)
        return result
