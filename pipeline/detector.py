"""SiteSentry safety monitoring pipeline: DNN engine selection and YOLO detection parsing.

The person detector and the PPE analyzer both consume YOLO-style ONNX outputs,
so the raw-output parser lives here as a shared module-level function.
"""

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0  # COCO class id for "person" in YOLOv8 person-model outputs


def select_dnn_engine() -> tuple[str, bool]:
    """Pick the best available cv2.dnn inference-engine backend and apply it.

    OpenCV 5.0.0 exposes ENGINE_NEW (later 5.x added ENGINE_OPENCV), so probe
    both with getattr. The 5.0.0 setter is deprecated and string-typed, so
    applying the int enum is expected to fail there — that is caught and
    reported via ``applied``; 5.x uses the new graph engine by default.

    Returns:
        (name, applied): backend name ("ENGINE_NEW" / "ENGINE_OPENCV" /
        "ENGINE_CLASSIC" fallback) and whether it was successfully applied.
    """
    engine = getattr(cv2.dnn, "ENGINE_OPENCV", getattr(cv2.dnn, "ENGINE_NEW", None))
    setter = getattr(cv2.dnn, "setInferenceEngineBackendType", None)
    if engine is None or setter is None:
        logger.info("cv2.dnn inference-engine selector unavailable; using defaults")
        return "ENGINE_CLASSIC", False
    # Map the constant back to a readable name for logging/telemetry.
    names = {
        getattr(cv2.dnn, "ENGINE_OPENCV", -1): "ENGINE_OPENCV",
        getattr(cv2.dnn, "ENGINE_NEW", -1): "ENGINE_NEW",
        getattr(cv2.dnn, "ENGINE_CLASSIC", -1): "ENGINE_CLASSIC",
    }
    name = names.get(int(engine), "UNKNOWN")
    try:
        setter(engine)
        logger.info("cv2.dnn inference engine set to %s", name)
        return name, True
    except Exception as exc:  # pragma: no cover - depends on backend build
        # Expected on opencv-python 5.0.0: the setter is deprecated/string-typed
        # and 5.x uses the new graph engine by default. Not an error.
        logger.debug("dnn engine %s not applied (%s); using default", name, exc)
        return name, False


def _make_blob(frame: np.ndarray, input_size: int) -> tuple[np.ndarray, str]:
    """Build a dnn input blob, preferring FP16 with an FP32 fallback.

    Returns:
        (blob, precision): precision is "fp16" or "fp32".
    """
    try:
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(input_size, input_size),
            swapRB=True,
            crop=False,
            ddepth=cv2.CV_16F,
        )
        return blob, "fp16"
    except cv2.error as exc:
        logger.debug("FP16 blob failed, falling back to FP32: %s", exc)
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(input_size, input_size),
            swapRB=True,
            crop=False,
        )
        return blob, "fp32"


def _parse_yolo_detections(
    out: np.ndarray,
    input_size: int,
    conf_threshold: float,
    nms_threshold: float,
    class_filter: set[int] | None = None,
) -> list[dict]:
    """Parse a YOLO-style ONNX output tensor into detections.

    Handles the (1, C, 8400) row format: rows 0-3 are [cx, cy, w, h] in
    input-image pixels and rows 4..C-1 are per-class confidence scores,
    with one column per anchor. The winning class of each anchor is taken;
    anchors whose score is below ``conf_threshold`` (or whose class is not in
    ``class_filter`` when given) are dropped, then cv2.dnn.NMSBoxes
    suppresses overlaps.

    Returns:
        List of {"bbox": (x1, y1, x2, y2), "conf": float, "class_id": int}
        with boxes expressed in the ``input_size`` x ``input_size`` frame.
    """
    preds = np.squeeze(np.asarray(out, dtype=np.float32))
    if preds.ndim != 2 or preds.shape[0] < 5:
        return []
    num_classes = preds.shape[0] - 4
    boxes: list[list[float]] = []
    confidences: list[float] = []
    class_ids: list[int] = []
    for anchor in range(preds.shape[1]):
        scores = preds[4:, anchor]
        class_id = int(np.argmax(scores))
        conf = float(scores[class_id])
        if conf < conf_threshold:
            continue
        if class_filter is not None and class_id not in class_filter:
            continue
        cx, cy, w, h = (float(preds[i, anchor]) for i in range(4))
        x1 = min(max(cx - w / 2.0, 0.0), float(input_size))
        y1 = min(max(cy - h / 2.0, 0.0), float(input_size))
        x2 = min(max(cx + w / 2.0, 0.0), float(input_size))
        y2 = min(max(cy + h / 2.0, 0.0), float(input_size))
        boxes.append([x1, y1, x2, y2])
        confidences.append(conf)
        class_ids.append(class_id)
    if not boxes:
        return []
    kept = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if kept is None or len(kept) == 0:
        return []
    kept_idx = np.asarray(kept).flatten()
    return [
        {
            "bbox": tuple(boxes[i]),
            "conf": confidences[i],
            "class_id": class_ids[i],
        }
        for i in kept_idx
        if i < len(boxes)
    ]


class PersonDetector:
    """Detect people in frames using a YOLOv8 ONNX model via cv2.dnn.

    Gracefully degrades: if the model file is missing or fails to load,
    ``self.enabled`` is False and ``detect()`` returns an empty list.
    """

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        input_size: int = 640,
    ):
        """Load the ONNX model if present; otherwise disable detection."""
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.net = None
        self.enabled = False
        self.precision = "fp32"

        if model_path and os.path.isfile(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                self.enabled = True
                logger.info("Loaded person-detection model: %s", model_path)
            except cv2.error as exc:
                logger.warning("Failed to load ONNX model %s: %s", model_path, exc)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Return [{"bbox": (x1,y1,x2,y2), "conf", "class_id"}] for detected people.

        Boxes are rescaled from model-input space back to the frame size.
        """
        if not self.enabled or self.net is None:
            return []
        h, w = frame.shape[:2]
        blob, self.precision = _make_blob(frame, self.input_size)
        self.net.setInput(blob)
        try:
            output = self.net.forward()
        except cv2.error as exc:
            logger.warning("DNN forward failed: %s", exc)
            return []
        sx, sy = w / self.input_size, h / self.input_size
        dets = _parse_yolo_detections(
            output,
            self.input_size,
            self.conf_threshold,
            self.nms_threshold,
            class_filter={PERSON_CLASS_ID},
        )
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            det["bbox"] = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        return dets
