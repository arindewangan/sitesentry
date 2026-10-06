"""SiteSentry safety monitoring pipeline: Kalman-filter multi-object tracking."""

from collections import deque

import cv2
import numpy as np


def iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    """A single tracked person: Kalman-filter state, bbox history, fall flag."""

    def __init__(self, track_id: int, bbox: tuple, conf: float = 0.0):
        """Create a track with a (4,2) Kalman filter (state: cx, cy, vx, vy)."""
        self.id = track_id
        self.bbox = tuple(bbox)
        self.conf = float(conf)  # confidence of the most recent matched detection
        self.hits = 1
        self.misses = 0
        self.history: deque = deque(maxlen=30)  # recent (bbox, center) snapshots
        self.fall_candidate = False

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        cx, cy = self._center(bbox)
        self.kf.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kf.statePost = self.kf.statePre.copy()
        self._record(bbox)

    @staticmethod
    def _center(bbox: tuple) -> tuple[float, float]:
        """Return the (cx, cy) of an (x1, y1, x2, y2) box."""
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _record(self, bbox: tuple):
        """Append a (bbox, center, vy) snapshot to the history window."""
        cx, cy = self._center(bbox)
        state = self.kf.statePost
        vy = float(state[3, 0]) if state is not None else 0.0
        self.history.append({"bbox": tuple(bbox), "cx": cx, "cy": cy, "vy": vy})

    def predict(self) -> tuple[float, float]:
        """Advance the filter and return the predicted (cx, cy)."""
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def update(self, bbox: tuple | None, conf: float | None = None):
        """Correct the filter with a new detection, or count a miss.

        A miss keeps the predicted box; fall-candidate analysis runs on hits.
        """
        if bbox is None:
            self.misses += 1
            pred_cx, pred_cy = self.predict()
            w = self.bbox[2] - self.bbox[0]
            h = self.bbox[3] - self.bbox[1]
            self.bbox = (pred_cx - w / 2, pred_cy - h / 2, pred_cx + w / 2, pred_cy + h / 2)
            return
        self.misses = 0
        self.hits += 1
        self.bbox = tuple(bbox)
        if conf is not None:
            self.conf = float(conf)
        cx, cy = self._center(bbox)
        self.kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        self._record(bbox)
        self._check_fall()

    def _check_fall(self):
        """Set fall_candidate when the aspect ratio flips tall->flat with a downward spike."""
        if len(self.history) < 2:
            return
        first = self.history[0]
        last = self.history[-1]
        w0, h0 = first["bbox"][2] - first["bbox"][0], first["bbox"][3] - first["bbox"][1]
        w1, h1 = last["bbox"][2] - last["bbox"][0], last["bbox"][3] - last["bbox"][1]
        ar0 = h0 / w0 if w0 > 0 else 0.0
        ar1 = h1 / w1 if w1 > 0 else 0.0
        # OpenCV image y grows downward, so positive vy = moving down.
        downward_spike = last["vy"] > 3.0
        if ar0 > 1.1 and ar1 < 0.7 and downward_spike:
            self.fall_candidate = True


class MultiTracker:
    """Associate detections to Kalman-filtered tracks via IoU + centroid cost."""

    def __init__(self, max_misses: int = 8, iou_threshold: float = 0.3):
        """Set the stale-track prune limit and the minimum IoU to match."""
        self.max_misses = max_misses
        self.iou_threshold = iou_threshold
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[dict]) -> list[Track]:
        """Match detections to tracks, update filters, and prune stale tracks."""
        det_boxes = [tuple(d["bbox"]) for d in detections]
        det_confs = [float(d.get("conf", 0.0)) for d in detections]
        assigned_dets: set[int] = set()

        # Greedy best-pair matching on a combined IoU/centroid cost.
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self.tracks):
            px, py = track.predict()
            for di, box in enumerate(det_boxes):
                score = iou(track.bbox, box)
                if score < self.iou_threshold:
                    continue
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                pairs.append((score - dist / 1000.0, ti, di))
        pairs.sort(reverse=True)
        assignment: dict[int, int] = {}  # track_idx -> det_idx
        for _, ti, di in pairs:
            if ti in assignment or di in assigned_dets:
                continue
            assignment[ti] = di
            assigned_dets.add(di)

        for ti, track in enumerate(self.tracks):
            if ti in assignment:
                di = assignment[ti]
                track.update(det_boxes[di], det_confs[di])
            else:
                track.update(None)

        for di, box in enumerate(det_boxes):
            if di not in assigned_dets:
                self.tracks.append(Track(self._next_id, box, det_confs[di]))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks
