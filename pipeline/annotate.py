"""SiteSentry safety monitoring pipeline: annotated frame rendering.

Note: OpenCV 5.x renders cv2.putText through HarfBuzz automatically, so the
ASCII labels below (e.g. "ID3 HELMET:OK VEST:MISS") use clean glyph shaping
with no extra freetype dependency.
"""

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_OK_COLOR = (0, 200, 0)  # green: PPE compliant
_BAD_COLOR = (0, 0, 255)  # red: violation
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _blur_face_regions(frame: np.ndarray, head: np.ndarray, hx: int, hy: int) -> np.ndarray:
    """Blur faces detected in the head crop; fall back to blurring the whole head region.

    The Haar cascade file ships with opencv-python (cv2.data.haarcascades);
    if it is missing, the entire head region is blurred for privacy.
    """
    cascade_path = os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
    )
    usable = os.path.isfile(cascade_path)
    cascade = cv2.CascadeClassifier(cascade_path) if usable else None
    if not usable or cascade is None or cascade.empty():
        # Cascade unavailable: blur the whole head region for privacy.
        head[:] = cv2.GaussianBlur(head, (31, 31), 0)
    else:
        gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        if len(faces) == 0:
            head[:] = cv2.GaussianBlur(head, (31, 31), 0)
        else:
            for fx, fy, fw, fh in faces:
                roi = head[fy: fy + fh, fx: fx + fw]
                head[fy: fy + fh, fx: fx + fw] = cv2.GaussianBlur(roi, (31, 31), 0)
    frame[hy: hy + head.shape[0], hx: hx + head.shape[1]] = head
    return frame


def annotate_frame(
    frame: np.ndarray,
    tracks: list,
    ppe_results: dict,
    events: list,
    metrics: dict,
    blur_faces: bool = True,
    zones=None,
) -> np.ndarray:
    """Draw tracking boxes, PPE labels, zone overlays, and timing info.

    Args:
        frame: BGR frame to annotate (copied; the input is not modified).
        tracks: Track objects (or dicts) with .id/.bbox/.conf.
        ppe_results: {track_id: ppe dict with helmet/vest flags}.
        events: Pipeline events (unused for drawing except fall alerts).
        metrics: Dict like PipelineMetrics.to_dict(); "total_ms" is shown top-left.
        blur_faces: Blur detected faces in each track's head region.
        zones: Optional ZoneMonitor; its polygons are overlaid.

    Returns:
        The annotated frame with the same shape and dtype (uint8) as the input.
    """
    out = frame.copy()
    if zones is not None:
        zones.draw(out)

    for track in tracks:
        # Accept Track objects or plain dicts {"id", "bbox"} (e.g. from JSON).
        tid = getattr(track, "id", None)
        tbbox = getattr(track, "bbox", None)
        if isinstance(track, dict):
            tid = track.get("id", tid)
            tbbox = track.get("bbox", tbbox)
        ppe = ppe_results.get(tid, {})
        helmet_ok = bool(ppe.get("helmet", False))
        vest_ok = bool(ppe.get("vest", False))
        compliant = helmet_ok and vest_ok
        color = _OK_COLOR if compliant else _BAD_COLOR
        x1, y1, x2, y2 = (int(v) for v in tbbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = (
            f"ID{tid} "
            f"HELMET:{'OK' if helmet_ok else 'MISS'} "
            f"VEST:{'OK' if vest_ok else 'MISS'}"
        )
        # Label with a filled background for readability; above the box,
        # or just inside it when the box starts near the top edge.
        lx, ly = (x1 + 4, y1 - 8) if y1 > 44 else (x1 + 4, y1 + 24)
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.55, 2)
        cv2.rectangle(out, (lx - 4, ly - th - 6), (lx + tw + 4, ly + 6),
                      (10, 10, 10), cv2.FILLED)
        cv2.putText(out, label, (lx, ly), _FONT, 0.55, color, 2)

        if blur_faces:
            hb = ppe.get("head_box")
            if hb is not None:
                # Blur the head-item region via the cascade helper: it blurs
                # detected faces, else the whole head region (privacy-safe).
                hx1, hy1, hx2, hy2 = (int(v) for v in hb)
                hx1 = max(0, hx1); hy1 = max(0, hy1)
                hx2 = min(out.shape[1], hx2); hy2 = min(out.shape[0], hy2)
                if hx2 > hx1 and hy2 > hy1:
                    head = out[hy1:hy2, hx1:hx2].copy()
                    _blur_face_regions(out, head, hx1, hy1)
            else:
                # Fallback: top 22% of the track box (face zone for tall boxes).
                h = y2 - y1
                head_h = max(1, int(h * 0.22))
                if y1 + head_h <= out.shape[0] and x2 > x1:
                    head = out[y1: y1 + head_h, x1:x2].copy()
                    _blur_face_regions(out, head, x1, y1)

    total_ms = float(metrics.get("total_ms", 0.0))
    fps = float(metrics.get("fps", 0.0))
    cv2.putText(
        out,
        f"{total_ms:.1f} ms  {fps:.1f} fps",
        (10, 28),
        _FONT,
        0.7,
        (255, 255, 255),
        2,
    )
    return out
