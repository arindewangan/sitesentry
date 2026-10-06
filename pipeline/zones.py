"""SiteSentry safety monitoring pipeline: restricted-zone intrusion monitoring."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ZoneMonitor:
    """Track restricted polygonal zones and flag track intrusions.

    Also maintains a MOG2 background model (via ``update_background``) that
    can feed motion-based analytics in a deployed build.
    """

    def __init__(self):
        """Initialize with no zones and a fresh MOG2 subtractor."""
        self.zones: list[dict] = []
        self.bg = cv2.createBackgroundSubtractorMOG2()

    def add_zone(self, name: str, polygon_points: list[tuple], restricted: bool = True):
        """Register a named polygon zone.

        Args:
            name: Zone identifier used in events and overlays.
            polygon_points: List of (x, y) vertices.
            restricted: Only restricted zones raise ZONE_INTRUSION events.
        """
        pts = np.array(polygon_points, dtype=np.int32)
        self.zones.append({"name": name, "polygon": pts, "restricted": restricted})

    def update_background(self, frame: np.ndarray):
        """Feed a frame to the MOG2 background model."""
        self.bg.apply(frame)

    def check(self, tracks: list, frame: np.ndarray) -> list[dict]:
        """Return a ZONE_INTRUSION event per track whose center is inside a restricted zone."""
        events = []
        for zone in self.zones:
            if not zone["restricted"]:
                continue
            poly = zone["polygon"].astype(np.float32)
            for track in tracks:
                bbox = track.bbox
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                    events.append(
                        {
                            "type": "ZONE_INTRUSION",
                            "zone": zone["name"],
                            "track_id": track.id,
                            "conf": 1.0,
                            "bbox": tuple(bbox),
                        }
                    )
        return events

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Overlay zone polygons: red = restricted, blue = non-restricted."""
        for zone in self.zones:
            color = (0, 0, 255) if zone["restricted"] else (255, 0, 0)
            cv2.polylines(frame, [zone["polygon"]], True, color, 2)
            x, y = zone["polygon"][0]
            cv2.putText(
                frame,
                str(zone["name"]),
                (int(x) + 4, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return frame
