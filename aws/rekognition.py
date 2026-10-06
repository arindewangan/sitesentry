"""Rekognition client with a local (offline) fallback.

AWS mode calls ``rekognition.detect_protective_equipment`` for the
independent second opinion on PPE. Local mode returns a clearly labeled
``local-simulated`` stub — it performs NO real detection, so downstream
code must treat its agreement verdict as unavailable (None).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ._common import DEFAULT_BASE_DIR, resolve_mode


class RekognitionClient:
    """PPE second-opinion client (real Rekognition or local stub)."""

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        """Create the client; ``base_dir`` is accepted for API parity (no
        local files are written by this client)."""
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
        self.mode = resolve_mode()
        self._client = None
        if self.mode == "aws":
            try:
                import boto3

                self._client = boto3.client("rekognition")
            except Exception:
                self.mode = "local"

    def detect_ppe(self, image_bytes: bytes) -> dict:
        """Detect protective equipment on the persons in ``image_bytes``.

        Returns ``{"persons": [...], "source": "rekognition"|"local-simulated"}``.
        In local mode ``persons`` is empty and a ``note`` explains that
        Rekognition is unavailable offline — callers must NOT treat this
        as a real detection result.
        """
        if self.mode == "aws" and self._client is not None:
            resp = self._client.detect_protective_equipment(
                Image={"Bytes": image_bytes},
                SummarizationAttributes={
                    "MinConfidence": 80,
                    "RequiredEquipmentTypes": ["HEAD_COVER", "FACE_COVER", "HAND_COVER"],
                },
            )
            return {"persons": resp.get("Persons", []), "source": "rekognition"}
        return {
            "persons": [],
            "source": "local-simulated",
            "note": (
                "Rekognition unavailable offline; run "
                "detect_protective_equipment on AWS for the independent "
                "second opinion."
            ),
        }
