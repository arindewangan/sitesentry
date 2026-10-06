"""S3 client with a local (offline) fallback.

AWS mode uses ``put_object`` against a real bucket. Local mode writes the
bytes under ``data/local/s3/<key>`` and returns a ``local://`` URI that is
clearly labeled as local.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ._common import DEFAULT_BASE_DIR, local_dir, resolve_mode


class S3Client:
    """Object storage client (real S3 or local file fallback)."""

    def __init__(
        self,
        bucket: str = "sitesentry-demo-media",
        prefix: str = "",
        base_dir: Optional[Path | str] = None,
    ) -> None:
        """Create the client; ``base_dir`` overrides the project root used
        for local-mode storage (tests pass a tmp dir)."""
        self.bucket = bucket
        self.prefix = prefix
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
        self.mode = resolve_mode()
        self._client = None
        if self.mode == "aws":
            try:
                import boto3

                self._client = boto3.client("s3")
            except Exception:
                self.mode = "local"

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}{key}" if self.prefix else key

    def upload_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload raw bytes; returns the object URI.

        AWS: ``s3://<bucket>/<key>``. Local: ``local://data/local/s3/<key>``.
        """
        full_key = self._full_key(key)
        if self.mode == "aws" and self._client is not None:
            self._client.put_object(
                Bucket=self.bucket,
                Key=full_key,
                Body=data,
                ContentType=content_type,
            )
            return f"s3://{self.bucket}/{full_key}"
        dest = local_dir(self.base_dir, "s3") / full_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return f"local://data/local/s3/{full_key}"
