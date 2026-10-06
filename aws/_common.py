"""Shared helpers for the SiteSentry local/AWS client layer.

Every client in this package follows the same rule:

* Try real AWS via boto3's default credential chain.
* Fall back to local mode (JSONL / file store under
  ``<base_dir>/data/local/<service>/``) when credentials are missing,
  the region is missing, any boto client construction fails, or the
  environment variable ``SITENTRY_AWS`` is set to ``off``.

Local-mode outputs are always clearly labeled ``local`` and must never
be presented as real AWS results. No credentials are ever hardcoded.

Importing this module performs no network I/O.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

#: Project root (the ``sitesentry/`` directory) used as the default base dir.
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent

#: Values of SITENTRY_AWS that force local (offline) mode.
_AWS_OFF_VALUES = {"off", "0", "false", "no", "disabled"}


def aws_force_off() -> bool:
    """Return True when the ``SITENTRY_AWS`` env var disables AWS usage."""
    return os.environ.get("SITENTRY_AWS", "").strip().lower() in _AWS_OFF_VALUES


def has_aws_credentials() -> bool:
    """Best-effort check of the default credential chain (no network I/O).

    Looks at env vars / shared config via a boto3 Session. Any failure
    means "no usable credentials".
    """
    try:
        import boto3

        session = boto3.Session()
        creds = session.get_credentials()
        if creds is None:
            return False
        region = (
            session.region_name
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        return region is not None
    except Exception:
        return False


def resolve_mode() -> str:
    """Return ``"aws"`` or ``"local"`` for a new client."""
    if aws_force_off():
        return "local"
    return "aws" if has_aws_credentials() else "local"


def local_dir(base_dir: Path | str | None, service: str) -> Path:
    """Return (creating) the local store dir for a service."""
    root = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
    d = root / "data" / "local" / service
    d.mkdir(parents=True, exist_ok=True)
    return d


def utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict) -> None:
    """Append a dict as one JSON line to ``path`` (creating parents)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read all JSON lines from ``path``; missing file -> []."""
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
