"""DynamoDB client with a local (offline) fallback.

AWS mode uses a DynamoDB table via ``boto3.resource("dynamodb")``.
Local mode appends each item as one JSON line to
``data/local/dynamodb/events.jsonl`` and filters in memory, which is
fine for demo/hackathon scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ._common import DEFAULT_BASE_DIR, append_jsonl, read_jsonl, resolve_mode, utcnow_iso


class DynamoClient:
    """Event store client (real DynamoDB or local JSONL fallback)."""

    def __init__(
        self,
        table: str = "sitesentry-events",
        base_dir: Optional[Path | str] = None,
    ) -> None:
        """Create the client; ``base_dir`` overrides the project root used
        for local-mode storage (tests pass a tmp dir)."""
        self.table_name = table
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
        self.mode = resolve_mode()
        self._table = None
        if self.mode == "aws":
            try:
                import boto3

                self._table = boto3.resource("dynamodb").Table(table)
            except Exception:
                self.mode = "local"
        self._events_file = self.base_dir / "data" / "local" / "dynamodb" / "events.jsonl"

    def put_event(self, item: dict) -> bool:
        """Store one event dict; stamps ``stored_at``. Returns True."""
        record = dict(item)
        record["stored_at"] = utcnow_iso()
        if self.mode == "aws" and self._table is not None:
            self._table.put_item(Item=record)
        else:
            append_jsonl(self._events_file, record)
        return True

    def query_events(self, filters: Optional[dict] = None) -> list[dict]:
        """Return stored events, optionally filtered by exact match on each
        key in ``filters``."""
        if self.mode == "aws" and self._table is not None:
            resp = self._table.scan()
            items = resp.get("Items", [])
            while "LastEvaluatedKey" in resp:  # pragma: no cover - demo scale
                resp = self._table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
                items.extend(resp.get("Items", []))
        else:
            items = read_jsonl(self._events_file)
        if not filters:
            return list(items)
        return [
            it for it in items
            if all(it.get(k) == v for k, v in filters.items())
        ]

    def get_pending_approvals(self) -> list[dict]:
        """Return stored items whose ``approval_status`` is ``"pending"``."""
        return self.query_events({"approval_status": "pending"})

    def set_approval_status(
        self,
        approval_id: str,
        status: str,
        decided_by: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Optional[dict]:
        """Update the approval record with ``approval_id``; returns the
        updated record, or None if no such record exists."""
        updates = {
            "approval_status": status,
            "decided_at": utcnow_iso(),
        }
        if decided_by is not None:
            updates["decided_by"] = decided_by
        if note is not None:
            updates["decision_note"] = note
        if self.mode == "aws" and self._table is not None:
            expr = "SET approval_status=:s, decided_at=:t"
            vals = {":s": status, ":t": updates["decided_at"]}
            if decided_by is not None:
                expr += ", decided_by=:b"
                vals[":b"] = decided_by
            if note is not None:
                expr += ", decision_note=:n"
                vals[":n"] = note
            try:
                resp = self._table.update_item(
                    Key={"approval_id": approval_id},
                    UpdateExpression=expr,
                    ExpressionAttributeValues=vals,
                    ReturnValues="ALL_NEW",
                )
                return resp.get("Attributes")
            except Exception:
                return None
        records = read_jsonl(self._events_file)
        updated: Optional[dict] = None
        for rec in records:
            if rec.get("approval_id") == approval_id:
                rec.update(updates)
                updated = rec
        if updated is None:
            return None
        self._events_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._events_file, "w", encoding="utf-8") as fh:
            import json

            for rec in records:
                fh.write(json.dumps(rec, default=str) + "\n")
        return updated
