"""SNS client with a local (offline) fallback.

AWS mode publishes to a real SNS topic. Local mode appends the message to
``data/local/sns/outbox.jsonl`` and returns a ``local-<uuid>`` message id,
clearly labeled as local.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from ._common import DEFAULT_BASE_DIR, append_jsonl, local_dir, resolve_mode, utcnow_iso


class SNSClient:
    """Notification publisher (real SNS or local outbox fallback)."""

    def __init__(
        self,
        topic_arn: str = "",
        default_email: str = "",
        base_dir: Optional[Path | str] = None,
    ) -> None:
        """Create the client; ``base_dir`` overrides the project root used
        for local-mode storage (tests pass a tmp dir)."""
        self.topic_arn = topic_arn
        self.default_email = default_email
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
        self.mode = resolve_mode()
        self._client = None
        if self.mode == "aws":
            try:
                import boto3

                self._client = boto3.client("sns")
            except Exception:
                self.mode = "local"
        self._outbox = self.base_dir / "data" / "local" / "sns" / "outbox.jsonl"

    def publish(self, subject: str, message: str) -> str:
        """Publish a notification; returns the message id.

        AWS: real ``sns.publish`` message id. Local: ``"local-<uuid>"`` and
        the message is appended to the local outbox JSONL file.
        """
        if self.mode == "aws" and self._client is not None and self.topic_arn:
            kwargs = {"Subject": subject, "Message": message}
            if self.topic_arn:
                kwargs["TopicArn"] = self.topic_arn
            resp = self._client.publish(**kwargs)
            return resp.get("MessageId", "")
        message_id = f"local-{uuid.uuid4()}"
        append_jsonl(
            self._outbox,
            {
                "message_id": message_id,
                "subject": subject,
                "message": message,
                "ts": utcnow_iso(),
                "mode": "local",
            },
        )
        return message_id
