"""CloudWatch client with a local (offline) fallback.

AWS mode sends metrics via ``cloudwatch.put_metric_data`` and log lines
via CloudWatch Logs ``put_log_events``. Local mode appends to JSONL files
under ``data/local/cloudwatch/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ._common import DEFAULT_BASE_DIR, append_jsonl, resolve_mode, utcnow_iso


class CloudWatchClient:
    """Metrics + log client (real CloudWatch or local JSONL fallback)."""

    def __init__(
        self,
        namespace: str = "SiteSentry",
        base_dir: Optional[Path | str] = None,
    ) -> None:
        """Create the client; ``base_dir`` overrides the project root used
        for local-mode storage (tests pass a tmp dir)."""
        self.namespace = namespace
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
        self.mode = resolve_mode()
        self._metrics = None
        self._logs = None
        if self.mode == "aws":
            try:
                import boto3

                self._metrics = boto3.client("cloudwatch")
            except Exception:
                self.mode = "local"
        self._metrics_file = (
            self.base_dir / "data" / "local" / "cloudwatch" / "metrics.jsonl"
        )
        self._log_file = (
            self.base_dir / "data" / "local" / "cloudwatch" / "log-events.jsonl"
        )

    def put_metric(self, name: str, value: float, unit: str = "Milliseconds") -> bool:
        """Emit one metric data point. Returns True."""
        if self.mode == "aws" and self._metrics is not None:
            self._metrics.put_metric_data(
                Namespace=self.namespace,
                MetricData=[{"MetricName": name, "Value": value, "Unit": unit}],
            )
        else:
            append_jsonl(
                self._metrics_file,
                {
                    "namespace": self.namespace,
                    "name": name,
                    "value": value,
                    "unit": unit,
                    "ts": utcnow_iso(),
                    "mode": "local",
                },
            )
        return True

    def log_event(self, message: str) -> bool:
        """Write one log line. Returns True."""
        if self.mode == "aws" and self._metrics is not None:
            self._put_logs_event(message)
        else:
            append_jsonl(
                self._log_file,
                {"message": message, "ts": utcnow_iso(), "mode": "local"},
            )
        return True

    # -- AWS CloudWatch Logs plumbing -------------------------------------
    def _logs_client(self):  # lazy: only built when AWS logging is used
        if self._logs is None:
            import boto3

            self._logs = boto3.client("logs")
        return self._logs

    def _put_logs_event(self, message: str) -> None:
        """Best-effort put_log_events to ``<namespace>/agent`` log stream."""
        import time

        from botocore.exceptions import ClientError

        logs = self._logs_client()
        group, stream = self.namespace, "agent"
        try:
            logs.create_log_group(logGroupName=group)
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ResourceAlreadyExistsException":
                raise
        try:
            logs.create_log_stream(logGroupName=group, logStreamName=stream)
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ResourceAlreadyExistsException":
                raise
        kwargs: dict = {
            "logGroupName": group,
            "logStreamName": stream,
            "logEvents": [{"timestamp": int(time.time() * 1000), "message": message}],
        }
        try:
            desc = logs.describe_log_streams(
                logGroupName=group, logStreamNamePrefix=stream
            )["logStreams"][0]
            if "uploadSequenceToken" in desc:
                kwargs["sequenceToken"] = desc["uploadSequenceToken"]
        except Exception:
            pass
        logs.put_log_events(**kwargs)
