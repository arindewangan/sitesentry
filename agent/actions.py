"""Action executor: the agent's hands on the AWS (or local) clients.

Each method maps one agent-level action to concrete client calls:

* ``log_event``       -> DynamoDB event store (+ CloudWatch metric)
* ``archive_evidence``-> S3 evidence upload
* ``send_alert``      -> SNS notification to the supervisor
* ``second_opinion``  -> Rekognition PPE check vs the pipeline's verdict
* ``request_approval``-> pending approval record + supervisor notification
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActionExecutor:
    """Executes agent actions against the injected AWS/local clients."""

    def __init__(
        self,
        s3,
        dynamodb,
        sns,
        rekognition,
        cloudwatch,
        supervisor_email: str = "supervisor@example.com",
        evidence_prefix: str = "evidence/",
    ) -> None:
        """Wire up the five clients; ``supervisor_email`` receives alert
        notifications."""
        self.s3 = s3
        self.dynamodb = dynamodb
        self.sns = sns
        self.rekognition = rekognition
        self.cloudwatch = cloudwatch
        self.supervisor_email = supervisor_email
        self.evidence_prefix = evidence_prefix

    # -- act: log ----------------------------------------------------------
    def log_event(self, event: dict, decision: dict) -> dict:
        """Persist the event + decision to the event store.

        Returns the stored item (with ``approval_status`` set to
        ``"pending"`` when the decision needs approval, else ``"none"``).
        """
        item = dict(event)
        item["decision"] = decision
        item["site_id"] = item.get("site_id") or decision.get("site_id", "site-a")
        # The event itself is never "pending": the approval workflow lives in
        # the separate approval record created by request_approval(), which is
        # what get_pending_approvals() returns.
        item["approval_status"] = "none"
        self.dynamodb.put_event(item)
        self.cloudwatch.put_metric("EventsLogged", 1, unit="Count")
        return item

    # -- act: archive ------------------------------------------------------
    def archive_evidence(self, jpeg_bytes: Optional[bytes], event: dict) -> Optional[str]:
        """Upload the evidence JPEG to S3 (or the local store).

        Returns the object URI, or None when no image bytes were supplied.
        """
        if not jpeg_bytes:
            return None
        event_id = event.get("event_id") or f"evt-{uuid.uuid4().hex[:8]}"
        key = (
            f"{self.evidence_prefix}"
            f"{event.get('site_id', 'site-a')}/"
            f"{event.get('type', 'event')}/"
            f"{event_id}.jpg"
        )
        return self.s3.upload_bytes(key, jpeg_bytes, content_type="image/jpeg")

    # -- act: alert --------------------------------------------------------
    def send_alert(self, event: dict, decision: dict, summary: str) -> Optional[str]:
        """Publish an SNS alert to the supervisor. Returns the message id."""
        subject = (
            f"[SiteSentry:{decision.get('severity', 'info').upper()}] "
            f"{event.get('type', 'EVENT')} at {event.get('site_id', 'site-a')}"
        )
        body = (
            f"SiteSentry alert -> {self.supervisor_email}\n\n"
            f"Decision: {decision.get('action')} "
            f"(severity {decision.get('severity')})\n"
            f"Policy: {decision.get('policy_rule', {}).get('rationale', '')}\n\n"
            f"{summary}\n\n"
            f"Event: {event.get('type')} "
            f"subject={decision.get('subject_key')} "
            f"offense#{decision.get('offense_count')}\n"
            f"Vision evidence: {event.get('evidence', 'n/a')}"
        )
        message_id = self.sns.publish(subject, body)
        self.cloudwatch.put_metric("AlertsSent", 1, unit="Count")
        return message_id

    # -- act: second opinion -----------------------------------------------
    def second_opinion(
        self, jpeg_bytes: Optional[bytes], event: dict, pipeline_ppe: dict
    ) -> dict:
        """Ask Rekognition for an independent PPE verdict and compare it to
        the pipeline's helmet verdict.

        Returns ``{"agreement": bool|None, "detail": dict}``. ``agreement``
        is None when the comparison cannot be made (local-simulated mode or
        no persons / no image bytes) — never claim agreement offline.
        """
        if not jpeg_bytes:
            return {
                "agreement": None,
                "detail": {"reason": "no image bytes available"},
            }
        result = self.rekognition.detect_ppe(jpeg_bytes)
        if result.get("source") == "local-simulated":
            return {"agreement": None, "detail": result}

        pipeline_helmet = bool(pipeline_ppe.get("helmet"))
        rek_helmet: Optional[bool] = None
        persons = result.get("persons", [])
        if persons:
            rek_helmet = any(
                det.get("Type") == "HEAD_COVER"
                and (det.get("CoversBodyPart") or {}).get("Value") is True
                for person in persons
                for part in person.get("BodyParts", [])
                if part.get("Name") == "HEAD"
                for det in part.get("EquipmentDetections", [])
            )
        agreement = None if rek_helmet is None else (rek_helmet == pipeline_helmet)
        self.cloudwatch.put_metric("SecondOpinions", 1, unit="Count")
        return {
            "agreement": agreement,
            "detail": {
                "source": result.get("source"),
                "pipeline_helmet": pipeline_helmet,
                "rekognition_helmet": rek_helmet,
                "persons_seen": len(persons),
            },
        }

    # -- act: approval -----------------------------------------------------
    def request_approval(self, decision: dict, event: dict) -> dict:
        """Enqueue a human approval request (work stoppage / emergency
        response). Stored with ``approval_status="pending"`` so
        ``get_pending_approvals()`` finds it. Returns the approval dict."""
        approval = {
            "record_kind": "approval",
            "approval_id": f"apr-{uuid.uuid4().hex[:8]}",
            "approval_type": "work-stoppage",
            "approval_status": "pending",
            "requested_at": _utcnow_iso(),
            "requested_by": "sitesentry-agent",
            "supervisor_email": self.supervisor_email,
            "decision": decision,
            "event": event,
            "event_type": event.get("type"),
            "site_id": event.get("site_id", "site-a"),
        }
        self.dynamodb.put_event(approval)
        self.sns.publish(
            subject=(
                f"[SiteSentry] APPROVAL NEEDED: {decision.get('action')} "
                f"for {event.get('type')} at {approval['site_id']}"
            ),
            message=(
                f"Approval {approval['approval_id']} requested by "
                f"sitesentry-agent for {self.supervisor_email}.\n\n"
                f"Decision: {decision.get('action')} "
                f"(severity {decision.get('severity')})\n"
                f"Policy rationale: "
                f"{decision.get('policy_rule', {}).get('rationale', '')}\n"
                f"Message: {decision.get('message')}"
            ),
        )
        self.cloudwatch.put_metric("ApprovalsRequested", 1, unit="Count")
        return approval
