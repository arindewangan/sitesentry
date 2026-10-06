"""The SiteSentry agentic loop: perceive -> plan -> act.

``AgentLoop.process(frame_result)`` runs one full cycle:

1. **perceive** — take the vision pipeline's frame result (events with
   vision evidence attached).
2. **plan** — run each event through ``policy.decide()`` to get an
   auditable action + severity.
3. **act** — log to the event store, archive evidence (severity >= medium),
   publish SNS alerts, run a Rekognition second opinion on every Nth
   flagged frame, and enqueue human approvals where the policy demands one.

Every stage appends to an inspectable trace so the chain
``vision evidence -> policy decision -> action`` is always visible.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .policy import decide


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentLoop:
    """Perceive-plan-act loop over vision pipeline frame results."""

    def __init__(
        self,
        actions,
        event_store,
        site_id: str = "site-a",
        trace_limit: int = 300,
        rekognition_sample_rate: int = 5,
    ) -> None:
        """``actions`` is an ``ActionExecutor``; ``event_store`` is a
        ``DynamoClient`` (or any object with ``put_event``,
        ``get_pending_approvals`` and ``set_approval_status``)."""
        self.actions = actions
        self.event_store = event_store
        self.site_id = site_id
        self.trace_limit = trace_limit
        self.rekognition_sample_rate = max(1, rekognition_sample_rate)
        self._trace: list[dict] = []
        self._flagged_frames = 0

    # -- trace --------------------------------------------------------------
    def _trace_add(self, stage: str, detail: str) -> None:
        self._trace.append({"ts": _utcnow_iso(), "stage": stage, "detail": detail})
        if len(self._trace) > self.trace_limit:
            del self._trace[: len(self._trace) - self.trace_limit]

    def get_trace(self) -> list[dict]:
        """Return a copy of the perceive/plan/act trace."""
        return list(self._trace)

    # -- main cycle ----------------------------------------------------------
    def process(self, frame_result: dict) -> dict:
        """Run one perceive-plan-act cycle over a pipeline frame result.

        ``frame_result``: ``{"frame_id", "events": [...],
        "jpeg_bytes": b..., "pipeline_ppe": {...}}``. Returns
        ``{"events", "decisions", "actions_taken", "trace"}``.
        """
        events = list(frame_result.get("events") or [])
        decisions: list[dict] = []
        actions_taken: list[dict] = []
        jpeg_bytes = frame_result.get("jpeg_bytes")
        frame_id = frame_result.get("frame_id", "?")

        # ---- perceive -----------------------------------------------------
        if events:
            evidence_bits = "; ".join(
                f"{e.get('type')}: {e.get('evidence', 'no evidence text')}"
                for e in events
            )
            self._trace_add(
                "perceive",
                f"frame {frame_id}: vision pipeline flagged {len(events)} "
                f"event(s) -> {evidence_bits}",
            )
        else:
            self._trace_add("perceive", f"frame {frame_id}: no events flagged")

        flagged_this_frame = False
        for event in events:
            event = dict(event)
            event.setdefault("site_id", self.site_id)

            # ---- plan -----------------------------------------------------
            decision = decide(event, {"site_id": self.site_id})
            decisions.append(decision)
            self._trace_add(
                "plan",
                f"{event.get('type')} subject={decision['subject_key']} "
                f"offense#{decision['offense_count']} -> policy action="
                f"{decision['action']} severity={decision['severity']} "
                f"needs_approval={decision['needs_approval']} "
                f"[rule: {decision['policy_rule'].get('rationale', '')}]",
            )

            # ---- act ------------------------------------------------------
            item = self.actions.log_event(event, decision)
            actions_taken.append({"kind": "log_event", "event_id": item.get("event_id")})

            if decision.get("archive"):
                uri = self.actions.archive_evidence(jpeg_bytes, event)
                actions_taken.append({"kind": "archive_evidence", "uri": uri})
                self._trace_add("act", f"archived evidence -> {uri}")

            if decision["action"] == "SNS_ALERT" or decision.get("immediate_alert"):
                summary = (
                    f"frame {frame_id}: {event.get('type')} "
                    f"({decision['subject_key']}, offense "
                    f"#{decision['offense_count']})"
                )
                mid = self.actions.send_alert(event, decision, summary)
                actions_taken.append({"kind": "send_alert", "message_id": mid})
                self._trace_add("act", f"SNS alert published -> {mid}")

            if decision.get("severity") in ("medium", "high", "critical"):
                flagged_this_frame = True

            if decision.get("needs_approval"):
                approval = self.actions.request_approval(decision, event)
                actions_taken.append(
                    {
                        "kind": "request_approval",
                        "approval_id": approval["approval_id"],
                    }
                )
                self._trace_add(
                    "act",
                    f"approval requested -> {approval['approval_id']} "
                    f"(type={approval['approval_type']}, status=pending)",
                )

            self._trace_add(
                "act",
                f"event {event.get('event_id', '?')} ({event.get('type')}) "
                f"handled: action={decision['action']}",
            )

        # ---- second opinion: every Nth flagged frame -----------------------
        if flagged_this_frame:
            self._flagged_frames += 1
            if self._flagged_frames % self.rekognition_sample_rate == 0:
                pipeline_ppe = frame_result.get("pipeline_ppe") or {}
                verdict = self.actions.second_opinion(jpeg_bytes, {}, pipeline_ppe)
                actions_taken.append(
                    {"kind": "second_opinion", "verdict": verdict}
                )
                self._trace_add(
                    "act",
                    f"Rekognition second opinion on flagged frame #{self._flagged_frames}: "
                    f"agreement={verdict.get('agreement')} "
                    f"detail={verdict.get('detail')}",
                )

        return {
            "events": events,
            "decisions": decisions,
            "actions_taken": actions_taken,
            "trace": self.get_trace(),
        }

    # -- human approvals ------------------------------------------------------
    def approve(self, action_id: str, approved_by: str) -> Optional[dict]:
        """Approve a pending approval. If the approved request was a
        work-stoppage recommendation, send an escalated SNS. Returns the
        updated record, or None if the id is unknown."""
        updated = self.event_store.set_approval_status(
            action_id, "approved", decided_by=approved_by
        )
        if updated is None:
            self._trace_add("act", f"approve({action_id}): unknown approval id")
            return None
        decision = updated.get("decision", {}) or {}
        if (
            decision.get("action") == "REQUEST_APPROVAL"
            or updated.get("approval_type") == "work-stoppage"
        ):
            mid = self.actions.sns.publish(
                subject=(
                    f"[ESCALATED] Work-stoppage APPROVED by {approved_by} "
                    f"at {updated.get('site_id', self.site_id)}"
                ),
                message=(
                    f"Approval {action_id} approved by {approved_by}.\n"
                    f"Event: {updated.get('event_type')} "
                    f"decision={decision.get('action')} "
                    f"severity={decision.get('severity')}.\n"
                    f"Enforce the work stoppage per site safety procedure."
                ),
            )
            self._trace_add(
                "act",
                f"approval {action_id} APPROVED by {approved_by}; "
                f"escalated SNS -> {mid}",
            )
        else:
            self._trace_add(
                "act", f"approval {action_id} APPROVED by {approved_by}"
            )
        return updated

    def reject(self, action_id: str, approved_by: str, note: str = "") -> Optional[dict]:
        """Reject a pending approval with an optional note. Returns the
        updated record, or None if the id is unknown."""
        updated = self.event_store.set_approval_status(
            action_id, "rejected", decided_by=approved_by, note=note or None
        )
        if updated is None:
            self._trace_add("act", f"reject({action_id}): unknown approval id")
        else:
            self._trace_add(
                "act",
                f"approval {action_id} REJECTED by {approved_by}"
                + (f" note={note}" if note else ""),
            )
        return updated

    def get_pending_approvals(self) -> list[dict]:
        """Return all approval records still awaiting a human decision."""
        return self.event_store.get_pending_approvals()
