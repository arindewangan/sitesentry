"""Safety policy table and repeat-offender ledger for SiteSentry.

The policy table is an auditable list of rules: each rule maps an event
type + offense count range to an action, severity, whether a human must
approve, and a human-readable rationale. ``decide()`` consults it so every
agent decision can cite the exact rule that produced it.

The offense ledger is a module-level dict keyed
``(site_id, subject_key, event_type)``. ``subject_key`` is
``f"track:{track_id}"`` so repeat violations by the same tracked person
escalate. ``reset_ledger()`` exists for tests.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Auditable policy table
# ---------------------------------------------------------------------------
POLICY_TABLE: list[dict] = [
    {
        "event_type": "NO_HELMET",
        "min_offenses": 1,
        "max_offenses": 1,
        "action": "LOG_ARCHIVE",
        "severity": "medium",
        "needs_approval": False,
        "rationale": (
            "First helmet violation: log it and archive the evidence frame "
            "for the safety record."
        ),
    },
    {
        "event_type": "NO_HELMET",
        "min_offenses": 2,
        "max_offenses": 2,
        "action": "SNS_ALERT",
        "severity": "high",
        "needs_approval": False,
        "rationale": (
            "Repeat helmet offender: page the supervisor immediately; "
            "keep archiving evidence."
        ),
    },
    {
        "event_type": "NO_HELMET",
        "min_offenses": 3,
        "max_offenses": None,  # 3rd and beyond
        "action": "REQUEST_APPROVAL",
        "severity": "critical",
        "needs_approval": True,
        "rationale": (
            "Third strike: recommend work stoppage for this worker; a "
            "human supervisor must approve before enforcement."
        ),
    },
    {
        "event_type": "NO_VEST",
        "min_offenses": 1,
        "max_offenses": 1,
        "action": "LOG_ONLY",
        "severity": "low",
        "needs_approval": False,
        "rationale": "First hi-vis vest violation: log only, no page.",
    },
    {
        "event_type": "NO_VEST",
        "min_offenses": 2,
        "max_offenses": None,  # 2nd and beyond
        "action": "SNS_ALERT",
        "severity": "medium",
        "needs_approval": False,
        "archive": True,
        "rationale": (
            "Repeat vest offender: alert the supervisor and archive the "
            "evidence frame."
        ),
    },
    {
        "event_type": "ZONE_INTRUSION",
        "min_offenses": 1,
        "max_offenses": None,
        "action": "SNS_ALERT",
        "severity": "high",
        "needs_approval": False,
        "archive": True,
        "rationale": (
            "Restricted-zone entry is always safety-critical: alert and "
            "archive, every time."
        ),
    },
    {
        "event_type": "FALL_SUSPECTED",
        "min_offenses": 1,
        "max_offenses": None,
        "action": "REQUEST_APPROVAL",
        "severity": "critical",
        "needs_approval": True,
        "immediate_alert": True,
        "rationale": (
            "Suspected fall: request supervisor approval for emergency "
            "response AND page immediately — a possible injury cannot wait."
        ),
    },
]

#: Fallback rule for event types not present in POLICY_TABLE.
_FALLBACK_RULE = {
    "event_type": "*",
    "min_offenses": 1,
    "max_offenses": None,
    "action": "LOG_ONLY",
    "severity": "low",
    "needs_approval": False,
    "rationale": "Unknown event type: logged for review.",
}

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# ---------------------------------------------------------------------------
# Offense ledger
# ---------------------------------------------------------------------------
_OFFENSE_LEDGER: dict[tuple[str, str, str], int] = {}


def record_offense(site_id: str, subject_key: str, event_type: str) -> int:
    """Record one offense and return the new cumulative count for
    ``(site_id, subject_key, event_type)``."""
    key = (site_id, subject_key, event_type)
    _OFFENSE_LEDGER[key] = _OFFENSE_LEDGER.get(key, 0) + 1
    return _OFFENSE_LEDGER[key]


def reset_ledger() -> None:
    """Clear the offense ledger (used by tests)."""
    _OFFENSE_LEDGER.clear()


def get_ledger() -> dict:
    """Return a copy of the current ledger (read-only audit view)."""
    return dict(_OFFENSE_LEDGER)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------
def _match_rule(event_type: str, offense_count: int) -> dict:
    for rule in POLICY_TABLE:
        if rule["event_type"] != event_type:
            continue
        max_off = rule["max_offenses"]
        if rule["min_offenses"] <= offense_count and (
            max_off is None or offense_count <= max_off
        ):
            return rule
    return _FALLBACK_RULE


def decide(event: dict, context: dict | None = None) -> dict:
    """Decide what to do about a perception ``event``.

    ``event`` must carry ``type`` (or ``event_type``) and ideally
    ``track_id``; ``context`` may carry ``site_id`` (default ``"site-a"``).

    Returns ``{"action", "severity", "needs_approval", "message",
    "policy_rule", ...}``. The ``message`` always cites the vision
    evidence when the event carries an ``evidence`` field.
    """
    context = context or {}
    event_type = event.get("type") or event.get("event_type") or "UNKNOWN"
    site_id = context.get("site_id", "site-a")
    track_id = event.get("track_id", "unknown")
    subject_key = f"track:{track_id}"

    offense_count = record_offense(site_id, subject_key, event_type)
    rule = _match_rule(event_type, offense_count)

    severity = rule["severity"]
    archive = rule.get(
        "archive", _SEVERITY_RANK.get(severity, 0) >= _SEVERITY_RANK["medium"]
    )
    message = (
        f"{event_type} (offense #{offense_count} for {subject_key} at "
        f"{site_id}): action={rule['action']}, severity={severity}. "
        f"{rule['rationale']}"
    )
    evidence = event.get("evidence")
    if evidence:
        message += f" Vision evidence: {evidence}"

    return {
        "action": rule["action"],
        "severity": severity,
        "needs_approval": bool(rule["needs_approval"]),
        "message": message,
        "policy_rule": dict(rule),
        "offense_count": offense_count,
        "subject_key": subject_key,
        "site_id": site_id,
        "archive": bool(archive),
        "immediate_alert": bool(rule.get("immediate_alert", False)),
        "evidence": evidence,
    }
