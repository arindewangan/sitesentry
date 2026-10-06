"""Human-readable incident summaries for SiteSentry.

``summarize_incidents()`` is rule-based by default. If
``SITENTRY_BEDROCK=1`` it tries Amazon Bedrock (``bedrock-runtime``
``converse``) with the model id from ``SITENTRY_BEDROCK_MODEL`` — defaulting
to the documented placeholder ``anthropic.claude-3-5-sonnet-20240620-v1:0``
— and falls back to the rule-based summary on ANY exception. It never
raises.
"""

from __future__ import annotations

import os
from collections import Counter

#: Placeholder model id used when SITENTRY_BEDROCK_MODEL is unset. This is
#: only a default string; no Bedrock call is made unless SITENTRY_BEDROCK=1.
DEFAULT_BEDROCK_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"


def _rule_based_summary(events: list[dict]) -> str:
    total = len(events)
    by_type = Counter(e.get("type", "UNKNOWN") for e in events)
    by_sev = Counter(
        (e.get("decision") or {}).get("severity", "unknown") for e in events
    )
    offenders = Counter()
    for e in events:
        dec = e.get("decision") or {}
        key = (
            e.get("site_id", "?"),
            dec.get("subject_key") or f"track:{e.get('track_id', '?')}",
            e.get("type", "UNKNOWN"),
        )
        offenders[key] += 1

    lines = [f"SiteSentry incident summary: {total} event(s) recorded."]
    if by_type:
        lines.append(
            "By type: " + ", ".join(f"{t} x{n}" for t, n in by_type.most_common())
        )
    if by_sev:
        lines.append(
            "By severity: "
            + ", ".join(f"{s} x{n}" for s, n in by_sev.most_common())
        )
    top = offenders.most_common(3)
    if top:
        lines.append("Top repeat offenders:")
        for (site, subject, etype), n in top:
            lines.append(f"  - {subject} @ {site}: {etype} x{n}")
    approvals = [e for e in events if e.get("approval_status") == "pending"]
    if approvals:
        lines.append(f"{len(approvals)} approval(s) still pending human review.")
    return "\n".join(lines)


def _bedrock_summary(events: list[dict]) -> str:
    """Try a Bedrock converse call; raise on any failure so the caller can
    fall back to the rule-based summary."""
    import boto3

    model = os.environ.get("SITENTRY_BEDROCK_MODEL", DEFAULT_BEDROCK_MODEL)
    region = os.environ.get("AWS_REGION") or os.environ.get(
        "AWS_DEFAULT_REGION", "us-east-1"
    )
    client = boto3.client("bedrock-runtime", region_name=region)
    prompt = (
        "You are a construction-site safety assistant. Summarize these "
        "safety events for a site supervisor: counts by type and severity, "
        "and the top repeat offenders. Keep it under 150 words.\n\n"
        + "\n".join(
            f"- {e.get('type')} severity={(e.get('decision') or {}).get('severity')} "
            f"subject={(e.get('decision') or {}).get('subject_key')} "
            f"site={e.get('site_id')}"
            for e in events[:50]
        )
    )
    resp = client.converse(
        modelId=model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    return resp["output"]["message"]["content"][0]["text"]


def summarize_incidents(events: list[dict]) -> str:
    """Return a human-readable summary of ``events``.

    Uses Amazon Bedrock when ``SITENTRY_BEDROCK=1``; falls back to a
    rule-based summary on any exception. Never raises.
    """
    try:
        if os.environ.get("SITENTRY_BEDROCK") == "1":
            text = _bedrock_summary(events)
            if text:
                return text
        return _rule_based_summary(events)
    except Exception:
        return _rule_based_summary(events)
