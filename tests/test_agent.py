"""Tests for the SiteSentry agentic loop (policy, loop, approvals, summary).

No network, no cv2, no real AWS calls: all clients run in local mode
(``SITENTRY_AWS=off``) against a tmp dir.
"""

import os

import pytest

os.environ["SITENTRY_AWS"] = "off"

from agent import AgentLoop, ActionExecutor, decide, reset_ledger, summarize_incidents
from aws import make_clients


@pytest.fixture
def clients(tmp_path):
    return make_clients(base_dir=tmp_path)


@pytest.fixture
def loop(clients):
    reset_ledger()
    actions = ActionExecutor(
        s3=clients["s3"],
        dynamodb=clients["dynamodb"],
        sns=clients["sns"],
        rekognition=clients["rekognition"],
        cloudwatch=clients["cloudwatch"],
    )
    return AgentLoop(actions, clients["dynamodb"], site_id="site-a")


def _event(etype, track_id=7, **kw):
    e = {
        "event_id": f"e-{etype}-{track_id}",
        "type": etype,
        "track_id": track_id,
        "evidence": f"vision: person#{track_id} {etype.lower()} conf=0.91",
    }
    e.update(kw)
    return e


# ---------------------------------------------------------------- policy ---
def test_no_helmet_escalation_ladder():
    reset_ledger()
    ctx = {"site_id": "site-a"}
    d1 = decide(_event("NO_HELMET"), ctx)
    assert (d1["action"], d1["severity"], d1["needs_approval"]) == (
        "LOG_ARCHIVE",
        "medium",
        False,
    )
    d2 = decide(_event("NO_HELMET"), ctx)
    assert (d2["action"], d2["severity"], d2["needs_approval"]) == (
        "SNS_ALERT",
        "high",
        False,
    )
    d3 = decide(_event("NO_HELMET"), ctx)
    assert (d3["action"], d3["severity"], d3["needs_approval"]) == (
        "REQUEST_APPROVAL",
        "critical",
        True,
    )
    d4 = decide(_event("NO_HELMET"), ctx)  # 4th stays critical
    assert d4["needs_approval"] and d4["severity"] == "critical"


def test_no_vest_first_then_repeat():
    reset_ledger()
    ctx = {"site_id": "site-a"}
    d1 = decide(_event("NO_VEST", track_id=3), ctx)
    assert (d1["action"], d1["severity"]) == ("LOG_ONLY", "low")
    d2 = decide(_event("NO_VEST", track_id=3), ctx)
    assert (d2["action"], d2["severity"]) == ("SNS_ALERT", "medium")


def test_zone_intrusion_always_high():
    reset_ledger()
    ctx = {"site_id": "site-a"}
    d1 = decide(_event("ZONE_INTRUSION", track_id=9), ctx)
    d2 = decide(_event("ZONE_INTRUSION", track_id=9), ctx)
    for d in (d1, d2):
        assert d["action"] == "SNS_ALERT" and d["severity"] == "high"
        assert not d["needs_approval"]


def test_fall_suspected_critical_with_approval():
    reset_ledger()
    d = decide(_event("FALL_SUSPECTED", track_id=1), {"site_id": "site-a"})
    assert d["action"] == "REQUEST_APPROVAL"
    assert d["severity"] == "critical"
    assert d["needs_approval"] is True


def test_ledger_reset_and_per_subject_isolation():
    reset_ledger()
    decide(_event("NO_HELMET", track_id=1), {"site_id": "site-a"})
    d_other = decide(_event("NO_HELMET", track_id=2), {"site_id": "site-a"})
    assert d_other["offense_count"] == 1  # different subject -> own count
    reset_ledger()
    d = decide(_event("NO_HELMET", track_id=1), {"site_id": "site-a"})
    assert d["offense_count"] == 1


def test_decision_cites_vision_evidence():
    reset_ledger()
    d = decide(_event("NO_HELMET"), {"site_id": "site-a"})
    assert "helmet" in d["message"].lower() or "vision" in d["message"].lower()
    assert d["evidence"] is not None


# ---------------------------------------------------------------- loop -----
FAKE_FRAME = {
    "frame_id": 42,
    "events": [
        {
            "event_id": "evt-1",
            "type": "NO_HELMET",
            "track_id": 7,
            "evidence": "vision: person#7 helmet_conf=0.12 (no helmet)",
        },
        {
            "event_id": "evt-2",
            "type": "ZONE_INTRUSION",
            "track_id": 9,
            "evidence": "vision: person#9 bbox inside zone polygon Z1",
        },
    ],
    "jpeg_bytes": b"\xff\xd8fake-jpeg-bytes",
    "pipeline_ppe": {"helmet": False},
}


def test_process_trace_order_and_vision_link(loop):
    import copy

    out = loop.process(copy.deepcopy(FAKE_FRAME))
    assert len(out["events"]) == 2
    assert len(out["decisions"]) == 2
    stages = [t["stage"] for t in out["trace"]]
    # first occurrence order must be perceive -> plan -> act
    assert stages.index("perceive") < stages.index("plan") < stages.index("act")
    assert set(stages) == {"perceive", "plan", "act"}
    # decisions reference the vision evidence
    for dec, evt in zip(out["decisions"], out["events"]):
        assert evt["evidence"] in dec["message"]
    # SNS alert fired for the ZONE_INTRUSION (high severity)
    kinds = [a["kind"] for a in out["actions_taken"]]
    assert "log_event" in kinds and "send_alert" in kinds
    # evidence archived for severity >= medium
    assert any(a["kind"] == "archive_evidence" for a in out["actions_taken"])


def test_process_second_opinion_sampling(loop):
    import copy

    sampling_loop = AgentLoop(
        loop.actions, loop.event_store, rekognition_sample_rate=2
    )
    out1 = sampling_loop.process(copy.deepcopy(FAKE_FRAME))
    assert not [a for a in out1["actions_taken"] if a["kind"] == "second_opinion"]
    out2 = sampling_loop.process(copy.deepcopy(FAKE_FRAME))
    so = [a for a in out2["actions_taken"] if a["kind"] == "second_opinion"]
    assert len(so) == 1
    # local-simulated mode -> agreement must be None (never claim agreement)
    assert so[0]["verdict"]["agreement"] is None


def test_approve_reject_update_pending(loop):
    import copy

    frame = copy.deepcopy(FAKE_FRAME)
    frame["events"] = [
        {
            "event_id": "evt-fall",
            "type": "FALL_SUSPECTED",
            "track_id": 1,
            "evidence": "vision: person#1 horizontal, motionless 3s",
        }
    ]
    out = loop.process(frame)
    req = [a for a in out["actions_taken"] if a["kind"] == "request_approval"]
    assert len(req) == 1
    approval_id = req[0]["approval_id"]

    pending = loop.get_pending_approvals()
    assert any(p["approval_id"] == approval_id for p in pending)

    updated = loop.approve(approval_id, approved_by="supervisor@example.com")
    assert updated["approval_status"] == "approved"
    assert all(p["approval_id"] != approval_id for p in loop.get_pending_approvals())

    # reject path
    out2 = loop.process(copy.deepcopy(frame))
    approval_id2 = [
        a for a in out2["actions_taken"] if a["kind"] == "request_approval"
    ][0]["approval_id"]
    rejected = loop.reject(approval_id2, "supervisor@example.com", note="false alarm")
    assert rejected["approval_status"] == "rejected"
    assert rejected["decision_note"] == "false alarm"
    assert all(p["approval_id"] != approval_id2 for p in loop.get_pending_approvals())


def test_approve_unknown_id_returns_none(loop):
    assert loop.approve("apr-does-not-exist", "someone") is None
    assert loop.reject("apr-does-not-exist", "someone") is None


# ------------------------------------------------------------- summary -----
def test_summarize_incidents_non_empty():
    reset_ledger()
    events = [
        {"type": "NO_HELMET", "site_id": "site-a",
         "decision": {"severity": "medium", "subject_key": "track:7"}},
        {"type": "NO_HELMET", "site_id": "site-a",
         "decision": {"severity": "high", "subject_key": "track:7"}},
        {"type": "FALL_SUSPECTED", "site_id": "site-a", "approval_status": "pending",
         "decision": {"severity": "critical", "subject_key": "track:1"}},
    ]
    text = summarize_incidents(events)
    assert isinstance(text, str) and len(text) > 0
    assert "NO_HELMET" in text and "track:7" in text


def test_summarize_bedrock_failure_falls_back(monkeypatch):
    # SITENTRY_BEDROCK=1 with no creds must still return a rule-based summary.
    monkeypatch.setenv("SITENTRY_BEDROCK", "1")
    text = summarize_incidents([{"type": "NO_VEST", "site_id": "site-a",
                                 "decision": {"severity": "low"}}])
    assert isinstance(text, str) and "NO_VEST" in text
