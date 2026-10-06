"""Tests for the AWS client layer in LOCAL mode.

``SITENTRY_AWS=off`` forces every client into local fallback; each test
does a round-trip against a tmp dir passed as ``base_dir``. No network, no
credentials, no real AWS calls.
"""

import json
import os

import pytest

os.environ["SITENTRY_AWS"] = "off"

from aws import (
    CloudWatchClient,
    DynamoClient,
    RekognitionClient,
    S3Client,
    SNSClient,
    make_clients,
)


@pytest.fixture(autouse=True)
def _local_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("SITENTRY_AWS", "off")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _clients(tmp_path):
    return make_clients(base_dir=tmp_path)


def test_make_clients_local_modes(tmp_path):
    clients = _clients(tmp_path)
    assert set(clients) == {"s3", "dynamodb", "sns", "rekognition", "cloudwatch"}
    for c in clients.values():
        assert c.mode == "local"


def test_s3_upload_round_trip(tmp_path):
    s3 = S3Client(base_dir=tmp_path)
    assert s3.mode == "local"
    uri = s3.upload_bytes("evidence/site-a/evt-1.jpg", b"\xff\xd8jpeg",
                          content_type="image/jpeg")
    assert uri == "local://data/local/s3/evidence/site-a/evt-1.jpg"
    assert (tmp_path / "data" / "local" / "s3" / "evidence" / "site-a" / "evt-1.jpg").read_bytes() == b"\xff\xd8jpeg"


def test_dynamodb_put_query_pending(tmp_path):
    db = DynamoClient(base_dir=tmp_path)
    assert db.mode == "local"
    assert db.put_event({"event_id": "e1", "type": "NO_HELMET", "track_id": 7}) is True
    assert db.put_event({"event_id": "e2", "type": "NO_VEST",
                         "approval_status": "pending",
                         "approval_id": "apr-1"}) is True
    found = db.query_events({"type": "NO_HELMET"})
    assert len(found) == 1 and found[0]["event_id"] == "e1"
    assert "stored_at" in found[0]
    pending = db.get_pending_approvals()
    assert len(pending) == 1 and pending[0]["approval_id"] == "apr-1"
    updated = db.set_approval_status("apr-1", "approved", decided_by="boss")
    assert updated["approval_status"] == "approved"
    assert db.get_pending_approvals() == []


def test_sns_publish_outbox(tmp_path):
    sns = SNSClient(base_dir=tmp_path)
    assert sns.mode == "local"
    mid = sns.publish("subject-A", "hello supervisor")
    assert mid.startswith("local-")
    lines = (tmp_path / "data" / "local" / "sns" / "outbox.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["message_id"] == mid and rec["subject"] == "subject-A"


def test_rekognition_local_simulated(tmp_path):
    rek = RekognitionClient(base_dir=tmp_path)
    assert rek.mode == "local"
    res = rek.detect_ppe(b"\xff\xd8jpeg")
    assert res["source"] == "local-simulated"
    assert res["persons"] == []
    assert "Rekognition unavailable offline" in res["note"]


def test_cloudwatch_metrics_and_logs(tmp_path):
    cw = CloudWatchClient(base_dir=tmp_path)
    assert cw.mode == "local"
    assert cw.put_metric("LatencyMs", 12.5) is True
    assert cw.log_event("agent started") is True
    metrics = (tmp_path / "data" / "local" / "cloudwatch" / "metrics.jsonl").read_text().strip().splitlines()
    logs = (tmp_path / "data" / "local" / "cloudwatch" / "log-events.jsonl").read_text().strip().splitlines()
    assert len(metrics) == 1 and json.loads(metrics[0])["name"] == "LatencyMs"
    assert len(logs) == 1 and json.loads(logs[0])["message"] == "agent started"
