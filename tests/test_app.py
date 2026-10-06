"""Flask dashboard tests for the SiteSentry web app."""
import io
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402


def make_client():
    application = app_module.create_app()
    application.config["TESTING"] = True
    return application.test_client()


def test_index_200_contains_branding():
    c = make_client()
    r = c.get("/")
    assert r.status_code == 200
    assert b"SiteSentry" in r.data


def test_health_has_status_key():
    c = make_client()
    r = c.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert "status" in body
    assert "aws_mode" in body
    assert "engine" in body


def test_upload_generated_jpeg():
    c = make_client()
    rng = np.random.default_rng(42)
    noise = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", noise)
    assert ok
    data = {"file": (io.BytesIO(buf.tobytes()), "noise.jpg")}
    r = c.post("/api/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_data(as_text=True)[:300]
    body = r.get_json()
    assert "events" in body
    assert "annotated_image" in body
    assert "metrics" in body
    assert "per_frame_ms" in body


def test_stream_200_or_503():
    c = make_client()
    r = c.get("/api/stream")
    assert r.status_code in (200, 503)
    if r.status_code == 503:
        body = r.get_json()
        assert body is not None and "error" in body


def test_supervisor_page_200():
    c = make_client()
    r = c.get("/supervisor")
    assert r.status_code == 200
    assert b"Supervisor" in r.data


def test_aux_endpoints():
    c = make_client()
    for path in ("/api/events", "/api/trace", "/api/pending",
                 "/api/history", "/api/summary", "/api/agreement",
                 "/api/engine"):
        r = c.get(path)
        assert r.status_code == 200, path
    r = c.post("/api/config", json={"blur_faces": True})
    assert r.status_code == 200
    assert r.get_json()["config"]["blur_faces"] is True
    r = c.get("/api/sample/99")
    assert r.status_code == 404
    assert "error" in r.get_json()
