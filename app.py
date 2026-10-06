#!/usr/bin/env python3
"""SiteSentry - AI construction-site safety monitor (Flask dashboard).

Defensive by design: the sibling ``pipeline/``, ``agent/`` and ``aws/``
modules may not exist yet or may change API. Every import is guarded and
local stubs keep the dashboard serving pages + JSON, with ``/api/health``
reporting ``degraded`` whenever a stub is in use. Never leaks stack traces
to API clients.
"""

import base64
import glob
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ---------------------------------------------------------------------------
# Defensive imports of sibling modules
# ---------------------------------------------------------------------------

_pipeline_import_error = None
try:
    from pipeline import SafetyPipeline  # type: ignore
except Exception as exc:  # ModuleNotFoundError, ImportError, ...
    SafetyPipeline = None  # type: ignore
    _pipeline_import_error = exc

_agent_import_error = None
try:
    from agent import AgentLoop  # type: ignore
except Exception as exc:
    AgentLoop = None  # type: ignore
    _agent_import_error = exc

ActionExecutor = None  # type: ignore
try:
    from agent import ActionExecutor as _AE  # type: ignore
    ActionExecutor = _AE
except Exception:
    try:
        from pipeline import ActionExecutor as _AE2  # type: ignore
        ActionExecutor = _AE2
    except Exception:
        ActionExecutor = None

_aws_import_error = None
try:
    from aws import make_clients  # type: ignore
except Exception as exc:
    make_clients = None  # type: ignore
    _aws_import_error = exc

_real_annotate = None
try:
    from pipeline import annotate_frame as _real_annotate  # type: ignore
except Exception:
    _real_annotate = None


# ---------------------------------------------------------------------------
# Local stub fallbacks (API-compatible with the documented contracts)
# ---------------------------------------------------------------------------

class _StubPipeline:
    """process(frame) -> {"tracks","events","metrics","engine"}."""

    engine_name = "stub-local"

    def __init__(self, **kwargs):
        self.blur_faces = bool(kwargs.get("blur_faces", False))

    def process(self, frame):
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        ms = (time.perf_counter() - t0) * 1000.0
        return {
            "tracks": [],
            "events": [],
            "metrics": {"per_frame_ms": round(ms, 2), "frame_w": w, "frame_h": h},
            "engine": {"name": self.engine_name, "detectors": [], "mode": "stub"},
        }


class _StubActionExecutor:
    def __init__(self, aws_clients=None, **kwargs):
        self.aws_clients = aws_clients or {}

    def execute(self, action):
        return {"status": "simulated", "action": action}


class _StubAgentLoop:
    """process(frame_result); get_trace(); get_pending_approvals();
    approve(action_id, by); reject(action_id, by)."""

    def __init__(self, executor=None, aws_clients=None, **kwargs):
        self.executor = executor
        self.aws_clients = aws_clients or {}
        self._trace = []
        self._pending = {}
        self._history = []
        self._seq = 0
        self._tseq = 0
        self._log("act", "AgentLoop initialized (local stub) - approvals are simulated.")

    def _log(self, step, text):
        self._tseq += 1
        self._trace.append({
            "seq": self._tseq,
            "step": step,
            "text": text,
            "t": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._trace) > 500:
            self._trace = self._trace[-500:]

    def process(self, frame_result):
        frame_result = frame_result or {}
        events = frame_result.get("events") or []
        metrics = frame_result.get("metrics") or {}
        ms = metrics.get("per_frame_ms")
        ms_txt = f" in {ms:.1f} ms" if isinstance(ms, (int, float)) else ""
        self._log("perceive", f"Frame analyzed: {len(events)} event(s){ms_txt}.")
        decisions = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            sev = str(ev.get("severity", "low")).lower()
            etype = str(ev.get("type", "event"))
            if sev in ("high", "critical"):
                aid = self._queue_approval(ev)
                decisions.append({"action_id": aid, "decision": "pending",
                                  "reason": f"{etype} requires supervisor approval"})
                self._log("plan", f"High-severity '{etype}' -> queued approval {aid}.")
            else:
                self._log("plan", f"'{etype}' ({sev}) -> log only, no approval needed.")
        queued = len(decisions)
        self._log("act", f"Cycle complete: {queued} approval(s) queued, "
                         f"{len(events) - queued} event(s) logged.")
        return {"decisions": decisions}

    def _queue_approval(self, ev):
        self._seq += 1
        aid = f"act-{self._seq:04d}"
        self._pending[aid] = {
            "action_id": aid,
            "title": f"Acknowledge: {ev.get('type', 'incident')}",
            "detail": ev.get("message", ""),
            "severity": ev.get("severity", "high"),
            "event": {k: ev.get(k) for k in ("type", "severity", "message") if k in ev},
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "by": None,
        }
        return aid

    def get_trace(self):
        return list(self._trace)

    def get_pending_approvals(self):
        return [dict(v) for v in self._pending.values()]

    def get_history(self):
        return list(self._history)

    def _decide(self, action_id, decision, by):
        item = self._pending.pop(action_id, None)
        if item is None:
            for h in self._history:
                if h.get("action_id") == action_id:
                    return {"error": "already decided", "action_id": action_id,
                            "decision": h.get("status"), "by": h.get("by")}
            return {"error": "unknown action_id", "action_id": action_id}
        item = dict(item)
        item["status"] = decision
        item["by"] = by
        item["decided_at"] = datetime.now(timezone.utc).isoformat()
        self._history.append(item)
        self._log("act", f"Supervisor '{by}' {decision} {action_id}: {item['title']}.")
        return {"action_id": action_id, "decision": decision, "by": by}

    def approve(self, action_id, by="supervisor"):
        return self._decide(action_id, "approved", by)

    def reject(self, action_id, by="supervisor"):
        return self._decide(action_id, "rejected", by)


def _stub_aws_clients():
    return {"s3": None, "dynamodb": None, "sns": None,
            "rekognition": None, "cloudwatch": None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_local_client(c):
    if c is None:
        return True
    if getattr(c, "mode", None) == "local":
        return True
    if getattr(c, "local", False):
        return True
    name = type(c).__name__.lower()
    return any(k in name for k in ("local", "stub", "fallback", "mock", "fake", "simulated"))


def _aws_mode(clients):
    try:
        vals = list(clients.values())
    except Exception:
        return "local"
    return "local" if vals and all(_is_local_client(v) for v in vals) else "aws"


def _jsonable(obj):
    """Coerce anything into JSON-serializable data (never raises)."""
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return {"error": "unserializable payload"}


def _engine_info(pipeline):
    """Describe the detection engine; works with the real pipeline
    (module-level ``engine_info()``), an instance attribute, or the stub."""
    info = getattr(pipeline, "engine_info", None)
    if callable(info):
        try:
            info = info()
        except Exception:
            info = None
    if not isinstance(info, dict):
        try:
            from pipeline import engine_info as _engine_info_fn  # type: ignore
            info = _engine_info_fn()
        except Exception:
            info = None
    if isinstance(info, dict) and info:
        d = dict(_jsonable(info))
        dnn = d.get("dnn_engine")
        d.setdefault("name", f"opencv-{dnn}" if dnn else "opencv")
        d.setdefault("mode", "live")
        d.setdefault("description", "OpenCV DNN detection pipeline.")
        return d
    name = getattr(pipeline, "engine_name", None) or "stub-local"
    return {
        "name": name,
        "mode": "stub" if str(name).startswith("stub") else "live",
        "detectors": list(getattr(pipeline, "detectors", None) or []),
        "description": "Local stub engine - no detectors loaded." if str(name).startswith("stub")
                       else "Detection engine metadata.",
    }


def _jpg_b64(img, quality=80, max_w=960):
    try:
        h, w = img.shape[:2]
        if w > max_w:
            img = cv2.resize(img, (max_w, int(h * max_w / w)))
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return ""


_SEV_COLORS = {
    "critical": (0, 0, 255),
    "high": (0, 140, 255),
    "medium": (0, 200, 200),
    "low": (0, 180, 0),
}


def _annotate(frame, events, blur_faces=False):
    """Draw event boxes (and optional face blur) onto a copy of the frame."""
    img = frame.copy()
    try:
        for ev in events or []:
            if not isinstance(ev, dict):
                continue
            bbox = ev.get("bbox")
            label = str(ev.get("type", "event"))
            sev = str(ev.get("severity", "low")).lower()
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (int(v) for v in bbox[:4])
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
            if blur_faces and "face" in label.lower():
                roi = img[y1:y2, x1:x2]
                if roi.size:
                    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (31, 31), 0)
                continue
            color = _SEV_COLORS.get(sev, (0, 180, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} [{sev}]", (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    except Exception:
        pass
    return img


def _extract_video_frames(data, ext, max_frames=12):
    suffix = ext if ext in VIDEO_EXTS else ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 1:
            n = min(max_frames, total)
            idxs = sorted({int(round(i * (total - 1) / (n - 1))) for i in range(n)} if n > 1 else {0})
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, fr = cap.read()
                if ok and fr is not None:
                    frames.append(fr)
        else:
            # Unknown length: read sequentially.
            while len(frames) < max_frames:
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                frames.append(fr)
        cap.release()
        return frames
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _safe_trace(loop):
    try:
        t = loop.get_trace()
        return list(t) if isinstance(t, (list, tuple)) else []
    except Exception:
        return []


def _safe_pending(loop):
    try:
        p = loop.get_pending_approvals()
        return list(p) if isinstance(p, (list, tuple)) else []
    except Exception:
        return []


def _safe_history(loop):
    try:
        h = loop.get_history()
        if isinstance(h, (list, tuple)) and h:
            return list(h)
    except Exception:
        pass
    # Real loop: decided approvals live in the event store.
    try:
        store = getattr(loop, "event_store", None)
        if store is not None and hasattr(store, "query_events"):
            items = store.query_events({"record_kind": "approval"})
            return [i for i in items
                    if str(i.get("approval_status", "")).lower() in ("approved", "rejected")]
    except Exception:
        pass
    return []


def _norm_trace_entry(e):
    """Normalize a trace entry to {seq, step, text, t}.

    Real loop: {"ts","stage","detail"}. Stub: {"seq","step","text","t"}.
    """
    if not isinstance(e, dict):
        return {"step": "act", "text": str(e), "t": ""}
    step = str(e.get("step") or e.get("stage") or "act").lower()
    if step not in ("perceive", "plan", "act"):
        step = "act"
    out = {
        "step": step,
        "text": str(e.get("text") or e.get("detail") or e.get("message") or ""),
        "t": str(e.get("t") or e.get("ts") or e.get("time") or ""),
    }
    if e.get("seq") is not None:
        out["seq"] = e.get("seq")
    return out


def _norm_approval(p):
    """Normalize an approval record to {action_id,title,detail,severity,
    status,created_at,by} for the supervisor UI.

    Real loop (event store): {"approval_id","approval_type",
    "approval_status","requested_at","decided_by","decision":{...},
    "event":{...},"event_type",...}.
    Stub: {"action_id","title","detail","severity","status",
    "created_at","by",...}.
    """
    if not isinstance(p, dict):
        return {"action_id": str(p), "title": str(p), "detail": "",
                "severity": "high", "status": "pending",
                "created_at": "", "by": None}
    decision = p.get("decision") if isinstance(p.get("decision"), dict) else {}
    event = p.get("event") if isinstance(p.get("event"), dict) else {}
    action_id = str(p.get("action_id") or p.get("approval_id") or "?")
    title = p.get("title")
    if not title:
        etype = p.get("event_type") or event.get("type") or "incident"
        atype = p.get("approval_type") or "approval"
        title = f"{atype}: {etype}"
        if decision.get("action"):
            title += f" -> {decision['action']}"
    detail = p.get("detail") or decision.get("message") or ""
    if not detail:
        pr = decision.get("policy_rule")
        if isinstance(pr, dict):
            detail = pr.get("rationale", "")
    if not detail and event:
        detail = f"event {event.get('type')} conf={event.get('conf', '?')}"
    severity = str(p.get("severity") or decision.get("severity") or "high").lower()
    return {
        "action_id": action_id,
        "title": str(title),
        "detail": str(detail or ""),
        "severity": severity,
        "status": str(p.get("status") or p.get("approval_status") or "pending"),
        "created_at": str(p.get("created_at") or p.get("requested_at") or ""),
        "decided_at": str(p.get("decided_at") or ""),
        "by": p.get("by") or p.get("decided_by"),
        "raw": p,
    }


def _annotate_frame_best(frame, result, pipeline, blur):
    """Annotate with the real pipeline annotator when available.

    Falls back to the local _annotate() when running against stubs.
    The real annotator draws per-track PPE badges, face blur (default ON),
    zone overlays and per-frame latency.
    """
    if _real_annotate is not None:
        try:
            tracks = result.get("tracks") or []
            ppe_results = {
                t.get("id"): (t.get("ppe") or {})
                for t in tracks if isinstance(t, dict)
            }
            return _real_annotate(
                frame, tracks, ppe_results, result.get("events") or [],
                result.get("metrics") or {}, blur_faces=bool(blur),
                zones=getattr(pipeline, "zones", None),
            )
        except Exception:
            pass
    return _annotate(frame, result.get("events") or [], blur_faces=blur)


def _process_frames(st, frames):
    """Run pipeline + agent loop over frames; accumulate global state."""
    pipeline = st["pipeline"]
    loop = st["loop"]
    all_events, decisions, per_frame = [], [], []
    annotated_b64, thumb_b64 = "", ""
    blur = bool(getattr(pipeline, "blur_faces", False))

    for fi, frame in enumerate(frames):
        if frame is None:
            continue
        t0 = time.perf_counter()
        try:
            result = pipeline.process(frame)
        except Exception:
            result = None
        if not isinstance(result, dict):
            result = {"tracks": [], "events": [], "metrics": {}, "engine": {}}
        wall_ms = (time.perf_counter() - t0) * 1000.0
        metrics = dict(result.get("metrics") or {})
        # Real pipeline reports {"total_ms", ...}; stub reports per_frame_ms.
        ms = metrics.get("per_frame_ms") or metrics.get("total_ms") or wall_ms
        try:
            ms = float(ms)
        except (TypeError, ValueError):
            ms = wall_ms
        metrics["per_frame_ms"] = round(ms, 2)
        result["metrics"] = metrics
        per_frame.append(metrics["per_frame_ms"])

        # Enrich the frame result for the agent loop (real loop expects
        # frame_id / jpeg_bytes / pipeline_ppe; harmless for the stub).
        try:
            ok, _buf = cv2.imencode(".jpg", frame,
                                     [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            jpeg_bytes = _buf.tobytes() if ok else None
        except Exception:
            jpeg_bytes = None
        result.setdefault("frame_id", f"upload-{fi}")
        result.setdefault("jpeg_bytes", jpeg_bytes)
        result.setdefault("pipeline_ppe", {})

        try:
            loop_out = loop.process(result)
        except Exception:
            loop_out = None
        loop_decisions = []
        if isinstance(loop_out, dict):
            for d in loop_out.get("decisions", []) or []:
                if isinstance(d, dict):
                    decisions.append(d)
                    loop_decisions.append(d)
            # Rekognition second-opinion verdicts surface via actions_taken.
            for a in loop_out.get("actions_taken", []) or []:
                if isinstance(a, dict) and a.get("kind") == "second_opinion":
                    verdict = a.get("verdict") or {}
                    agree = verdict.get("agreement")
                    if agree is True:
                        st["agreement"]["agreed"] += 1
                        st["agreement"]["total"] += 1
                    elif agree is False:
                        st["agreement"]["disagreed"] += 1
                        st["agreement"]["total"] += 1

        events = result.get("events") or []
        # When the loop returns one decision per event (real loop), borrow
        # the policy severity / message for display.
        aligned = (len(loop_decisions) == len(events) and len(events) > 0)
        if fi == 0:
            ann = _annotate_frame_best(frame, result, pipeline, blur)
            annotated_b64 = _jpg_b64(ann)
            thumb_b64 = _jpg_b64(ann, quality=60, max_w=192)
        for ei, ev in enumerate(events):
            if not isinstance(ev, dict):
                continue
            ev = dict(ev)
            ev.setdefault("type", "event")
            ev.setdefault("severity", "low")
            if aligned:
                dec = loop_decisions[ei]
                if dec.get("severity"):
                    ev["severity"] = dec["severity"]
                if dec.get("message"):
                    ev["message"] = dec["message"]
            if "message" not in ev:
                conf = ev.get("conf")
                ev["message"] = (f"{ev['type']} (conf {conf:.2f})"
                                 if isinstance(conf, (int, float))
                                 else str(ev["type"]))
            ev["frame_idx"] = fi
            ev["id"] = ev.get("id") or f"ev-{uuid.uuid4().hex[:8]}"
            ev["t"] = ev.get("t") or ev.get("ts") or datetime.now(timezone.utc).isoformat()
            if fi == 0 and thumb_b64:
                ev["evidence"] = thumb_b64
            all_events.append(ev)
            st["events"].append(ev)
            ra = ev.get("rekognition_agrees")
            if ra is True:
                st["agreement"]["agreed"] += 1
            elif ra is False:
                st["agreement"]["disagreed"] += 1
            if ra is not None:
                st["agreement"]["total"] += 1

    st["events"] = st["events"][-300:]
    st["latency"].extend(per_frame)
    st["latency"] = st["latency"][-500:]
    st["frames_processed"] += len(per_frame)

    avg = round(sum(per_frame) / len(per_frame), 2) if per_frame else 0.0
    return {
        "annotated_image": annotated_b64,
        "events": all_events,
        "decisions": decisions,
        "trace_tail": [_norm_trace_entry(t) for t in _safe_trace(loop)[-8:]],
        "metrics": {"per_frame_ms": avg, "frames": len(per_frame)},
        "per_frame_ms": avg,
        "frames_processed": len(per_frame),
    }


def _summary(st):
    by_type, by_sev = {}, {}
    for ev in st["events"]:
        t = str(ev.get("type", "event"))
        s = str(ev.get("severity", "low")).lower()
        by_type[t] = by_type.get(t, 0) + 1
        by_sev[s] = by_sev.get(s, 0) + 1
    lat = st["latency"]
    return {
        "total_incidents": len(st["events"]),
        "by_type": by_type,
        "by_severity": by_sev,
        "avg_latency_ms": round(sum(lat) / len(lat), 2) if lat else 0.0,
        "frames_processed": st["frames_processed"],
    }


def _agreement_stats(st):
    a = st["agreement"]
    total = a["total"]
    return {
        "agreed": a["agreed"],
        "disagreed": a["disagreed"],
        "total": total,
        "agreement_rate": round(a["agreed"] / total, 3) if total else None,
        "note": ("Rekognition second opinion unavailable - local AWS fallback mode; "
                 "counts accumulate when the pipeline reports 'rekognition_agrees'.")
                if st["aws_mode"] == "local" else
                "Agreement between the local detector and the Rekognition second opinion.",
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024

    # ---- singletons (all defensive) -------------------------------------
    if SafetyPipeline is not None:
        try:
            pipeline = SafetyPipeline()
            real_pipeline = True
        except Exception:
            pipeline = _StubPipeline()
            real_pipeline = False
    else:
        pipeline = _StubPipeline()
        real_pipeline = False

    if make_clients is not None:
        try:
            aws_clients = make_clients()
            if not isinstance(aws_clients, dict):
                aws_clients = _stub_aws_clients()
        except Exception:
            aws_clients = _stub_aws_clients()
    else:
        aws_clients = _stub_aws_clients()
    aws_mode = _aws_mode(aws_clients)

    real_executor = False
    executor = None
    if ActionExecutor is not None:
        c = aws_clients if isinstance(aws_clients, dict) else {}
        attempts = [
            # Real signature: ActionExecutor(s3, dynamodb, sns, rekognition,
            # cloudwatch, ...).
            lambda: ActionExecutor(s3=c.get("s3"), dynamodb=c.get("dynamodb"),
                                   sns=c.get("sns"),
                                   rekognition=c.get("rekognition"),
                                   cloudwatch=c.get("cloudwatch")),
            lambda: ActionExecutor(c.get("s3"), c.get("dynamodb"), c.get("sns"),
                                   c.get("rekognition"), c.get("cloudwatch")),
            lambda: ActionExecutor(aws_clients=aws_clients),
            lambda: ActionExecutor(),
        ]
        for attempt in attempts:
            try:
                executor = attempt()
            except Exception:
                continue
            else:
                real_executor = True
                break
    if executor is None:
        executor = _StubActionExecutor(aws_clients=aws_clients)

    real_agent = False
    loop = None
    if AgentLoop is not None and real_executor:
        dd = aws_clients.get("dynamodb") if isinstance(aws_clients, dict) else None
        attempts = [
            # Real signature: AgentLoop(actions, event_store, site_id=...).
            lambda: AgentLoop(actions=executor, event_store=dd),
            lambda: AgentLoop(executor, dd),
            lambda: AgentLoop(executor=executor, aws_clients=aws_clients),
            lambda: AgentLoop(),
        ]
        for attempt in attempts:
            try:
                loop = attempt()
            except Exception:
                continue
            else:
                real_agent = True
                break
    if loop is None:
        loop = _StubAgentLoop(executor=executor, aws_clients=aws_clients)
        real_agent = False

    st = {
        "pipeline": pipeline,
        "aws_clients": aws_clients,
        "executor": executor,
        "loop": loop,
        "aws_mode": aws_mode,
        "real_pipeline": real_pipeline,
        "real_agent": real_agent,
        "events": [],          # accumulated event records (with evidence thumbnails)
        "latency": [],         # per-frame ms history
        "frames_processed": 0,
        "agreement": {"agreed": 0, "disagreed": 0, "total": 0},
        "config": {"blur_faces": bool(getattr(pipeline, "blur_faces", False))},
    }
    app.config["SITESENTRY"] = st

    engine = _engine_info(pipeline)

    # ---- pages -----------------------------------------------------------
    @app.get("/")
    def index():
        samples = [n for n in (1, 2, 3)
                   if os.path.exists(os.path.join(WORK_DIR, "assets", f"sample{n}.jpg"))]
        return render_template(
            "index.html",
            aws_mode=st["aws_mode"],
            demo_badge=(st["aws_mode"] == "local"),
            engine=engine,
            blur_faces=st["config"]["blur_faces"],
            samples=samples,
        )

    @app.get("/supervisor")
    def supervisor():
        return render_template(
            "supervisor.html",
            aws_mode=st["aws_mode"],
            demo_badge=(st["aws_mode"] == "local"),
            engine=engine,
        )

    # ---- upload ----------------------------------------------------------
    @app.post("/api/upload")
    def api_upload():
        try:
            f = request.files.get("file")
            if f is None or not f.filename:
                return jsonify({"error": "no file provided (multipart field 'file')"}), 400
            data = f.read()
            if not data:
                return jsonify({"error": "empty file"}), 400
            ext = os.path.splitext(f.filename)[1].lower()
            is_video = ext in VIDEO_EXTS or (f.mimetype or "").startswith("video")
            if is_video:
                frames = _extract_video_frames(data, ext, max_frames=12)
            else:
                arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    return jsonify({"error": "could not decode image"}), 400
                frames = [frame]
            if not frames:
                return jsonify({"error": "no frames extracted from upload"}), 400
            payload = _process_frames(st, frames)
            payload["source"] = "video" if is_video else "image"
            payload["filename"] = f.filename
            return jsonify(_jsonable(payload))
        except Exception as exc:
            return jsonify({"error": "upload failed", "detail": str(exc)[:200]}), 500

    @app.get("/api/sample/<int:n>")
    def api_sample(n):
        if n not in (1, 2, 3):
            return jsonify({"error": "sample not found (n must be 1..3)"}), 404
        path = os.path.join(WORK_DIR, "assets", f"sample{n}.jpg")
        if not os.path.exists(path):
            return jsonify({"error": f"assets/sample{n}.jpg not found"}), 404
        return send_file(path, mimetype="image/jpeg")

    # ---- live stream (graceful when no camera) ---------------------------
    @app.get("/api/stream")
    def api_stream():
        if not glob.glob("/dev/video*"):
            return jsonify({"error": "no camera",
                            "hint": "use upload or samples"}), 503
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            return jsonify({"error": "no camera",
                            "hint": "use upload or samples"}), 503

        def gen():
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    ok2, buf = cv2.imencode(".jpg", frame)
                    if not ok2:
                        continue
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buf.tobytes() + b"\r\n")
            finally:
                cap.release()

        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    # ---- agent / supervisor APIs -----------------------------------------
    @app.get("/api/events")
    def api_events():
        evs = list(reversed(st["events"][-50:]))
        return jsonify(_jsonable({"events": evs, "count": len(evs)}))

    @app.get("/api/trace")
    def api_trace():
        trace = [_norm_trace_entry(t) for t in _safe_trace(st["loop"])[-50:]]
        return jsonify(_jsonable({"trace": trace, "count": len(trace)}))

    @app.get("/api/pending")
    def api_pending():
        pending = [_norm_approval(p) for p in _safe_pending(st["loop"])]
        return jsonify(_jsonable({"pending": pending, "count": len(pending)}))

    @app.get("/api/history")
    def api_history():
        history = [_norm_approval(p)
                   for p in reversed(_safe_history(st["loop"]))]
        return jsonify(_jsonable({"history": history, "count": len(history)}))

    def _decide(action_id, decision):
        # Real loop: approve(action_id, approved_by) / reject(action_id,
        # approved_by, note=""), returning the updated record or None.
        # Stub: approve(action_id, by=...). Try each signature defensively.
        body = request.get_json(silent=True) or {}
        by = body.get("by") or "supervisor"
        note = body.get("note") or ""
        loop = st["loop"]
        fn = loop.approve if decision == "approved" else loop.reject
        if decision == "approved":
            attempts = [lambda: fn(action_id, by=by),
                        lambda: fn(action_id, approved_by=by),
                        lambda: fn(action_id, by)]
        else:
            attempts = [lambda: fn(action_id, by=by),
                        lambda: fn(action_id, approved_by=by, note=note),
                        lambda: fn(action_id, by)]
        out = None
        sig_error = None
        for attempt in attempts:
            try:
                out = attempt()
            except TypeError as exc:
                sig_error = exc
                continue
            except Exception as exc:
                return jsonify({"error": "decision failed",
                                "detail": str(exc)[:200]}), 500
            else:
                sig_error = None
                break
        if sig_error is not None:
            return jsonify({"error": "decision failed",
                            "detail": f"incompatible approve/reject signature: {sig_error}"[:200]}), 500
        if out is None or (isinstance(out, dict)
                            and out.get("error") == "unknown action_id"):
            return jsonify({"error": "unknown action_id",
                            "action_id": action_id}), 404
        if not isinstance(out, dict):
            out = {"action_id": action_id}
        norm = _norm_approval(out)
        return jsonify(_jsonable({"action_id": action_id,
                                  "decision": norm.get("status") or decision,
                                  "by": by,
                                  "record": norm}))

    @app.post("/api/approve/<action_id>")
    def api_approve(action_id):
        return _decide(action_id, "approved")

    @app.post("/api/reject/<action_id>")
    def api_reject(action_id):
        return _decide(action_id, "rejected")

    @app.get("/api/summary")
    def api_summary():
        return jsonify(_jsonable(_summary(st)))

    @app.get("/api/agreement")
    def api_agreement():
        return jsonify(_jsonable(_agreement_stats(st)))

    @app.get("/api/health")
    def api_health():
        status = "ok" if st["real_pipeline"] else "degraded"
        return jsonify({
            "status": status,
            "aws_mode": st["aws_mode"],
            "engine": engine.get("name"),
            "pipeline": "real" if st["real_pipeline"] else "stub",
            "agent": "real" if st["real_agent"] else "stub",
            "time": datetime.now(timezone.utc).isoformat(),
        })

    @app.get("/api/engine")
    def api_engine():
        return jsonify(_jsonable(engine))

    @app.post("/api/config")
    def api_config():
        try:
            body = request.get_json(silent=True) or {}
            if "blur_faces" in body:
                val = bool(body["blur_faces"])
                try:
                    st["pipeline"].blur_faces = val
                except Exception:
                    pass
                st["config"]["blur_faces"] = val
            return jsonify({"ok": True, "config": dict(st["config"])})
        except Exception as exc:
            return jsonify({"error": "config update failed",
                            "detail": str(exc)[:200]}), 500

    # ---- error handlers (JSON for API, never a stack trace) ---------------
    @app.errorhandler(404)
    def _not_found(exc):
        if request.path.startswith("/api/"):
            return jsonify({"error": "not found"}), 404
        return render_template("index.html", aws_mode=st["aws_mode"],
                               demo_badge=(st["aws_mode"] == "local"),
                               engine=engine,
                               blur_faces=st["config"]["blur_faces"],
                               samples=[]), 404

    @app.errorhandler(500)
    def _internal(exc):
        if request.path.startswith("/api/"):
            return jsonify({"error": "internal error"}), 500
        return "Internal error", 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
