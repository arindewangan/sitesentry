# SiteSentry — Technical Report
**Agentic Vision Safety Inspector for Construction Sites**
OpenCV AI Competition 2026 · Solo entry

---

## 1. Problem

Construction remains one of the deadliest civilian occupations. In India,
construction-site incidents kill and injure thousands of workers every year,
and safety audits consistently identify **PPE non-compliance — missing helmets
and high-visibility vests — as the leading preventable factor**. The current
practice on most sites is a safety officer doing periodic walkthroughs with a
clipboard: coverage is sparse, violations between walkthroughs go unseen, and
there is no timestamped evidence trail when an incident is investigated.

SiteSentry closes that gap: a camera feed becomes a tireless safety inspector
that watches every frame, flags PPE violations, zone intrusions and suspected
falls, and — crucially — acts on what it sees through an auditable agent loop.

## 2. Users

- **Primary: site safety officers** at infrastructure contractors (metro, road,
  high-rise builders). They get a live dashboard, an evidence-backed incident
  log, and one-click escalation instead of clipboards.
- **Secondary: project managers / site supervisors**, who receive SNS email
  alerts on high-severity events and approve or reject consequential actions
  (e.g. a work-stoppage recommendation) from the supervisor view.
- **Beneficiaries: the workers themselves** — earlier detection of missing PPE
  and falls means faster intervention. The system is assistive by design: it
  surfaces likely violations *for human review*, never for automated
  disciplinary action.

## 3. Architecture

See `docs/architecture.svg` / `docs/architecture.png`.

```
Browser (judge's laptop: upload / webcam / dashboard)
        │ HTTPS multipart + MJPEG
        ▼
EC2 t4g.micro (Graviton2 Arm, Ubuntu 24.04, ap-south-1) — Flask + gunicorn + systemd
  ┌ OpenCV 5 vision pipeline (cv2 5.0.0, new DNN engine)
  │   decode/resize → person detect (YOLOv8n ONNX / PPE-model Person class)
  │   → Kalman multi-object tracking → PPE analysis per track (SafetyVision
  │   YOLOv8s: Hardhat / NO-Hardhat / Safety Vest / NO-Safety Vest / Fall-Detected)
  │   → restricted-zone polygons (BackgroundSubtractorMOG2)
  │   → annotate + face blur → per-stage latency metrics
  └ Agentic loop: perceive → plan → act
        │ boto3 (local-fallback when credentials absent)
        ▼
  S3 (evidence archive, 30-day lifecycle) · DynamoDB (incident event log)
  SNS (supervisor email alerts) · Rekognition DetectProtectiveEquipment
  (independent second opinion + agreement matrix) · CloudWatch (latency metrics)
```

**Agentic qualification.** Image results directly change system behavior: a
`NO_HELMET` detection on a repeat offender escalates the action from
log-and-archive to an SNS alert, and a third offense (or a suspected fall)
creates a *work-stoppage recommendation that requires supervisor approval*
before any escalation is sent. This is the competition's listed agentic
example — "safety monitoring, and human-in-the-loop operations" — implemented
literally: vision evidence → policy decision → action or approval request.

## 4. OpenCV 5 implementation

OpenCV is the core functional component, not a wrapper. Specific OpenCV 5
features exercised:

1. **New DNN engine** — both models (YOLOv8n person detector, SafetyVision
   YOLOv8s PPE detector) run natively in `cv2.dnn` with **no ONNX Runtime
   dependency**. The engine is selected at runtime: `ENGINE_OPENCV` when
   present (later 5.x), `ENGINE_NEW` on 5.0.0, with the choice logged at
   startup (`pipeline/engine_info()` and `/api/engine`). The environment used
   for this build is opencv-python **5.0.0**, which exposes `ENGINE_NEW`.
2. **FP16 blob path** — `cv2.dnn.blobFromImage(..., ddepth=cv2.CV_16F)` is
   attempted first with automatic fallback to FP32; the active precision is
   reported per detector.
3. **KalmanFilter** multi-object tracker with IoU association; per-track
   history drives the fall heuristic (aspect-ratio flip + vertical-velocity
   spike over a 30-frame window).
4. **BackgroundSubtractorMOG2** for restricted-zone motion, combined with
   `cv2.pointPolygonTest` polygon intrusion checks.
5. **HarfBuzz text stack** — `cv2.putText` annotations (per-frame latency,
   track labels) render through OpenCV 5's HarfBuzz backend.
6. **Haar cascade face blurring** on by default in the annotation stage; the
   whole head region is blurred when the cascade file is unavailable.

**Model note (honest).** The canonical Ultralytics release no longer ships a
`yolov8n.onnx` asset and the Hugging Face mirror returned 401 from the build
network, so person detection runs on the PPE model's own `Person` class
(class 11) — one forward pass yields persons, PPE items, violations and
falls. `pipeline/detector.py` still supports a drop-in `models/yolov8n.onnx`
COCO person detector if provided. The PPE model is SafetyVision YOLOv8s
(`v2/best_640.onnx`, 13 classes), whose model card reports held-out test
mAP@0.5 = 0.766 @896 / 0.754 @640 and ONNX@640 = 0.738. Its YOLOv8 lineage is
AGPL-3.0 — noted in `scripts/download_models.sh` and the LICENSE file.

## 5. AWS deployment

Meaningful, multi-service, Free-Tier-sized (region **ap-south-1**, Mumbai):

| Service | Role |
|---|---|
| EC2 t4g.micro (Graviton2 Arm) | Hosts Flask + OpenCV 5 pipeline (systemd + gunicorn, `deploy/sitesentry.service`) |
| S3 | `raw/` uploads, `annotated/` outputs, `evidence/` incident frames; 30-day lifecycle expiry |
| DynamoDB | `sitesentry-events` incident log (site_id + timestamp keys, severity GSI); approvals write back here |
| SNS | Email alerts to the supervisor on high/critical severity |
| Rekognition | `DetectProtectiveEquipment` as an **independent second opinion** on sampled flagged frames; agreement matrix vs the OpenCV 5 pipeline is shown in the UI and reported in §7 |
| CloudWatch | `PipelineLatencyMs` custom metric + structured logs; alarms on 5xx |
| IAM | Least-privilege instance profile; **no hardcoded credentials anywhere** (`.env`, never committed) |

**Reproducibility:** `scripts/deploy.sh` recreates S3/DynamoDB/SNS/EC2
idempotently; `requirements.txt` pins every dependency; `scripts/smoke_test.sh`
validates upload → events → event-store row end to end.

**Zero-spend build mode.** No AWS spend was authorized for this build, so every
boto3 client implements **automatic local-fallback**: with no credentials it
writes to `data/local/<service>/` (JSONL/file stores) and labels every output
`local`. The dashboard, agent loop, trace, approvals and gallery run
identically. The public demo surface is a clearly-labeled SIMULATED static
showcase (`demo/index.html`); the rules explicitly permit an arranged live
screen-share of the real Flask app as the working demonstration.

## 6. Agent design (perceive → plan → act)

- **Perceive** (`agent/tools.perceive_frame`): the OpenCV 5 pipeline returns
  tracks, PPE states, zone/fall events and per-stage latency.
- **Plan** (`agent/policy.py`): an auditable policy table maps
  (event type × offense count) → action + severity + whether human approval is
  required. Repeat-offender counting is per site/track/event-type.
- **Act** (`agent/actions.py`): DynamoDB log, S3 evidence archive, SNS alert,
  Rekognition second-opinion sampling, or a supervisor approval request.
- **Trace** (`agent/loop.py`): every cycle appends timestamped
  perceive/plan/act entries; the dashboard's trace panel shows vision evidence
  → policy decision → action, which is exactly the agentic bar ("visual
  evidence must change what the system does next").
- **Human control:** work-stoppage recommendations and fall escalations sit in
  the supervisor queue until approved/rejected in one click; the decision is
  written back to DynamoDB. An optional Bedrock incident summary degrades
  gracefully to a rule-based summary.

## 7. Evaluation

Methodology and full numbers: `docs/evaluation.md`. Headlines:

- **Model (published, held-out):** SafetyVision YOLOv8s test mAP@0.5 0.766
  (896) / 0.754 (640); per-class: Hardhat 0.937, Safety Vest 0.892,
  NO-Hardhat 0.754, Person 0.861, **NO-Safety Vest 0.386 (weakest)**,
  Fall-Detected 0.959.
- **Pipeline (this build, synthetic + sample images):** per-stage latency
  measured via `scripts/benchmark_cool.py` (means + stdev, CPU/engines
  recorded); end-to-end per-frame ms displayed live in the UI.
- **Rekognition agreement:** sampled flagged frames run through
  `DetectProtectiveEquipment`; the agreement matrix (OpenCV 5 vs Rekognition
  on HEAD_COVER) is shown in the dashboard and recorded in evaluation.md —
  disagreements are treated as first-class evidence, not hidden.
- **Failure cases (required, documented honestly):** see `docs/evaluation.md`
  § Failure cases — occluded workers, low light/glare, small/distant figures,
  NO-Safety Vest false negatives, motion blur in video, and the conservative
  "flag missing unless positive evidence" default that trades false alarms for
  fewer misses.

## 8. Limitations

1. Detection quality is bounded by the PPE model: NO-Safety Vest recall is
   poor (0.431); night/low-light and heavy occlusion degrade everything.
2. The demo runs on CPU; YOLOv8s@640 is ~0.5–0.8 s/frame on Lambda-class CPUs —
   fine for inspection cadence, not 30 fps video.
3. Fall detection is a 2D heuristic (aspect ratio + motion), not pose
   estimation — it suggests, never concludes; every fall event requires human
   approval before escalation.
4. Sample images bundled with the demo are AI-generated synthetic scenes
   (labeled as such), not real sites.
5. Local-fallback mode is not AWS: S3/DynamoDB/SNS/Rekognition/CloudWatch
   integrations are fully wired in code but exercised against local stores
   until credentials are provided.

## 9. Responsible use

Full statement: `docs/responsible_use.md`. In brief: **faces blurred by
default** in all annotated output; **no face recognition or identity tracking**
anywhere (tracks are anonymous numeric IDs); **human-in-the-loop** approval
for any consequential action; **30-day evidence retention** via S3 lifecycle;
**conservative thresholds** documented with their false-alarm cost; the system
is positioned as an *assistive pre-screening tool for qualified safety
personnel*, never for automated disciplinary action — the same boundary the
model card itself draws.

## 10. Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash scripts/download_models.sh   # ~55 MB, cited public sources
.venv/bin/python -m pytest tests/ -q
.venv/bin/python app.py            # http://127.0.0.1:5000
```

---

*Built solo for the OpenCV AI Competition 2026. OpenCV 5 + AWS + agentic vision,
aimed at the Agentic Vision special award and the overall rubric.*
