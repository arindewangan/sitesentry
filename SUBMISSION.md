# SiteSentry — Devpost Submission Copy

## Inspiration

Every year, thousands of construction workers in India are injured or killed
in incidents where the first link in the chain was something simple: a missing
helmet, a missing hi-vis vest. Site safety officers do walkthroughs with
clipboards, but a walkthrough covers a site for minutes a day — violations in
between go unseen, and when something does go wrong there is rarely
timestamped evidence of what led up to it.

We wanted to give every site a tireless safety inspector: a camera feed that
never blinks, that doesn't just *see* violations but *does something*
auditable about them — log, archive, alert, or escalate to a human. And we
wanted to build it on the two things this competition is about: OpenCV 5 doing
the real vision work, and AWS doing the real cloud work.

## What it does

**SiteSentry is an agentic vision safety inspector for construction sites.**

Point it at a site photo, a video clip, or a webcam. OpenCV 5 detects every
person, tracks them across frames with a Kalman filter, and checks each
worker's helmet and vest using a fine-tuned PPE detection model — all running
natively in the new OpenCV 5 DNN engine, no ONNX Runtime needed. It also
watches restricted zones (polygon ROIs + background subtraction) and flags
suspected falls.

Then the agent loop takes over — this is the part we're proudest of. Vision
evidence flows into an auditable policy table that decides what happens next:

- **First helmet offense** → logged to DynamoDB, evidence frame archived to S3.
- **Repeat offender** → SNS email alert fires to the site supervisor.
- **Third offense, or a suspected fall** → the agent *recommends a work
  stoppage* — but it cannot act alone. The recommendation lands in a supervisor
  queue, and a human approves or rejects it in one click. The decision is
  written back to the event log.

Every step is visible in a live **agent trace panel**: perceived evidence →
policy decision → action. Nothing is a black box.

For trust, AWS Rekognition runs `DetectProtectiveEquipment` as an independent
second opinion on flagged frames, and the dashboard shows the agreement matrix
between our OpenCV 5 pipeline and Rekognition — disagreements included. Faces
are blurred by default, nobody is identified (tracks are anonymous numbers),
and evidence auto-expires after 30 days.

## How we built it

**Vision (OpenCV 5.0.0, the star of the show):**
- Both neural nets — a YOLOv8n person detector and the SafetyVision YOLOv8s
  PPE detector (13 classes: Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest,
  Fall-Detected, Person, …) — run natively in `cv2.dnn` on the **new DNN
  engine** (`ENGINE_NEW` on 5.0.0), with an FP16 blob path and automatic FP32
  fallback. The engine selection is detected at runtime and logged.
- Kalman-filter multi-object tracking, `BackgroundSubtractorMOG2` +
  `pointPolygonTest` zone intrusion, and a 30-frame aspect-ratio/motion
  heuristic for fall candidates.
- Annotation through OpenCV 5's HarfBuzz text stack, with Haar-cascade face
  blurring on by default.

**Agent:** a perceive → plan → act loop (`agent/`). The policy table is a
plain auditable data structure — severity, escalation, and approval
requirements per (event type × offense count). MCP-style tools
(`perceive_frame`, `query_events`, `archive_evidence`, `request_approval`)
keep the agent's interface clean.

**Cloud (AWS, ap-south-1 Mumbai):** Flask on EC2 t4g.micro (Graviton2),
S3 evidence archive with 30-day lifecycle, DynamoDB incident log, SNS email
alerts, Rekognition second opinion, CloudWatch latency metrics, least-privilege
IAM — no hardcoded credentials anywhere. One `scripts/deploy.sh` reproduces
the stack; `requirements.txt` is fully pinned.

**Honest engineering:** every AWS client auto-degrades to a local JSONL/file
store when credentials are absent, so the entire loop runs with zero spend —
and every local output is labeled as local. The public demo page is a
clearly-marked SIMULATED showcase; the rules allow an arranged live
screen-share of the real Flask app, which is how judges see the true system.

## Challenges we ran into

1. **OpenCV 5's DNN engine API moved between 5.0.0 and later 5.x**
   (`ENGINE_NEW` → `ENGINE_OPENCV`). We detect the available enum at runtime
   and log the choice — the code works on both.
2. **The YOLOv8n ONNX download we planned on vanished** — the Ultralytics
   release no longer ships the asset and the mirror 401'd. Instead of stalling,
   we folded person detection into the PPE model's own `Person` class: one
   forward pass now yields persons + PPE items + falls, halving inference cost
   on the tiny Graviton instance. The code still accepts a drop-in YOLOv8n
   model.
3. **PPE accuracy is uneven** — the model's own card admits NO-Safety Vest
   recall is weak (0.431). We set a conservative policy (flag missing unless
   there's positive evidence), documented the failure modes honestly, and used
   Rekognition as a second opinion rather than pretending the problem away.
4. **Surveillance ethics.** A safety camera is one step from a surveillance
   camera, so we built the safeguards as features: face blur default-on, no
   identity recognition, human approval for consequential actions, 30-day
   retention.

## Accomplishments that we're proud of

- A **genuinely agentic** loop where vision changes behavior — the competition
  literally lists "safety monitoring, and human-in-the-loop operations" as the
  example, and that's what we built.
- Real OpenCV 5 depth: new DNN engine, FP16 path, Kalman tracking, MOG2 zones,
  HarfBuzz annotation — measured per-stage latencies, not hand-waving.
- A second-opinion architecture (OpenCV 5 vs Rekognition agreement matrix)
  that turns model disagreement into documented evaluation evidence.
- Full reproducibility: pinned deps, one-command AWS deploy, smoke test,
  offline-capable test suite, local-fallback for every cloud call.

## What we learned

- **Application novelty beats algorithm novelty.** Nobody needed a new
  detector; construction sites needed a *decision loop* around the detector.
- **Failure cases are a feature.** The rubric asks for them, the model card
  documents them, and being honest about NO-Safety Vest weakness made the
  policy design (conservative thresholds + human approval) obviously correct.
- **OpenCV 5's DNN engine is ready for modern YOLO models** without ONNX
  Runtime — but you must handle the 5.0.0-vs-later API difference.
- **Design for zero spend first.** Local-fallback clients meant we could build
  and demo the entire cloud architecture without an AWS bill, then flip one
  env var to go live.

## What's next

- **COOL benchmark track:** deploy on the Cloud-Optimized OpenCV Library AMI
  (Graviton) and publish measured before/after latency via
  `scripts/benchmark_cool.py`.
- **Pose-based fall confirmation** to replace the 2D heuristic.
- **Multi-camera site view** with homography-based site-map overlay
  (`findHomography` USAC, OpenCV 5 default).
- **On-device edge variant** for sites with poor connectivity, syncing
  evidence to S3 when back online.
- Pilot with a real contractor's safety team — with their workers' informed
  consent and the responsible-use safeguards audited first.

## Links

- GitHub repo: https://github.com/arindewangan/sitesentry
- Live demo (simulated showcase): https://arindewangan.github.io/sitesentry/demo/
- Demo video: TBD
