# SiteSentry — Agentic Vision Safety Inspector for Construction Sites

OpenCV AI Competition 2026 · solo entry · MIT

An agentic computer-vision safety inspector: **OpenCV 5** watches
construction-site footage for PPE violations, zone intrusions and falls, and an
**agent loop** turns the visual evidence into decisions — log, archive, alert,
or escalate for **human approval** — with a meaningful **AWS** footprint
(EC2 Graviton + S3 + DynamoDB + SNS + Rekognition + CloudWatch).

- 🎬 **Demo (simulated showcase):** open `demo/index.html` — clearly labeled,
  no backend needed. Judges: a live screen-share of the real Flask app can be
  arranged (explicitly allowed by the rules).
- 📄 **Technical report:** `TECHNICAL_REPORT.md` · **Submission copy:** `SUBMISSION.md`
- 🏗 **Architecture:** `docs/architecture.svg` / `docs/architecture.png`

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash scripts/download_models.sh   # YOLOv8n (optional) + SafetyVision YOLOv8s PPE ONNX, ~55 MB
.venv/bin/python -m pytest tests/ -q
.venv/bin/python app.py            # http://127.0.0.1:5000
```

Upload a site photo (or try the bundled synthetic samples at
`/api/sample/1..3`), watch the annotated feed, the live **agent trace**
(vision evidence → policy decision → action), and the supervisor approval
queue at `/supervisor`.

## Layout

```
app.py                  Flask entry (dashboard, API, MJPEG)
pipeline/               OpenCV 5: detector, tracker (Kalman), ppe, zones (MOG2),
                        annotate (+face blur), metrics
agent/                  loop, policy (auditable table), tools (MCP-style),
                        actions, summarize (Bedrock optional)
aws/                    boto3 clients: s3, dynamodb, sns, rekognition, cloudwatch
                        — every client auto-falls-back to data/local/ with NO creds
templates/ static/      dashboard + supervisor UI
scripts/                download_models.sh · deploy.sh (EC2 t4g.micro, ap-south-1)
                        smoke_test.sh · benchmark_cool.py
tests/                  pytest, fully offline (mocked CV + mocked AWS)
docs/                   architecture, evaluation (incl. failure cases),
                        responsible_use, demo_script, cool_benchmark, showcase.html
demo/index.html         simulated GitHub-Pages showcase (labeled as such)
assets/                 logo, banner, 3 AI-generated sample images (+SOURCES.md)
```

## AWS in 30 seconds

No AWS spend is needed to run everything: set nothing and all clients use
local fallback stores. To go live, copy `.env.example` → `.env`, fill in the
bucket/table/topic, run `scripts/deploy.sh --key-name … --supervisor-email …`,
and set `SITENTRY_AWS=on`. The Rekognition second opinion and Bedrock summary
are individually gated (`SITENTRY_REKOGNITION`, `SITENTRY_BEDROCK`).

## Responsible use (built in, not bolted on)

Faces blurred by default · no face recognition anywhere · human approval for
consequential actions · 30-day evidence retention · conservative thresholds,
documented. See `docs/responsible_use.md`.

## Model provenance

- Person detection: YOLOv8n COCO ONNX (optional drop-in at `models/yolov8n.onnx`)
- PPE: SafetyVision YOLOv8s, 13 classes
  (https://huggingface.co/ayushgupta7777/safetyvision-yolov8, `v2/best_640.onnx`;
  held-out test mAP@0.5 = 0.766). YOLOv8 lineage is **AGPL-3.0** — see
  `scripts/download_models.sh` and `LICENSE` before commercial use.
