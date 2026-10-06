# SiteSentry — Demo Script (≤5 min video)

Total budget: 5:00. Solo presenter; keep energy high, no dead air.

## Shot list

| Time | Shot | Narration (tight) |
|---|---|---|
| 0:00–0:20 | Presenter to camera, banner backdrop | "I'm [name], solo builder of SiteSentry — an agentic vision safety inspector for construction sites, built on OpenCV 5 and AWS for the OpenCV AI Competition." |
| 0:20–0:50 | Problem slides: site photos, stat cards | "Thousands of construction workers are hurt yearly where the first failure was simple: no helmet, no vest. Safety officers can't watch every corner. SiteSentry is a camera feed that never blinks — and acts on what it sees." |
| 0:50–1:00 | Architecture diagram (docs/architecture.png) | "OpenCV 5 does the vision, an agent loop turns evidence into decisions, AWS does the cloud. Let me show you." |
| 1:00–2:30 | **LIVE: upload sample2** (bare-headed worker) | Annotated feed appears: boxes, helmet/vest badges, per-frame ms. "Watch the trace panel: perceived — one track, no helmet, confidence 0.91. Policy: second offense on this track. Decision: SNS alert to the supervisor plus evidence archived to S3. The vision result *changed* what the system did." |
| 2:30–3:15 | **LIVE: supervisor approval** | "For a third offense — or a suspected fall — the agent can't act alone. It recommends a work stoppage. One click to approve or reject; the decision writes back to DynamoDB. Human in the loop, always." |
| 3:15–3:50 | Dashboard: incident log, evidence gallery, Rekognition agreement card | "Every event is timestamped and queryable. Rekognition gives an independent second opinion — here's the agreement matrix against our OpenCV pipeline, disagreements included." |
| 3:50–4:20 | Failure cases (evaluation.md) + face blur demo | "It fails honestly: occluded workers, glare, distant figures — and the vest-absence class is weak, so we default conservative and keep humans in charge. Faces are blurred by default; nobody is identified." |
| 4:20–4:50 | Latency numbers + what's next | "Per-stage latencies are measured, not claimed. Next: the COOL-on-Graviton benchmark and a real pilot with a contractor's safety team." |
| 4:50–5:00 | Close to camera | "SiteSentry: OpenCV 5 sees, the agent decides, a human approves. Thank you." |

## B-roll sources

- `demo/index.html` (simulated showcase) for cutaways
- `docs/architecture.png` for the architecture beat
- Screen recordings of the real Flask app (upload → trace → approval) — the
  core of the video; record at 1080p, cursor visible

## Presenter notes

- Treat archived evidence as sensitive; don't linger on unblurred faces.
- State once, on camera: "Sample site images in the demo are AI-generated."
- The "team" shot requirement: the 0:00 and 4:50 to-camera segments satisfy it
  for a solo entry.
