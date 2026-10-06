# SiteSentry — Known Failure Cases

Honest notes on where the system underperforms today. Linked from the
dashboard ("Known failure cases" card).

## Detection

- **Heavy occlusion / night footage** — recall drops when workers or PPE are
  partially hidden or the scene is under-lit. The annotated output will show
  fewer boxes than a human reviewer would draw.
- **Small or distant PPE** — hard hats and vests beyond ~15 m from the camera
  are frequently missed at 640×480 input resolution.
- **Motion blur** — fast-moving equipment smears edges; tracks may fragment
  into short, disconnected segments.
- **Face blur trade-off** — enabling face blur can erase small faces entirely
  at low resolution, which also removes the head cue the detector uses.

## Video ingestion

- **Codecs** — H.264 MP4 is the tested path. Other codecs (HEVC, VP9 in odd
  containers) may extract fewer than the requested 12 evenly-spaced frames,
  or fail to decode on hosts without the right OpenCV backend.
- **Very long videos** — only up to 12 frames are sampled, so brief incidents
  between sampled frames are missed by design.

## Live stream

- **No camera on host** — `/api/stream` returns `503 {"error":"no camera"}`
  when no `/dev/video*` device exists. Use upload or the synthetic samples.
- **Single consumer** — the MJPEG generator holds the camera; concurrent
  viewers contend for the same device.

## Cloud / AWS fallback

- In **local AWS fallback mode** (badge shown in the header), S3 evidence
  archival, DynamoDB persistence, SNS alerts and the Rekognition second
  opinion are all stubbed. The Rekognition agreement card will report
  `n/a` until a real pipeline emits `rekognition_agrees` on events.

## Supervisor loop

- Approvals are in-memory only in this demo build — restarting the server
  clears the pending queue and decision history.
