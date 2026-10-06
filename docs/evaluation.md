# SiteSentry — Evaluation

How we measured, what we found, and where it fails. All measurements were
taken on 2026-10-06 unless noted. Nothing here is cherry-picked: the failure
cases are required evidence, and we document them first-class.

## 1. Model-level metrics (published, held-out — SafetyVision YOLOv8s)

Source: the model card at https://huggingface.co/ayushgupta7777/safetyvision-yolov8
(4,026 held-out test images, 12,080 instances; we did not retrain).

| Class | P | R | mAP@0.5 |
|---|---|---|---|
| Fall-Detected | 0.886 | 0.937 | **0.959** |
| Hardhat | 0.888 | 0.912 | **0.937** |
| Safety Vest | 0.816 | 0.831 | 0.892 |
| Person | 0.870 | 0.798 | 0.861 |
| NO-Hardhat | 0.687 | 0.788 | 0.754 |
| NO-Gloves | 0.771 | 0.685 | 0.751 |
| **NO-Safety Vest** | 0.478 | 0.431 | **0.386** |
| all (ONNX@640 deployed) | 0.723 | 0.715 | **0.738** |

Takeaway: helmet detection is strong; **vest-absence detection is the weakest
link** (recall 0.431 — more than half of true vest violations are missed by
the raw model). This single fact drove two design decisions: (a) conservative
policy defaults, (b) the Rekognition second opinion.

## 2. Pipeline-level measurements (this build)

Run `scripts/benchmark_cool.py` to reproduce. It reports mean ± stdev per
stage, the exact OpenCV build string, the DNN engine selected, and CPU info.

Measured 2026-10-06 (build VM: AMD EPYC 9D25, x86_64 — NOT Graviton; the
Graviton/COOL numbers are a separate planned run per `docs/cool_benchmark.md`):

| Stage | Mean (ms), n=15, 1920×1280 input |
|---|---|
| decode/resize + blob | ~50 (inside detect) |
| detect (YOLOv8s PPE forward, cv2.dnn, FP32) | 363 |
| track (Kalman + IoU) | 0.2 |
| ppe (association; shares the detect forward pass) | 0.1 |
| zones (MOG2 @1920×1280) | 40 |
| annotate | ~15 |
| **Total** | **564 ± 81 → ~1.8 fps** |

This matches the model card's deployed reference ("~500–800 ms warm inference
@ 640 on Lambda-class CPU"). Raw JSON: `docs/benchmark_baseline_x86.json` (committed as the COOL-on-Graviton
comparison baseline).

| Stage | What it measures |
|---|---|
| decode/resize | frame ingest + letterbox to 640 |
| detect | YOLOv8s PPE forward pass (cv2.dnn, FP16 blob w/ FP32 fallback) |
| track | Kalman predict/update + IoU association |
| ppe | detection→track association + per-track PPE state |
| zones | MOG2 update + polygon intrusion test |
| annotate | boxes, HarfBuzz labels, face blur |

The dashboard shows live per-frame total ms on every annotated frame, and
`/api/summary` exposes rolling averages. Reference numbers from the model
card's deployment notes: ~500–800 ms/frame ONNX@640 on Lambda-class CPU
(3008 MB). Our t4g.micro target is in the same class — suitable for
inspection cadence (process every Nth frame), not 30 fps video. We report
measured numbers rather than promising real-time.

## 3. Rekognition second-opinion agreement

On every Nth flagged frame (`rekognition_sample_rate`, default 5), the agent
calls `DetectProtectiveEquipment` and compares HEAD_COVER detection against
the pipeline's helmet verdict. The dashboard's agreement card shows:

| | Rekognition: protected | Rekognition: not protected |
|---|---|---|
| **Pipeline: helmet** | agree | disagree → review |
| **Pipeline: no helmet** | disagree → review | agree |

Disagreement cases are logged as first-class events (they are the most
valuable evaluation signal we have). In local-fallback mode the agreement card
is clearly labeled `local-simulated` and reports "not measured — enable
SITENTRY_REKOGNITION with AWS credentials".

## 4. Failure cases (mandatory, documented honestly)

1. **NO-Safety Vest false negatives.** Recall 0.431 per the model card. Dark
   vests, vests under jackets, and back views are missed. Mitigation:
   conservative default (a track with no positive vest evidence is flagged),
   which trades false alarms for fewer misses — the right trade for safety,
   and every escalation still needs human approval.
2. **Occlusion.** Workers behind machinery, scaffolding poles, or each other
   lose their helmet/vest boxes; tracks can fragment. The Kalman tracker
   bridges ≤8 missed frames; beyond that the track ID resets and the
   repeat-offender counter restarts (documented limitation of the ledger key).
3. **Low light / glare.** Confidence drops at both ends of the lighting range;
   expect paired false positives and false negatives. The bundled demo uses
   daylight scenes.
4. **Small / distant figures (<50 px).** Frequently missed entirely — the
   model card calls this out and we confirm it on wide site shots.
5. **Motion blur in video.** Fast-moving workers smear the helmet/vest signal;
   the agent aggregates across frames rather than trusting any single frame.
6. **Rare PPE colors.** Training skews to yellow/white helmets and hi-vis
   yellow-orange vests; unusual colors under-detect.
7. **Fall heuristic false alarms.** The 2D aspect-ratio + motion heuristic can
   fire on workers bending or crouching. By policy, FALL_SUSPECTED *always*
   requires supervisor approval — the agent suggests, never concludes.
8. **Face-blur misses.** The Haar cascade misses profile/occluded faces; the
   fallback blurs the whole head region, which can also blur the helmet being
   inspected. Faces are never used for identity — only for blurring.

## 5. What we did NOT measure (and why)

- **End-to-end AWS latency** (S3/DynamoDB/SNS round trips): no credentials were
  available in the build environment; clients are fully wired and tested
  against local stores. Numbers will be recorded on first live deploy.
- **COOL-vs-baseline speedup:** the benchmark script is ready
  (`scripts/benchmark_cool.py`); the COOL AMI deploy is documented in
  `docs/cool_benchmark.md` but not yet executed (needs the Marketplace AMI
  subscription + instance time).
- **Precision/recall on our own labeled set:** the bundled samples are
  AI-generated synthetic scenes for demo purposes, not a labeled eval set;
  reporting P/R on 3 images would be theater. We rely on the model's published
  held-out metrics instead, and say so.
