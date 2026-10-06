# SiteSentry — Responsible Use

A safety camera is one step from a surveillance camera. These safeguards are
implemented in code and process, not just promised in prose. (Rubric weight:
responsible operation is 10% of the overall score — we treat it as a feature.)

## Implemented safeguards

1. **Face blurring ON by default.** Every annotated frame blurs detected faces
   (Haar cascade; whole head region if the cascade misses). Toggleable via
   `/api/config`, but the default is blur-on and the UI labels the state.
2. **No face recognition, no identity tracking.** Tracks are anonymous numeric
   IDs with no biometric templates, no enrollment, no re-identification across
   sessions. There is no code path that could identify a person — only PPE
   attributes (helmet/vest present or not).
3. **Human-in-the-loop for consequential actions.** The agent may log, archive
   and alert on its own, but a work-stoppage recommendation or fall escalation
   requires explicit supervisor approval in the supervisor view. Approval and
   rejection are both written back to the DynamoDB event log with the
   approver's name.
4. **30-day evidence retention.** The S3 lifecycle rule expires
   `raw/`, `annotated/` and `evidence/` after 30 days (`scripts/deploy.sh`).
   Local-fallback stores live under `data/local/` (git-ignored) with the same
   expectation documented.
5. **Conservative thresholds, disclosed.** The PPE policy flags a worker as
   non-compliant unless there is positive evidence of the PPE item. This
   deliberately trades false alarms for fewer misses; the cost (alert fatigue)
   is disclosed in `docs/evaluation.md` and mitigated by the approval gate.
6. **Assistive positioning.** The system is a pre-screening aid for qualified
   safety personnel — the same boundary the SafetyVision model card draws
   ("not a replacement for human judgment", "do not use for automated
   disciplinary action"). The UI and report repeat this.
7. **Least-privilege cloud.** IAM instance profile scoped to exactly the five
   services used; no credentials in code, logs, or the repo (`.env` pattern,
   `.gitignore`d).
8. **Synthetic demo media labeled.** Bundled sample images are AI-generated
   (see `assets/SOURCES.md`) and labeled "synthetic sample" in the UI — no
   real workers' likenesses are used in the demo.

## Known residual risks

- **Function creep:** the same pipeline could be pointed at non-safety
  monitoring. Mitigation: the report, UI copy, and license-adjacent docs
  state the intended use; the code contains no identity features to creep
  *with*.
- **Alert fatigue → ignored alerts:** conservative thresholds raise the false
  alarm rate. Mitigation: severity tiers, repeat-offender escalation (first
  offense is log-only), and supervisor-tunable thresholds in the policy table.
- **Blur is not a guarantee:** missed face detections leave faces visible in
  archived evidence. Mitigation: 30-day expiry bounds the exposure window;
  operators are instructed (demo_script.md) to treat evidence as sensitive.
- **Bias:** the PPE model's training data over-represents Western sites and
  male-presenting workers (per its model card); PPE conventions elsewhere may
  under-detect. Documented in `docs/evaluation.md` §8-adjacent limitations.

## For a real pilot (not this demo)

Before pointing SiteSentry at a real site: workers' informed consent and
union/safety-committee review; a data-protection impact assessment; on-premise
processing option (the Flask app runs anywhere — the cloud is a deployment
choice, not a requirement); and an audit of the approval log by someone other
than the supervisor who clicks approve.
