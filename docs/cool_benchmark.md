# SiteSentry — COOL Benchmark Plan (optional special-award track)

COOL = Cloud-Optimized OpenCV Library: OpenCV builds on AWS Marketplace AMIs
(Ubuntu 24.04, Graviton/Arm), installed at `/opt/cool` with per-Python venvs
at `/opt/cool/venvs/python_3.x`. Publicized figures: ~20–30% faster on ~70 core
algorithms, up to ~70% on some ops, ~1.5× average processing speedup.

## Procedure

1. Deploy the baseline: `scripts/deploy.sh` on a standard Ubuntu 24.04
   t4g.micro (pip `opencv-python==5.0.0`).
2. Run `.venv/bin/python scripts/benchmark_cool.py --frames 120` → save JSON
   as `docs/cool_baseline.json`.
3. Subscribe to the COOL AMI on AWS Marketplace (subscription itself is free;
   you pay only EC2 hours), launch the same t4g.micro from the COOL AMI.
4. Activate `/opt/cool/venvs/python_3.12/bin/python` (check available venvs),
   install the app requirements *except* opencv-python (use COOL's build),
   copy the app over.
5. Run the same benchmark → `docs/cool_optimized.json`.
6. Compare: per-stage mean ± stdev, total ms, fps. Report speedup per stage
   and overall, plus instance cost (t4g.micro ≈ $0 on new-account Free Tier,
   ~$6–7/mo otherwise in ap-south-1).

## What counts for the award

The Best Use of COOL rubric needs: verified COOL execution of the core
image/video workload on Graviton (30%), architecture/technical quality (25%),
measured perf/cost/reliability value vs baseline (20%), innovation (15%),
reproducibility/demo (10%). The two JSON files plus this procedure are the
reproducibility evidence.

## Status

NOT YET EXECUTED — needs the Marketplace AMI subscription and ~1 hour of
t4g.micro time. The benchmark harness is ready; this file is the plan.
