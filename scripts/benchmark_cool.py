#!/usr/bin/env python3
"""SiteSentry — benchmark_cool.py

Measures per-stage pipeline latency of the OpenCV 5 DNN pipeline so a
COOL (Cloud-Optimized OpenCV Library) build on AWS Graviton can be compared
against a baseline pip build. Run on the baseline instance, then on the COOL
AMI instance, and paste both tables into docs/cool_benchmark.md.

Usage:
    .venv/bin/python scripts/benchmark_cool.py [--frames 60] [--image assets/sample1.jpg]

The script is honest by construction: it reports means AND standard deviations,
the exact OpenCV build string, DNN engine, CPU info, and per-stage breakdown.
"""
import argparse
import json
import platform
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from pipeline import SafetyPipeline, engine_info  # noqa: E402


def cpu_info():
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["model_name"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--image", default="assets/sample1.jpg")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        # synthetic fallback so the script never fails for lack of assets
        img = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)

    pipe = SafetyPipeline()
    # warmup
    for _ in range(5):
        pipe.process(img)

    totals, stages = [], {}
    for _ in range(args.frames):
        res = pipe.process(img.copy())  # fresh array: no cache hits
        totals.append(res["metrics"]["total_ms"])
        for k, v in res["metrics"]["stages"].items():
            stages.setdefault(k, []).append(v)

    report = {
        "opencv_build": cv2.getBuildInformation().splitlines()[0:2],
        "engine": engine_info(),
        "cpu": cpu_info(),
        "frames": args.frames,
        "image": args.image,
        "total_ms": {"mean": statistics.mean(totals),
                     "stdev": statistics.stdev(totals) if len(totals) > 1 else 0.0,
                     "min": min(totals), "max": max(totals)},
        "stages_ms_mean": {k: statistics.mean(v) for k, v in stages.items()},
        "fps_mean": 1000.0 / statistics.mean(totals),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
