#!/usr/bin/env bash
# SiteSentry — download_models.sh
# Fetches the ONNX models used by the OpenCV 5 DNN pipeline.
# Sources are public and cited below; licenses noted per model.
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../models" && pwd)"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

echo "==> [1/2] YOLOv8n COCO person detector (OPTIONAL)"
echo "    NOTE (2026-10-06): the canonical ultralytics/assets URL no longer ships a"
echo "    yolov8n.onnx asset, and Hugging Face gated-mirror downloads returned 401"
echo "    from this build network. Person detection therefore runs on the PPE model's"
echo "    own 'Person' class (class 11) — one forward pass yields persons + PPE items"
echo "    + falls, which also halves inference cost on t4g.micro. If you place a"
echo "    YOLOv8n COCO ONNX at models/yolov8n.onnx, PersonDetector will use it"
echo "    automatically (see pipeline/detector.py)."
echo "    (Attempted sources, for the record:"
echo "     - https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx -> 404"
echo "     - https://huggingface.co/onnx-community/yolov8n/resolve/main/onnx/model.onnx -> 401)"
echo "    skipping."

echo "==> [2/2] SafetyVision YOLOv8s PPE detector, 13 classes (~43 MB, single-file ONNX)"
echo "    Source : https://huggingface.co/ayushgupta7777/safetyvision-yolov8 (file v2/best_640.onnx)"
echo "    Model card reports held-out test mAP@0.5 = 0.766 (imgsz 896) / 0.754 (640),"
echo "    ONNX@640 = 0.738. Classes: Fall-Detected, Gloves, Goggles, Hardhat, Mask,"
echo "    NO-Gloves, NO-Goggles, NO-Hardhat, NO-Mask, NO-Safety Vest, No_Harness,"
echo "    Person, Safety Vest."
echo "    License: YOLOv8 lineage is AGPL-3.0; no separate weights license is stated on"
echo "             the model card — treat as AGPL-3.0 and verify before commercial use."
if [ -f ppe_yolov8s.onnx ]; then
  echo "    already present, skipping."
else
  curl -sSL --retry 3 -o ppe_yolov8s.onnx \
    https://huggingface.co/ayushgupta7777/safetyvision-yolov8/resolve/main/v2/best_640.onnx
  echo "    saved: $(du -h ppe_yolov8s.onnx | cut -f1)"
fi

echo "==> verifying downloads are valid files"
file yolov8n.onnx ppe_yolov8s.onnx | sed 's/^/    /'

echo "==> quick OpenCV 5 load check"
"$(dirname "${BASH_SOURCE[0]}")/../.venv/bin/python" - <<'EOF'
import cv2, os, glob
files = sorted(glob.glob("models/*.onnx"))
assert files, "no .onnx models found in models/"
for f in files:
    net = cv2.dnn.readNet(f)
    print(f"  {f}: loaded OK, {len(net.getLayerNames())} layers")
EOF

echo "DONE. Models live in $MODEL_DIR (git-ignored; not committed)."
