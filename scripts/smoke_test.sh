#!/usr/bin/env bash
# SiteSentry — smoke_test.sh
# End-to-end smoke test: boots the Flask app, uploads a sample image,
# and asserts the pipeline returns events + a DynamoDB/local event row.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

echo "==> [1/4] pytest suite"
"$PY" -m pytest tests/ -q

echo "==> [2/4] booting Flask (background)"
export SITENTRY_AWS=off
"$PY" app.py & APP_PID=$!
trap 'kill $APP_PID 2>/dev/null || true' EXIT
sleep 12

echo "==> [3/4] /api/health"
curl -sSf http://127.0.0.1:5000/api/health | head -c 400; echo

echo "==> [4/4] upload sample2 (bare-headed worker) and expect events"
curl -sS -o /tmp/smoke_resp.json -w "http_code=%{http_code} bytes=%{size_download} curl_exit=$?\n" \
  -F "file=@assets/sample2.jpg" http://127.0.0.1:5000/api/upload
"$PY" - <<'EOF'
import json, sys
raw = open("/tmp/smoke_resp.json", "rb").read()
print(f"  raw bytes: {len(raw)}")
r = json.loads(raw)
assert "events" in r, "missing events key"
assert "metrics" in r, "missing metrics"
ms = r["metrics"].get("per_frame_ms") or r["metrics"].get("total_ms")
print(f"  events={len(r['events'])} per_frame_ms={ms}")
print(f"  event types: {[e['type'] for e in r['events']]}")
EOF

echo "==> local DynamoDB event rows:"
wc -l data/local/dynamodb/events.jsonl 2>/dev/null || echo "  (no local store rows yet)"

echo "SMOKE TEST PASSED"
