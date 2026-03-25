#!/usr/bin/env bash
# Pre–Home Assistant: confirm Docker mara-wakeword has patrick.onnx and is reachable,
# then stream from your Mac mic with mic_stream_to_wyoming_wake.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== 1. Container (repo: $ROOT) ==="
if ! docker compose ps --status running mara-wakeword 2>/dev/null | grep -q "mara-wakeword"; then
  echo "mara-wakeword is not running. Start with:"
  echo "  docker compose up -d --build mara-wakeword"
  exit 1
fi

echo "=== 2. Models on the wakeword volume (/data/wakeword_models) ==="
docker compose exec -T mara-wakeword sh -c 'ls -la /data/wakeword_models'

if ! docker compose exec -T mara-wakeword sh -c 'test -f /data/wakeword_models/patrick.onnx'; then
  echo ""
  echo "WARNING: patrick.onnx is missing inside the container."
  echo "The image only seeds an *empty* volume on first run. If this volume was created"
  echo "before patrick existed, copy the model in and restart:"
  echo "  docker compose cp wakeword_service/models/patrick.onnx mara-wakeword:/data/wakeword_models/"
  echo "  docker compose restart mara-wakeword"
  exit 1
fi

echo ""
echo "=== 3. Recent startup logs (look for 'loaded custom wake model' and patrick) ==="
docker compose logs --tail 40 mara-wakeword 2>&1 | sed 's/^/  /'

echo ""
echo "=== 4. Host port 10300 ==="
if command -v nc >/dev/null 2>&1; then
  if nc -z -w 2 127.0.0.1 10300 2>/dev/null; then
    echo "  127.0.0.1:10300 is open (good)."
  else
    echo "  127.0.0.1:10300 not reachable — check docker-compose ports / firewall."
    exit 1
  fi
else
  echo "  (skipped: install 'nc' for a quick port check, or use Docker Desktop port view)"
fi

echo ""
echo "=== 5. Stream Mac mic → Wyoming (run in another terminal while you say 'patrick') ==="
MIC_PY="$ROOT/wakeword_service/mic_stream_to_wyoming_wake.py"
if [[ -x "$ROOT/wakeword_service/.venv/bin/python" ]]; then
  echo "  cd wakeword_service"
  echo "  ../wakeword_service/.venv/bin/python mic_stream_to_wyoming_wake.py \\"
  echo "    --uri tcp://127.0.0.1:10300 --describe --ping --meter"
else
  echo "  cd wakeword_service && python3 -m venv .venv && source .venv/bin/activate \\"
  echo "    && pip install -r requirements_mic_client.txt"
  echo "  .venv/bin/python mic_stream_to_wyoming_wake.py \\"
  echo "    --uri tcp://127.0.0.1:10300 --describe --ping --meter"
fi

echo ""
echo "=== 6. See openWakeWord scores while you speak (optional) ==="
echo "  In .env next to docker-compose.yml set:"
echo "    WAKE_DEBUG_OWW=1"
echo "  Then: docker compose up -d mara-wakeword && docker compose logs -f mara-wakeword"
echo ""
echo "Success: the mic client prints a line with *** wake detection: and patrick;"
echo "or container logs show patrick scores rising when you speak (use WAKE_DEBUG_OWW=1)."
echo "Lower WAKE_THRESHOLD if scores peak below the threshold."
