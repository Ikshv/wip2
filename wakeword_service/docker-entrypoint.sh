#!/bin/sh
set -e
mkdir -p /data/wakeword_models
if [ -z "$(ls -A /data/wakeword_models 2>/dev/null)" ]; then
  if [ -d /app/bundled_models ] && [ -n "$(ls -A /app/bundled_models 2>/dev/null)" ]; then
    echo "Seeding wake word models from image into empty volume..."
    cp /app/bundled_models/* /data/wakeword_models/ 2>/dev/null || true
  fi
fi
exec "$@"
