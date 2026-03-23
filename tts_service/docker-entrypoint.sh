#!/bin/sh
set -e
mkdir -p /data/models
if [ -z "$(ls -A /data/models 2>/dev/null)" ]; then
  echo "Seeding Piper models from image into empty volume..."
  if [ -d /app/bundled_models ] && [ -n "$(ls -A /app/bundled_models 2>/dev/null)" ]; then
    cp /app/bundled_models/* /data/models/ 2>/dev/null || true
  fi
fi
exec "$@"
