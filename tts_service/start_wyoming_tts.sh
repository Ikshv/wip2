#!/bin/bash
# Startup script for Wyoming TTS server

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists (check parent directory)
if [ -d "../house-model-py312" ]; then
    source ../house-model-py312/bin/activate
    echo "Activated virtual environment: house-model-py312"
elif [ -d "../house-model" ]; then
    source ../house-model/bin/activate
    echo "Activated virtual environment: house-model"
fi

# Set default environment variables if not set
export PIPER_VOICE=${PIPER_VOICE:-"en_US-patrick-medium"}
export WYOMING_HOST=${WYOMING_HOST:-"0.0.0.0"}
export WYOMING_PORT=${WYOMING_PORT:-"10200"}

# Optional: Enable RVC
# export USE_RVC="true"
# export RVC_REPO="/path/to/rvc"
# export RVC_MODEL="/path/to/model.pth"
# export RVC_INDEX="/path/to/index.index"

echo "Starting Mara TTS Wyoming Server..."
echo "Voice: $PIPER_VOICE"
echo "Host: $WYOMING_HOST"
echo "Port: $WYOMING_PORT"

# Use Python from the virtual environment if activated, otherwise use system python3
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python3"
else
    PYTHON_CMD="python3"
fi

$PYTHON_CMD wyoming_mara_tts.py \
    --host "$WYOMING_HOST" \
    --port "$WYOMING_PORT" \
    --voice "$PIPER_VOICE"

