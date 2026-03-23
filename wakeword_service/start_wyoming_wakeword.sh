#!/bin/bash
# Startup script for Wyoming Wake Word server

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists (check parent directory)
if [ -d "../house-model-py312" ]; then
    source ../house-model-py312/bin/activate
    PYTHON_CMD="../house-model-py312/bin/python3"
    echo "Activated virtual environment: house-model-py312"
elif [ -d "../house-model" ]; then
    source ../house-model/bin/activate
    PYTHON_CMD="../house-model/bin/python3"
    echo "Activated virtual environment: house-model"
else
    PYTHON_CMD="python3"
    echo "WARNING: No virtual environment found, using system python3"
fi

echo "Starting Mara Wake Word Wyoming Server..."

# Use config file (config.yaml in same directory)
# Command line args can still override config values
$PYTHON_CMD wyoming_mara_wakeword.py --config "$SCRIPT_DIR/config.yaml" "$@"

