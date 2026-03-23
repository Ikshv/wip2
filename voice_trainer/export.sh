#!/bin/bash
# Export trained Piper model to ONNX
# Usage: ./export.sh [voice_name] [version]
# Example: ./export.sh goku v1

set -e

VOICE_NAME="${1:-goku}"
VERSION="${2:-v1}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_DIR="$SCRIPT_DIR/training"
RUN_DIR="$TRAINING_DIR/runs/${VOICE_NAME}_${VERSION}"
MODELS_DIR="$SCRIPT_DIR/../tts_service/models"

echo "============================================================"
echo "EXPORT PIPER MODEL TO ONNX"
echo "============================================================"
echo "Voice: $VOICE_NAME"
echo "Version: $VERSION"
echo "============================================================"

# Find the best checkpoint
CHECKPOINT=$(find "$RUN_DIR" -name "*.ckpt" -type f | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found in $RUN_DIR"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"

# Output filename
OUTPUT_NAME="en_US-${VOICE_NAME}-medium"
OUTPUT_ONNX="$RUN_DIR/${OUTPUT_NAME}.onnx"

# Export to ONNX
echo ""
echo "Exporting to ONNX..."
python -m piper_train.export_onnx \
    "$CHECKPOINT" \
    "$OUTPUT_ONNX"

echo ""
echo "============================================================"
echo "EXPORT COMPLETE"
echo "============================================================"
echo ""
echo "Exported model: $OUTPUT_ONNX"
echo ""

# Ask to copy to models folder
if [ -d "$MODELS_DIR" ]; then
    echo "Copy to TTS models folder? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        cp "$OUTPUT_ONNX" "$MODELS_DIR/"
        cp "${OUTPUT_ONNX}.json" "$MODELS_DIR/" 2>/dev/null || true
        echo "Copied to $MODELS_DIR/"
        echo ""
        echo "Restart your TTS service to use the new voice!"
    fi
else
    echo "To use this model:"
    echo "  cp $OUTPUT_ONNX ../tts_service/models/"
    echo "  cp ${OUTPUT_ONNX}.json ../tts_service/models/"
fi
