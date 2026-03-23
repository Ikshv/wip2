#!/bin/bash
# Prepare dataset for Google Colab upload
# Creates a zip file ready to upload to Google Drive

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_DIR="$SCRIPT_DIR/training/dataset"
OUTPUT_ZIP="$SCRIPT_DIR/goku_dataset_for_colab.zip"

echo "============================================================"
echo "PREPARE DATASET FOR COLAB"
echo "============================================================"

if [ ! -d "$DATASET_DIR/wavs" ]; then
    echo "ERROR: Dataset not found at $DATASET_DIR"
    echo "Run 'python merge_datasets.py' first"
    exit 1
fi

# Count files
CLIP_COUNT=$(ls -1 "$DATASET_DIR/wavs"/*.wav 2>/dev/null | wc -l | tr -d ' ')
echo "Dataset: $CLIP_COUNT clips"

# Create zip
echo ""
echo "Creating zip file..."
cd "$SCRIPT_DIR/training"
zip -r "$OUTPUT_ZIP" dataset/ -x "*.DS_Store"

# Get size
SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)
echo ""
echo "============================================================"
echo "READY FOR UPLOAD"
echo "============================================================"
echo ""
echo "Created: $OUTPUT_ZIP"
echo "Size: $SIZE"
echo ""
echo "Next steps:"
echo "1. Upload this zip to Google Drive"
echo "2. Extract it in Drive (right-click → Extract)"
echo "3. Open the Colab notebook:"
echo "   https://colab.research.google.com/github/rhasspy/piper/blob/master/notebooks/piper_multilingual_training_notebook.ipynb"
echo ""
echo "In the notebook, set dataset path to:"
echo "   /content/drive/MyDrive/dataset"
echo ""
