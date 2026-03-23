#!/bin/bash
# Train a Piper TTS voice
# Usage: ./train.sh [voice_name] [version]
# Example: ./train.sh goku v1

set -e

VOICE_NAME="${1:-goku}"
VERSION="${2:-v1}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_DIR="$SCRIPT_DIR/training"
DATASET_DIR="$TRAINING_DIR/dataset"
OUTPUT_DIR="$TRAINING_DIR/runs/${VOICE_NAME}_${VERSION}"

echo "============================================================"
echo "PIPER TTS TRAINING"
echo "============================================================"
echo "Voice: $VOICE_NAME"
echo "Version: $VERSION"
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

# Check if dataset exists
if [ ! -f "$DATASET_DIR/metadata.csv" ]; then
    echo "ERROR: Dataset not found at $DATASET_DIR"
    echo "Run 'python merge_datasets.py' first to create the training dataset"
    exit 1
fi

# Check if Piper is installed
if ! python -c "import piper_train" 2>/dev/null; then
    echo "ERROR: Piper training not installed"
    echo "Run './setup_training.sh' and follow the setup instructions"
    exit 1
fi

# Count clips
CLIP_COUNT=$(wc -l < "$DATASET_DIR/metadata.csv")
echo "Training on $CLIP_COUNT clips"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Preprocess
echo "[1/3] Preprocessing dataset..."
PREPROCESS_DIR="$OUTPUT_DIR/preprocessed"

if [ ! -d "$PREPROCESS_DIR" ]; then
    python -m piper_train.preprocess \
        --language en-us \
        --input-dir "$DATASET_DIR" \
        --output-dir "$PREPROCESS_DIR" \
        --dataset-format ljspeech \
        --single-speaker \
        --sample-rate 22050
else
    echo "  Preprocessed data exists, skipping..."
fi

# Step 2: Download base checkpoint for fine-tuning
CHECKPOINT_DIR="$TRAINING_DIR/checkpoints"
BASE_CHECKPOINT="$CHECKPOINT_DIR/en_US-lessac-medium.ckpt"

if [ ! -f "$BASE_CHECKPOINT" ]; then
    echo ""
    echo "[2/3] Downloading base checkpoint for fine-tuning..."
    mkdir -p "$CHECKPOINT_DIR"
    
    # Download lessac-medium checkpoint (good for fine-tuning)
    # Note: You may need to find a checkpoint URL or use a local one
    echo "  NOTE: You need a base Piper checkpoint for fine-tuning."
    echo "  Options:"
    echo "    1. Train from scratch (longer, needs more data)"
    echo "    2. Get a checkpoint from: https://github.com/rhasspy/piper/releases"
    echo ""
    echo "  For now, training from scratch..."
    FINETUNE_FLAG=""
else
    echo "[2/3] Using base checkpoint: $BASE_CHECKPOINT"
    FINETUNE_FLAG="--resume-from-checkpoint $BASE_CHECKPOINT"
fi

# Step 3: Train
echo ""
echo "[3/3] Starting training..."
echo "  This will take several hours. Progress is saved in checkpoints."
echo "  You can stop with Ctrl+C and resume later."
echo ""

# Determine accelerator
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    ACCELERATOR="gpu"
    DEVICES="1"
    BATCH_SIZE="32"
    echo "  Using: NVIDIA GPU (CUDA)"
elif python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    ACCELERATOR="mps"
    DEVICES="1"
    BATCH_SIZE="16"
    echo "  Using: Apple Silicon (MPS)"
else
    ACCELERATOR="cpu"
    DEVICES="auto"
    BATCH_SIZE="8"
    echo "  Using: CPU (this will be slow)"
fi

# Run training
python -m piper_train \
    --dataset-dir "$PREPROCESS_DIR" \
    --accelerator "$ACCELERATOR" \
    --devices "$DEVICES" \
    --batch-size "$BATCH_SIZE" \
    --validation-split 0.05 \
    --num-test-examples 5 \
    --max-epochs 10000 \
    --checkpoint-epochs 250 \
    --precision 32 \
    --quality medium \
    $FINETUNE_FLAG \
    --default-root-dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo "Checkpoints saved in: $OUTPUT_DIR"
echo ""
echo "To export the best checkpoint to ONNX:"
echo "  ./export.sh $VOICE_NAME $VERSION"
