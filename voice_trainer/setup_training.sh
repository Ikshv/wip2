#!/bin/bash
# Setup Piper training environment
# Run this once to prepare for training

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_DIR="$SCRIPT_DIR/training"

echo "============================================================"
echo "PIPER TTS TRAINING SETUP"
echo "============================================================"

# Create training directory structure
mkdir -p "$TRAINING_DIR"
cd "$TRAINING_DIR"

# Clone Piper if not exists
if [ ! -d "piper" ]; then
    echo "Cloning Piper repository..."
    git clone https://github.com/rhasspy/piper.git
else
    echo "Piper already cloned"
fi

# Create training venv if not exists
if [ ! -d "train_venv" ]; then
    echo "Creating training virtual environment..."
    python3.10 -m venv train_venv || python3.11 -m venv train_venv || python3.12 -m venv train_venv
fi

echo ""
echo "============================================================"
echo "NEXT STEPS"
echo "============================================================"
echo ""
echo "1. Activate the training environment:"
echo "   source training/train_venv/bin/activate"
echo ""
echo "2. Install PyTorch (choose based on your GPU):"
echo ""
echo "   # NVIDIA GPU (CUDA 12.1):"
echo "   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo ""
echo "   # Apple Silicon (MPS - experimental):"
echo "   pip install torch torchaudio"
echo ""
echo "   # CPU only (slow but works):"
echo "   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
echo ""
echo "3. Install Piper training dependencies:"
echo "   cd training/piper/src/python"
echo "   pip install -e ."
echo "   pip install -e \".[train]\""
echo ""
echo "4. Run the training script:"
echo "   cd $SCRIPT_DIR"
echo "   ./train.sh"
echo ""
