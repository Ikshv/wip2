# Voice Trainer

Complete pipeline for creating custom Piper TTS voices from YouTube videos.

## Quick Start

```bash
cd voice_trainer
source venv/bin/activate

# 1. Process a YouTube video
python process.py "https://youtube.com/watch?v=VIDEO_ID" --name "character_name"

# 2. Repeat for more videos...

# 3. Merge all datasets
python merge_datasets.py --name character_name

# 4. Train the model (requires GPU)
./train.sh character_name v1

# 5. Export to ONNX
./export.sh character_name v1
```

## Setup

```bash
cd voice_trainer

# Create virtual environment (if not exists)
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Make sure ffmpeg is installed
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

## Phase 1: Collect Training Data

### Process YouTube Videos

```bash
# Process a video with automatic download, transcription, and slicing
python process.py "https://www.youtube.com/watch?v=VIDEO_ID" --name "goku_game1"

# Process more videos
python process.py "https://www.youtube.com/watch?v=VIDEO_ID2" --name "goku_game2"
python process.py "https://www.youtube.com/watch?v=VIDEO_ID3" --name "goku_anime"
```

### Options

```bash
python process.py URL [OPTIONS]

Options:
  --name, -n       Output folder name (default: video title)
  --model, -m      Whisper model: tiny/base/small/medium/large-v3 (default: medium)
  --language, -l   Language code (default: en)
  --min-duration   Minimum clip duration in seconds (default: 0.5)
  --max-duration   Maximum clip duration in seconds (default: 15.0)
```

### Output Structure

```
output/
├── goku_game1/sliced/
│   ├── *.wav files (22050 Hz mono)
│   ├── metadata.csv
│   └── metadata.json
├── goku_game2/sliced/
│   └── ...
└── transcripts/
    └── *.json (raw Whisper output)
```

## Phase 2: Prepare Training Dataset

### Merge All Datasets

```bash
# Auto-detect and merge all datasets in output/
python merge_datasets.py --name goku

# Or specify which ones to merge
python merge_datasets.py --name goku output/goku_game1/sliced output/goku_game2/sliced
```

This creates:
```
training/dataset/
├── wavs/           # All clips renamed: goku_00000.wav, goku_00001.wav, ...
├── metadata.csv    # LJSpeech format for Piper
├── metadata.json   # Full metadata with sources
└── dataset_info.json
```

### Review & Clean (Recommended)

Listen to random samples and remove bad clips:
- Background noise/music
- Multiple speakers
- Grunts without words
- Bad transcriptions

## Phase 3: Train the Model

### Setup Training Environment

```bash
./setup_training.sh

# Follow the printed instructions to:
# 1. Activate training venv
# 2. Install PyTorch (CUDA/MPS/CPU)
# 3. Install Piper training tools
```

### Train

```bash
# Train version 1
./train.sh goku v1

# Later, with more data, train version 2
./train.sh goku v2
```

Training takes **4-24 hours** depending on:
- GPU speed
- Amount of data
- Whether fine-tuning or training from scratch

### Export to ONNX

```bash
./export.sh goku v1
```

This creates `en_US-goku-medium.onnx` ready for your TTS server.

## Phase 4: Use the Voice

```bash
# Copy to TTS models folder
cp training/runs/goku_v1/en_US-goku-medium.onnx* ../tts_service/models/

# Restart TTS server (it auto-discovers new models)
```

## Data Requirements

| Data Amount | Training Type | Quality |
|-------------|---------------|---------|
| 15-30 min | Fine-tune existing | Decent |
| 1-2 hours | Train from scratch | Good |
| 5+ hours | Train from scratch | Excellent |

## Whisper Model Selection

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| tiny | 39M | Fastest | Lower | ~1GB |
| base | 74M | Fast | OK | ~1GB |
| small | 244M | Medium | Good | ~2GB |
| **medium** | 769M | Slower | Better | ~5GB |
| large-v3 | 1.5G | Slowest | Best | ~10GB |

`medium` is recommended for character voice clips.

## Tips

- **Game audio** is cleaner than TV/movie audio (no background music)
- **Consistent voice actor** - don't mix different actors for the same character
- **Full sentences** work better than short exclamations for TTS
- **Remove battle grunts** - they confuse the model
- **Version your training runs** - compare v1 vs v2 with more data
