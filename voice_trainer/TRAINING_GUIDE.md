# Piper TTS Training Guide

Your complete reference for training custom voices.

---

## Current Dataset Status

```
Voice: goku
Clips: 447
Duration: 25 minutes
Sources: goku_fighterZ, goku_base_fighterZ, goku_raging_blast
Location: training/dataset/
```

---

## Quick Reference

### Add More Training Data
```bash
cd ~/Desktop/vibes/whisp/voice_trainer
source venv/bin/activate

# Process new video
python process.py "https://youtube.com/watch?v=VIDEO_ID" --name "goku_newvideo"

# Re-merge everything
python merge_datasets.py --name goku --output training/dataset

# Check new totals
cat training/dataset/dataset_info.json
```

### Train with Google Colab (Recommended)
1. Upload `training/dataset/` to Google Drive
2. Open: https://colab.research.google.com/github/rhasspy/piper/blob/master/notebooks/piper_multilingual_training_notebook.ipynb
3. Follow the notebook steps
4. Download the resulting ONNX file

### Deploy
```bash
# Copy ONNX to TTS models folder
cp ~/Downloads/en_US-goku-medium.onnx* ../tts_service/models/
# Reload Home Assistant TTS integration
```

---

## Training with Google Colab (Recommended)

Using Google Colab is the **easiest and fastest** way to train Piper voices:
- **Free T4/A100 GPUs** - Much faster than local training
- **No dependency issues** - Everything pre-configured
- **Works from any computer** - Just need a browser

### Step 1: Prepare Your Dataset

Your dataset is already prepared at `training/dataset/`:
```
training/dataset/
├── wavs/           # 447 WAV files
├── metadata.csv    # Piper format (filename|text|text)
└── dataset_info.json
```

### Step 2: Upload to Google Drive

1. Go to https://drive.google.com
2. Create a folder called `piper_training`
3. Upload the entire `training/dataset/` folder
4. Your Drive should look like:
   ```
   My Drive/
   └── piper_training/
       └── dataset/
           ├── wavs/
           └── metadata.csv
   ```

### Step 3: Open the Training Notebook

Open this link in your browser:
**https://colab.research.google.com/github/rhasspy/piper/blob/master/notebooks/piper_multilingual_training_notebook.ipynb**

Or for the newer version:
**https://github.com/OHF-Voice/piper1-gpl** (check their notebooks folder)

### Step 4: Configure the Notebook

In the notebook, update these settings:
- **Dataset path**: `/content/drive/MyDrive/piper_training/dataset`
- **Language**: `en_us`
- **Quality**: `medium` (good balance of speed/quality)
- **Speaker**: Single speaker (since all clips are Goku)

### Step 5: Run Training

1. Click "Runtime" → "Run all"
2. When prompted, connect your Google Drive
3. Training takes **2-6 hours** depending on GPU
4. Monitor the loss - it should decrease over time

### Step 6: Export and Download

The notebook will export to ONNX format. Download:
- `en_US-goku-medium.onnx`
- `en_US-goku-medium.onnx.json`

### Step 7: Deploy

```bash
# Copy to your TTS models folder
cp ~/Downloads/en_US-goku-medium.onnx* ~/Desktop/vibes/whisp/tts_service/models/

# The TTS server auto-discovers new models on next request
# Or reload the Home Assistant integration
```

---

## Alternative: Local Training (Advanced)

Local training requires older PyTorch versions. If you prefer this:

### Option A: Docker (Cleanest)
```bash
# Use the official Piper training Docker image
docker pull rhasspy/piper-training
```

### Option B: Conda Environment
```bash
# Create environment with older packages
conda create -n piper python=3.9
conda activate piper
pip install torch==1.13.1 pytorch-lightning==1.7.7
```

---

## Training Timeline (Colab)

| GPU | 25 min data | 1 hour data | 2 hour data |
|-----|-------------|-------------|-------------|
| T4 (free) | ~4-6 hours | ~10-12 hours | ~20+ hours |
| A100 (Colab Pro) | ~1-2 hours | ~3-4 hours | ~6-8 hours |

---

## Workflow: Adding New Data Later

### Step 1: Find & Process Videos

```bash
cd ~/Desktop/vibes/whisp/voice_trainer
source venv/bin/activate

# Keep a list of videos you've processed
echo "https://youtube.com/watch?v=NEW_VIDEO" >> vids.txt

# Process each new video
python process.py "https://youtube.com/watch?v=NEW_VIDEO" --name "goku_source_name"
```

### Step 2: Review Clips (Optional but Recommended)

Listen to random samples in `output/goku_source_name/sliced/` and delete:
- Battle grunts without words
- Background music/noise
- Wrong character speaking
- Bad transcriptions

### Step 3: Merge All Datasets

```bash
# This auto-detects all folders in output/
python merge_datasets.py --name goku --output training/dataset_v2
```

### Step 4: Train New Version

```bash
./train.sh goku v2
```

### Step 5: Compare Versions

Keep both `v1` and `v2` checkpoints to compare quality.

---

## File Locations

```
voice_trainer/
├── venv/                    # Data processing venv (yt-dlp, whisper)
├── input/                   # Downloaded audio files
├── output/                  # Processed datasets (per-video)
│   ├── goku_fighterZ/
│   ├── goku_raging_blast/
│   └── ...
├── training/
│   ├── train_venv/          # Training venv (PyTorch, Piper)
│   ├── piper/               # Piper source code
│   ├── dataset/             # Merged training data
│   │   ├── wavs/
│   │   ├── metadata.csv
│   │   └── dataset_info.json
│   ├── checkpoints/         # Base models for fine-tuning
│   └── runs/                # Training outputs
│       ├── goku_v1/
│       └── goku_v2/
└── Scripts:
    ├── process.py           # Download + transcribe + slice
    ├── merge_datasets.py    # Combine datasets
    ├── setup_training.sh    # Install training tools
    ├── train.sh             # Run training
    └── export.sh            # Export to ONNX
```

---

## Troubleshooting

### "No module named piper_train"
```bash
source training/train_venv/bin/activate
cd training/piper/src/python
pip install -e ".[train]"
```

### MPS (Apple Silicon) crashes
Try reducing batch size in `train.sh`:
```bash
BATCH_SIZE="8"  # Instead of 16
```

### Out of VRAM
Reduce batch size or use CPU for preprocessing.

### YouTube download fails
```bash
# Update yt-dlp
pip install -U yt-dlp

# Or use cookies
yt-dlp --cookies-from-browser chrome "URL"
```

---

## Data Quality Tips

**Good clips:**
- Clear single speaker
- Full sentences (not just "Hah!" or "Yeah!")
- No background music
- Consistent audio quality

**Target amounts:**
- Fine-tuning: 15-30 minutes (you have 25 ✓)
- Training from scratch: 1-2+ hours

**Best sources for game characters:**
- Video game voice compilations (cleanest audio)
- Behind-the-scenes voice actor interviews
- Official character trailers

---

## Commands Cheat Sheet

```bash
# Activate data processing env
cd ~/Desktop/vibes/whisp/voice_trainer
source venv/bin/activate

# Activate training env
source training/train_venv/bin/activate

# Process video
python process.py "URL" --name "name"

# Merge datasets
python merge_datasets.py --name voice_name

# Train
./train.sh voice_name version

# Export
./export.sh voice_name version

# Check dataset info
cat training/dataset/dataset_info.json
```

---

*Last updated: January 28, 2026*
