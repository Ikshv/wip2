# Custom Wake Word Training Guide

Train your own wake words for Home Assistant using openWakeWord.

---

## Quick Start

1. **Open the training notebook:**
   https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb

2. **Enter your wake word** (e.g., "GLaDOS", "Hey Jarvis", "Computer")

3. **Listen to pronunciation** - tweak spelling if it sounds wrong
   - Example: "GLaDOS" might need to be "glados" or "glad os"

4. **Run all cells** (~1 hour)

5. **Download the files:**
   - `your_wake_word.tflite`
   - `your_wake_word.onnx`

6. **Copy to models folder:**
   ```bash
   cp ~/Downloads/glados.onnx ~/Desktop/vibes/whisp/wakeword_service/models/
   ```

7. **Update wake words in start script** (or use environment variable)

---

## Trained Wake Words

Place your trained `.onnx` files here:
```
wakeword_service/models/
├── glados.onnx
├── hey_computer.onnx
└── ...
```

---

## Wake Word Ideas

| Wake Word | Notes |
|-----------|-------|
| GLaDOS | Portal AI - spell as "glados" or "glad os" |
| Hey Computer | Star Trek style |
| OK House | Generic home assistant |
| Hey Jarvis | Iron Man AI |
| Evil House | Spooky option |

---

## Tips for Good Wake Words

1. **3-4 syllables** work best (e.g., "Hey Jarvis" = 3, "GLaDOS" = 2-3)
2. **Uncommon phrases** reduce false positives
3. **Clear consonants** (k, t, p, s) are easier to detect
4. **Avoid common words** that appear in normal speech

---

## Troubleshooting

### Wake word not detecting
- Try tweaking the spelling in the Colab notebook
- Lower the detection threshold (0.3 instead of 0.5)
- Speak closer to the microphone
- Reduce background noise

### Too many false positives
- Increase the threshold (0.7 instead of 0.5)
- Re-train with different spelling
- Choose a more unique wake word

### Model not loading
- Ensure file is `.onnx` format (not `.tflite`)
- Check file permissions
- Verify the model path in the config

---

## Using Custom Models

### Option 1: Environment Variable
```bash
export WAKE_WORDS="glados|hey computer|evil house"
./start_wyoming_wakeword.sh
```

### Option 2: Edit start script
Edit `start_wyoming_wakeword.sh`:
```bash
export WAKE_WORDS=${WAKE_WORDS:-"glados|hey computer"}
```

### Option 3: Command line
```bash
python wyoming_mara_wakeword.py --wake-words "glados" "hey computer"
```

---

## How It Works

The training notebook:
1. Uses **Piper TTS** to generate thousands of synthetic voice samples saying your wake word
2. Augments audio with noise, reverb, pitch shifts
3. Trains a small neural network to recognize the pattern
4. Exports to `.onnx` format for fast inference

Training takes ~1 hour on Colab's free GPU tier.

---

## Current Detection Methods

Your wake word service supports two methods:

### 1. openWakeWord (Recommended)
- Uses trained `.onnx` models
- Fast, low CPU usage
- Requires specific model for each wake word

### 2. Whisper Fallback
- Transcribes audio and looks for the phrase
- Works with ANY phrase (no training needed)
- Slower, higher CPU usage
- Set with: `--detection-method whisper`

---

*Last updated: January 28, 2026*
