# Quick Start Guide: Wyoming Protocol Integration

Get your custom TTS voices working with Home Assistant in minutes!

## Prerequisites Check

Run the test script first:
```bash
python3 test_wyoming.py
```

This will verify all dependencies are installed.

## Installation

```bash
# Install Wyoming Protocol dependencies
pip install wyoming

# Verify Piper TTS is installed
pip install piper-tts
```

## Start the Server

### Option 1: Basic (Standard TTS)
```bash
python3 wyoming_mara_tts.py
```

### Option 2: With RVC
```bash
export USE_RVC="true"
export RVC_REPO="/path/to/rvc"
export RVC_MODEL="/path/to/model.pth"
export RVC_INDEX="/path/to/index.index"

python3 wyoming_mara_tts.py --rvc
```

### Option 3: Using Startup Script
```bash
./start_wyoming_tts.sh
```

## Add to Home Assistant

1. **Note the IP address** shown when the server starts:
   ```
   Add to Home Assistant: tcp://192.168.1.XXX:10200
   ```

2. **In Home Assistant:**
   - Settings → Devices & Services
   - Add Integration → Search "Wyoming Protocol"
   - Enter: `tcp://YOUR_IP:10200`
   - Submit

3. **Configure Assist:**
   - Settings → Voice Assistants
   - Create/Edit assistant
   - Select your Wyoming TTS under Text-to-Speech
   - Save

## Test It

In Home Assistant Developer Tools → Services:
```yaml
service: tts.speak
data:
  entity_id: media_player.your_device
  message: "Hello from Mara!"
```

## Custom Wake Word Setup

1. **Train "evil house" wake word:**
   - Use [openWakeWord Colab Notebook](https://colab.research.google.com/github/dscripka/openWakeWord/blob/main/notebooks/Training_Notebook.ipynb)
   - Download the `.tflite` file

2. **Install openWakeWord add-on in Home Assistant**

3. **Add model to `/share/openwakeword/evil_house.tflite`**

4. **Configure Assist pipeline:**
   - Wake word: openWakeWord → "evil house"
   - TTS: Your Wyoming TTS service

## Troubleshooting

**Server won't start?**
- Check port 10200 is free: `lsof -i :10200`
- Verify dependencies: `pip list | grep wyoming`

**Home Assistant can't connect?**
- Check firewall allows port 10200
- Verify IP address is correct (not localhost)
- Test connectivity: `telnet YOUR_IP 10200`

**Need help?**
- See `WYOMING_SETUP.md` for detailed documentation
- Check server logs for error messages

## What You've Got

✅ Custom TTS with Piper voices  
✅ Optional RVC voice conversion  
✅ Wyoming Protocol integration  
✅ Home Assistant auto-discovery  
✅ Ready for custom wake words  

Enjoy your custom voice assistant! 🎤

