# Mara Voice Assistant Services

This project contains separate services for voice assistant functionality, organized into dedicated folders for better maintainability.

## Project Structure

```
whisp/
├── tts_service/          # Text-to-Speech service (Wyoming Protocol)
│   ├── wyoming_mara_tts.py
│   ├── start_wyoming_tts.sh
│   ├── requirements_wyoming.txt
│   ├── TTS.py
│   ├── en_US-*.onnx      # Piper voice models
│   └── ...
│
├── wakeword_service/     # Wake Word Detection service (Wyoming Protocol)
│   ├── wyoming_mara_wakeword.py
│   ├── start_wyoming_wakeword.sh
│   ├── requirements_wakeword.txt
│   ├── STT.py            # Original wake word detection code
│   └── ...
│
├── house-model-py312/    # Python 3.12 virtual environment (shared)
├── llm.py                # LLM integration (shared)
└── mara.tmpl             # Personality template (shared)
```

## Services

### 1. TTS Service (`tts_service/`)

Text-to-Speech service using Piper TTS with optional RVC voice conversion.

**Features:**
- Wyoming Protocol integration for Home Assistant
- Piper TTS with custom voices
- Optional RVC voice conversion
- Auto-discovery via Zeroconf

**Quick Start:**
```bash
cd tts_service
./start_wyoming_tts.sh
```

**Configuration:**
- Default port: `10200`
- Default voice: `en_US-patrick-medium`
- Set `PIPER_VOICE`, `WYOMING_HOST`, `WYOMING_PORT` environment variables

### 2. Wake Word Service (`wakeword_service/`)

Wake word detection service supporting custom wake words.

**Features:**
- Wyoming Protocol integration for Home Assistant
- Support for openWakeWord (recommended) or Whisper-based detection
- Custom wake word phrases
- Auto-discovery via Zeroconf

**Quick Start:**
```bash
cd wakeword_service
./start_wyoming_wakeword.sh
```

**Configuration:**
- Default port: `10300`
- Default wake words: `evil house`, `hey homey`
- Set `WYOMING_HOST`, `WYOMING_PORT`, `WAKE_WORDS` environment variables

**Detection Methods:**
1. **openWakeWord** (recommended): Better performance, lower CPU usage
   ```bash
   pip install openwakeword
   ```
2. **Whisper** (fallback): Uses faster-whisper for transcription-based detection
   ```bash
   pip install faster-whisper
   ```

## Installation

### Shared Virtual Environment

Both services share the same virtual environment (`house-model-py312`):

```bash
# Create virtual environment (if not exists)
python3.12 -m venv house-model-py312
source house-model-py312/bin/activate

# Install TTS dependencies
pip install -r tts_service/requirements_wyoming.txt

# Install Wake Word dependencies
pip install -r wakeword_service/requirements_wakeword.txt
```

### Individual Service Setup

Each service can be set up independently:

**TTS Service:**
```bash
cd tts_service
pip install -r requirements_wyoming.txt
```

**Wake Word Service:**
```bash
cd wakeword_service
pip install -r requirements_wakeword.txt
```

## Docker stack (TTS + wake word + Piper UI)

Build and run Wyoming TTS, wake word detection, and the Piper model manager on one compose stack:

```bash
docker compose up --build
```

- **TTS (Wyoming):** `tcp://<host>:10200` — Piper reads voices from the `piper_models` volume (`MODELS_DIR=/data/models`). The image seeds that volume from `tts_service/models/` the first time it is empty.
- **Wake word:** `tcp://<host>:10300` — the `wakeword_models` volume is seeded from `wakeword_service/models/` on first run (e.g. `patrick.onnx`). **Default Docker detection is openWakeWord with ONNX** (`WAKE_DETECTION_METHOD=openwakeword`): audio is fed **incrementally** per Wyoming chunk (reset only on `audio-start`). Use **`WAKE_DEBUG_OWW=1`** and `docker compose logs -f mara-wakeword` to see `patrick` scores while you speak; lower **`WAKE_THRESHOLD`** (e.g. `0.15`) if scores peak below the default. Set **`WAKE_DETECTION_METHOD=whisper`** for phrase-style detection; **`WAKE_DEBUG_WHISPER=1`** logs Whisper text in that mode.
- **Discovery:** The wake container registers **`_wyoming._tcp.local.`** (same service type Home Assistant’s Wyoming integration listens for). It advertises as **`Mara Wake Word._wyoming._tcp.local.`** with the same **Describe → Info** wake program as a Wyoming openWakeWord add-on. **`WYOMING_ZEROCONF_IPS`** defaults in `docker-compose.yml` to your **Tailscale** address so HA (on Tailscale) discovers **`tcp://100.112.105.113:10300`**. Override **`WYOMING_ZEROCONF_IPS`** in `.env` for LAN-only or a different MagicDNS/Tailscale IP.
- **Piper UI:** `http://<host>:8080` — list, upload (`.onnx` + matching `.onnx.json`), and remove Piper voices on the **same** volume as TTS so changes apply without rebuilding.

Optional `.env` next to `docker-compose.yml`:

```env
PIPER_VOICE=patrick
WAKE_WORDS=patrick
# If port 8080 is already in use on the host:
PIPER_UI_PORT=18080
# Wake / openWakeWord (optional)
WAKE_THRESHOLD=0.2
WAKE_DEBUG_OWW=0
# Wake / Whisper (optional)
WHISPER_MODEL=base.en
WAKE_DEBUG_WHISPER=0
```

### Start or stop services independently

Compose does **not** chain services together. You can run only the UI, only TTS, or any combination:

```bash
# Just the Piper model UI (TTS and wake word stay stopped)
docker compose up -d piper-ui

# Add TTS later without restarting the UI
docker compose up -d mara-tts

# Stop wake word only; leave TTS and UI running
docker compose stop mara-wakeword
```

`docker compose down` stops and removes containers for this project; use `stop` when you only want to pause specific services.

### Verify Mac mic → Docker `patrick` ONNX (before Home Assistant)

1. **Rebuild after code changes** to the wake service: `docker compose build mara-wakeword && docker compose up -d mara-wakeword`.
2. Run the checklist (models on the volume, port 10300, startup log line for custom ONNX):  
   `bash wakeword_service/verify_wake_docker.sh`
3. Stream the laptop mic to Wyoming: from `wakeword_service/` with the mic client venv, run  
   `.venv/bin/python mic_stream_to_wyoming_wake.py --uri tcp://127.0.0.1:10300 --describe --ping --meter`  
   Say **patrick**; you want `*** wake detection: 'patrick' ...` on the client. Tune **`WAKE_THRESHOLD`** / **`WAKE_DEBUG_OWW=1`** as in the wake bullet above if needed.
4. Only after that works, add the wake integration in Home Assistant at `tcp://<host>:10300`.

### If the Piper UI does not load

1. **URL:** On the machine running Docker, open `http://127.0.0.1:8080` (or the host port you set with `PIPER_UI_PORT`). From another device on your LAN, use `http://<that-computer’s-LAN-IP>:8080`, not `localhost`.
2. **Port in use:** If Docker fails to bind `8080`, set `PIPER_UI_PORT` in `.env` to another port (e.g. `18080`) and use that in the browser.
3. **Check the container:** `docker compose ps` and `docker compose logs piper-ui` — you should see Uvicorn listening on `0.0.0.0:8080` inside the container.

## Running Both Services

You can run both services simultaneously on different ports:

**Terminal 1 - TTS Service:**
```bash
cd tts_service
./start_wyoming_tts.sh
```

**Terminal 2 - Wake Word Service:**
```bash
cd wakeword_service
./start_wyoming_wakeword.sh
```

## Home Assistant Integration

Both services use the Wyoming Protocol and will be auto-discovered by Home Assistant if they're on the same network.

### Manual Configuration

If auto-discovery doesn't work, you can manually configure in Home Assistant:

1. **TTS Service:**
   - Settings → Voice Assistants → Add Integration
   - Search for "Wyoming" or "Piper"
   - Enter: `tcp://YOUR_IP:10200`

2. **Wake Word Service:**
   - Settings → Voice Assistants → Add Integration
   - Search for "Wyoming" or "openWakeWord"
   - Enter: `tcp://YOUR_IP:10300`

## Custom Wake Words

To use custom wake words with openWakeWord, you'll need to:

1. Train your own model (see openWakeWord documentation)
2. Place the model file in `~/.openWakeWord/models/`
3. Update the `_map_wake_word_to_model()` function in `wyoming_mara_wakeword.py`

Alternatively, use Whisper-based detection which works with any text phrase (see `STT.py` for reference).

## Troubleshooting

- **HA mic does not trigger wake but the Mac `mic_stream_to_wyoming_wake.py` test does:** Home Assistant sends a Wyoming **`detect`** event before **`audio-start`**. The wake server must accept it (return success); otherwise HA disconnects before streaming audio. Use a current `wakeword_service` image and watch logs for `Received Detect (names=…)` then `Audio stream started` when you use Assist.
- **Port conflicts**: Change ports in startup scripts or environment variables
- **Auto-discovery not working**: Check firewall settings, ensure services are on the same network
- **Wake word not detected**: Lower the threshold or try Whisper-based detection
- **TTS voice not found**: Ensure voice files (`.onnx`, `.onnx.json`) are in `tts_service/` directory

## Notes

- Both services can run independently or together
- The virtual environment is shared to avoid duplicate dependencies
- Original code (`STT.py`, `TTS.py`, `llm.py`) is preserved for reference
- Voice model files are stored in `tts_service/` directory


