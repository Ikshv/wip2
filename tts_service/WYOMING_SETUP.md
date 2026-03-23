# Wyoming Protocol TTS Server Setup

This guide explains how to set up your custom TTS (with RVC support) as a Wyoming Protocol server that Home Assistant can discover and use.

## Overview

The Wyoming Protocol is Home Assistant's native protocol for voice services. By running this server, Home Assistant can automatically discover and use your custom TTS voices.

## Installation

### 1. Install Wyoming Dependencies

**Important:** Use your virtual environment if you have one:

```bash
# Activate your virtual environment
source house-model/bin/activate

# Install dependencies
pip install -r requirements_wyoming.txt
```

Or install directly:
```bash
pip install wyoming
```

**Note:** If you encounter issues with `onnxruntime` (required by `piper-tts`), you may need to:
- Use Python 3.11 or 3.12 (Python 3.14 may not have all packages available yet)
- Or install dependencies manually: `pip install piper-tts onnxruntime`

### 2. Verify Piper TTS is Installed

```bash
pip install piper-tts
```

## Running the Server

### Basic Usage (Standard TTS)

```bash
python3 wyoming_mara_tts.py
```

This will start the server on `0.0.0.0:10200` using your default Piper voice.

### With Custom Options

```bash
python3 wyoming_mara_tts.py \
    --host 0.0.0.0 \
    --port 10200 \
    --voice en_US-patrick-medium
```

### With RVC Voice Conversion

```bash
export USE_RVC="true"
export RVC_REPO="/path/to/your/rvc/repo"
export RVC_MODEL="/path/to/model.pth"
export RVC_INDEX="/path/to/index.index"

python3 wyoming_mara_tts.py --rvc
```

Or use command-line arguments:

```bash
python3 wyoming_mara_tts.py \
    --rvc \
    --rvc-dir /path/to/rvc \
    --rvc-model /path/to/model.pth \
    --rvc-index /path/to/index.index
```

### Using the Startup Script

```bash
chmod +x start_wyoming_tts.sh
./start_wyoming_tts.sh
```

## Configuration

### Environment Variables

- `PIPER_VOICE`: Piper voice name (default: `en_US-patrick-medium`)
- `WYOMING_HOST`: Server host (default: `0.0.0.0`)
- `WYOMING_PORT`: Server port (default: `10200`)
- `USE_RVC`: Enable RVC conversion (`true`/`false`, default: `false`)
- `RVC_REPO`: Path to RVC repository
- `RVC_MODEL`: Path to RVC model file (.pth)
- `RVC_INDEX`: Path to RVC index file (.index)

## Adding to Home Assistant

### Step 1: Find Your Server IP

The server will display the IP address when it starts. Look for:
```
Add to Home Assistant: tcp://192.168.1.XXX:10200
```

### Step 2: Add Wyoming Integration

1. Open Home Assistant
2. Go to **Settings** → **Devices & Services**
3. Click **Add Integration**
4. Search for **"Wyoming Protocol"**
5. Enter your server address: `tcp://YOUR_IP:10200`
6. Click **Submit**

Home Assistant will automatically discover your TTS service!

### Step 3: Configure Assist Pipeline

1. Go to **Settings** → **Voice Assistants**
2. Create a new assistant or edit an existing one
3. Under **Text-to-Speech**, select your Wyoming TTS service
4. Save the configuration

## Testing

You can test the server is working by using Home Assistant's TTS service:

```yaml
# In Developer Tools → Services
service: tts.speak
data:
  entity_id: media_player.your_device
  message: "Hello, this is a test"
```

## Troubleshooting

### Server Won't Start

- Check that port 10200 is not in use: `lsof -i :10200`
- Verify Wyoming dependencies are installed: `pip list | grep wyoming`
- Check Python version (requires Python 3.8+)

### Home Assistant Can't Connect

- Verify the server is running and shows the IP address
- Check firewall settings (port 10200 must be accessible)
- Ensure you're using the correct IP address (not localhost)
- Try `telnet YOUR_IP 10200` to test connectivity

### RVC Not Working

- Verify RVC paths are correct
- Check that RVC Python environment exists at `RVC_REPO/.venv/bin/python`
- Ensure RVC model and index files exist
- Check server logs for RVC error messages

### Voice Not Found

- The server will automatically download voices on first use
- Check internet connection for voice downloads
- Verify voice name is correct (e.g., `en_US-patrick-medium`)

## Advanced Usage

### Running as a Service (systemd)

Create `/etc/systemd/system/mara-tts.service`:

```ini
[Unit]
Description=Mara TTS Wyoming Server
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/whisp
Environment="PIPER_VOICE=en_US-patrick-medium"
Environment="USE_RVC=true"
Environment="RVC_REPO=/path/to/rvc"
ExecStart=/usr/bin/python3 /path/to/whisp/wyoming_mara_tts.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable mara-tts
sudo systemctl start mara-tts
```

### Multiple Voice Instances

Run multiple servers on different ports for different voices:

```bash
# Standard voice
python3 wyoming_mara_tts.py --port 10200 --voice en_US-patrick-medium

# RVC voice
python3 wyoming_mara_tts.py --port 10201 --voice en_US-patrick-medium --rvc
```

Then add both to Home Assistant as separate Wyoming integrations.

## Setting Up openWakeWord for Custom Wake Words

### Step 1: Train Your "evil house" Wake Word

1. **Use the openWakeWord Training Notebook:**
   - Open the [Google Colab Training Notebook](https://colab.research.google.com/github/dscripka/openWakeWord/blob/main/notebooks/Training_Notebook.ipynb)
   - Enter "evil house" as your wake word
   - Follow the notebook to generate training data and train the model
   - Download the `.tflite` file when complete

2. **Alternative: Use Home Assistant Training:**
   - Home Assistant has a built-in wake word training tool
   - Go to Settings → Voice Assistants → Create Wake Word
   - Follow the guided process

### Step 2: Install openWakeWord Add-on

1. In Home Assistant: **Settings** → **Add-ons** → **Add-on Store**
2. Search for "openWakeWord"
3. Click **Install** and then **Start**

### Step 3: Add Your Custom Model

1. **If using Home Assistant OS:**
   - Use Samba add-on or SSH to access your Home Assistant
   - Navigate to `/share/openwakeword/`
   - Place your `evil_house.tflite` file here

2. **If using standalone openWakeWord:**
   - Place the `.tflite` file in the models directory
   - Restart the openWakeWord service

### Step 4: Configure in Home Assistant

1. **Settings** → **Devices & Services**
2. Under **Discovered**, you should see openWakeWord (Wyoming integration)
3. Click **Configure** → **Submit**

### Step 5: Add to Assist Pipeline

1. **Settings** → **Voice Assistants**
2. Create or edit your assistant
3. Under **Wake word**, select:
   - **Engine**: openWakeWord
   - **Wake word**: evil house
4. Under **Text-to-Speech**, select your Wyoming TTS service
5. Save the configuration

## Next Steps

- Test your custom wake word: say "evil house" to activate
- Configure your Assist pipeline with both wake word and TTS
- Test voice commands with your custom setup
- Adjust RVC settings if needed for voice quality

## Support

For issues or questions:
- Check Home Assistant Wyoming Protocol documentation
- Review server logs for error messages
- Verify all environment variables are set correctly

