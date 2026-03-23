#!/usr/bin/env python3
"""
Wyoming Protocol Wake Word Detection Server for Mara
Supports custom wake words using openWakeWord or Whisper-based detection
"""
import asyncio
import os
import sys
import argparse
import threading
import socket
import numpy as np
import yaml
from typing import Optional, List, Dict, Any
from collections import deque

try:
    from zeroconf import ServiceInfo, Zeroconf
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False

try:
    from wyoming.server import AsyncServer, AsyncEventHandler
    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.info import Info, Describe, WakeModel, WakeProgram, Attribution
    from wyoming.wake import Detection
    from wyoming.ping import Ping, Pong
    from wyoming.error import Error
except ImportError as e:
    print(f"ERROR: Wyoming dependencies not installed: {e}")
    print("Install with: pip install wyoming")
    sys.exit(1)

# Try to import openWakeWord (preferred method)
try:
    from openwakeword import Model
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    print("WARNING: openWakeWord not available. Install with: pip install openwakeword")
    print("Will attempt to use Whisper-based detection if available.")


def discover_custom_wakeword_models(models_dir: str) -> dict:
    """
    Discover custom wake word models in the models directory.
    Returns a dict mapping wake word names to model paths.
    """
    models = {}
    if not os.path.isdir(models_dir):
        return models
    
    for filename in os.listdir(models_dir):
        if filename.endswith('.onnx'):
            # Extract wake word name from filename
            # e.g., "glados.onnx" -> "glados"
            # e.g., "hey_computer_v1.onnx" -> "hey computer"
            name = filename.replace('.onnx', '')
            # Convert underscores to spaces, remove version suffixes
            name = name.replace('_', ' ')
            # Remove common suffixes like v1, v0.1, etc.
            import re
            name = re.sub(r'\s*v?\d+\.?\d*$', '', name)
            name = name.strip()
            
            models[name] = os.path.join(models_dir, filename)
            print(f"Discovered custom wake word model: '{name}' -> {filename}")
    
    return models

# Try to import Whisper for fallback detection
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class MaraWakeWordEventHandler(AsyncEventHandler):
    """Event handler for Wyoming wake word detection requests"""
    
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        wake_words: List[str],
        detection_method: str = "openwakeword",
        threshold: float = 0.5,
        models_dir: Optional[str] = None,
        custom_models: Optional[dict] = None,
        cooldown_seconds: float = 2.0,
        debug_scores: bool = False,
        min_score_to_log: float = 0.1,
    ):
        super().__init__(reader, writer)
        self.wake_words = wake_words
        self.detection_method = detection_method
        self.threshold = threshold
        self.models = {}
        self.whisper_model = None
        self.audio_buffer = deque(maxlen=16000 * 3)  # 3 seconds at 16kHz
        self.sample_rate = 16000
        self._last_detection_time = 0.0
        self._cooldown_seconds = cooldown_seconds
        self._debug_scores = debug_scores
        self._min_score_to_log = min_score_to_log
        self.models_dir = models_dir or os.path.join(os.path.dirname(__file__), "models")
        self.custom_models = custom_models or {}
        
        # Initialize detection models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize wake word detection models"""
        if self.detection_method == "openwakeword" and OPENWAKEWORD_AVAILABLE:
            try:
                # Load custom models if available
                custom_model_paths = list(self.custom_models.values())
                
                if custom_model_paths:
                    try:
                        # Load openWakeWord with custom models
                        self.models["custom"] = Model(
                            wakeword_models=custom_model_paths,
                            inference_framework="onnx"
                        )
                        print(f"Loaded {len(custom_model_paths)} custom wake word model(s)")
                    except Exception as e:
                        print(f"WARNING: Failed to load custom models: {e}")
                
                # Also try to load default openWakeWord models
                try:
                    # Try loading with no models specified (will use available ONNX models)
                    test_model = Model(inference_framework="onnx")
                    print("Loaded default openWakeWord model (ONNX)")
                    # Use a single model instance for all wake words
                    self.models["default"] = test_model
                except Exception as e:
                    error_msg = str(e).lower()
                    if "tflite" in error_msg:
                        print("WARNING: openWakeWord default models require tflite-runtime (not available on macOS ARM)")
                        if not self.models:
                            print("Falling back to Whisper-based detection")
                            self.detection_method = "whisper" if WHISPER_AVAILABLE else None
                    elif not self.models:
                        raise e
            except Exception as e:
                print(f"ERROR initializing openWakeWord: {e}")
                print("Falling back to Whisper-based detection")
                self.detection_method = "whisper" if WHISPER_AVAILABLE else None
        
        elif self.detection_method == "whisper" and WHISPER_AVAILABLE:
            try:
                self.whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
                print("Loaded Whisper model for wake word detection")
            except Exception as e:
                print(f"ERROR loading Whisper model: {e}")
                self.detection_method = None
        
        if not self.detection_method and not self.models:
            print("ERROR: No wake word detection method available!")
            print("Install openWakeWord: pip install openwakeword")
            print("Or install Whisper: pip install faster-whisper")
    
    def _map_wake_word_to_model(self, wake_word: str) -> Optional[str]:
        """Map custom wake word names to openWakeWord model names"""
        # Map common wake words to pre-trained models
        mapping = {
            "hey jarvis": "jarvis_v0.1.onnx",
            "hey siri": "hey_siri_v0.1.onnx",
            "hey google": "hey_google_v0.1.onnx",
            "alexa": "alexa_v0.1.onnx",
            "hey homey": "hey_jarvis_v0.1.onnx",  # Use jarvis as fallback
            "evil house": "hey_jarvis_v0.1.onnx",  # Use jarvis as fallback
        }
        wake_lower = wake_word.lower().strip()
        return mapping.get(wake_lower)
    
    def _map_model_to_wake_word(self, model_name: str) -> Optional[str]:
        """Map detected model name back to our wake word"""
        # First, check custom models - these have direct mappings
        for wake_word, model_path in self.custom_models.items():
            model_filename = os.path.basename(model_path)
            if model_filename in model_name or model_name in model_filename:
                return wake_word
        
        # Reverse mapping for built-in models
        model_to_wake = {
            "jarvis_v0.1.onnx": "hey homey",
            "hey_siri_v0.1.onnx": "hey homey",
            "hey_google_v0.1.onnx": "hey homey",
            "alexa_v0.1.onnx": "hey homey",
        }
        
        # Check if model name is in our mapping
        if model_name in model_to_wake:
            return model_to_wake[model_name]
        
        # Check if any of our wake words match the model name
        for wake_word in self.wake_words:
            wake_clean = wake_word.lower().replace(' ', '_').replace('-', '_')
            model_clean = model_name.lower().replace(' ', '_').replace('-', '_')
            if wake_clean in model_clean or model_clean in wake_clean:
                return wake_word
        
        # Return first wake word as fallback
        return self.wake_words[0] if self.wake_words else None
    
    async def disconnect(self):
        """Handle client disconnection"""
        print("Client disconnected")
        await super().disconnect()
    
    async def handle_event(self, event):
        """Handle incoming Wyoming Protocol events"""
        try:
            event_type = getattr(event, 'type', None)
            
            # Handle ping events (keepalive)
            if event_type == "ping":
                try:
                    pong_event = Pong().event()
                    await self.write_event(pong_event)
                    return True
                except Exception as e:
                    print(f"ERROR: Failed to respond to ping: {e}")
                    return False
            
            # Handle describe events
            if event_type == "describe" or event_type is None:
                try:
                    describe = Describe.from_event(event)
                    if describe is not None:
                        await self.handle_describe(describe)
                        return True
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"Failed to convert to Describe: {e}")
            
            # Handle audio start
            if event_type == "audio-start":
                try:
                    audio_start = AudioStart.from_event(event)
                    if audio_start:
                        self.sample_rate = audio_start.rate
                        self.audio_buffer.clear()
                        print(f"Audio stream started: {self.sample_rate}Hz")
                        return True
                except Exception as e:
                    print(f"Error handling audio start: {e}")
            
            # Handle audio chunks
            if event_type == "audio-chunk":
                try:
                    audio_chunk = AudioChunk.from_event(event)
                    if audio_chunk:
                        await self.handle_audio_chunk(audio_chunk)
                        return True
                except Exception as e:
                    print(f"Error handling audio chunk: {e}")
            
            # Handle audio stop
            if event_type == "audio-stop":
                try:
                    self.audio_buffer.clear()
                    print("Audio stream stopped")
                    return True
                except Exception as e:
                    print(f"Error handling audio stop: {e}")
            
        except Exception as e:
            print(f"ERROR in handle_event: {e}")
            import traceback
            traceback.print_exc()
        
        return False
    
    async def handle_describe(self, describe: Describe):
        """Handle Describe request - return available wake words"""
        print("Received Describe event")
        
        # Create wake word models
        wake_models = []
        for wake_word in self.wake_words:
            wake_model = WakeModel(
                name=wake_word,
                attribution=Attribution(name="Mara", url=""),
                installed=True,
                description=f"Custom wake word: {wake_word}",
                version="1.0.0",
                languages=["en", "en-US"],
                phrase=wake_word,
            )
            wake_models.append(wake_model)
        
        # Create wake program
        wake_program = WakeProgram(
            name="mara",
            attribution=Attribution(name="Mara", url=""),
            installed=True,
            description="Mara custom wake words",
            version="1.0.0",
            models=wake_models,
        )
        
        # Create info response (wake must be a list)
        info = Info(
            wake=[wake_program],
        )
        
        await self.write_event(info.event())
        print("Sent Info response with wake word models")
    
    async def handle_audio_chunk(self, audio_chunk: AudioChunk):
        """Process audio chunk for wake word detection"""
        try:
            # Convert audio bytes to numpy array
            if audio_chunk.width == 2:  # 16-bit
                audio_data = np.frombuffer(audio_chunk.audio, dtype=np.int16).astype(np.float32) / 32768.0
            elif audio_chunk.width == 4:  # 32-bit float
                audio_data = np.frombuffer(audio_chunk.audio, dtype=np.float32)
            else:
                print(f"Unsupported audio width: {audio_chunk.width}")
                return
            
            # Add to buffer
            self.audio_buffer.extend(audio_data)
            
            # Check for wake words periodically (every ~1 second of audio)
            if len(self.audio_buffer) >= self.sample_rate:  # 1 second
                await self._check_wake_words()
        
        except Exception as e:
            print(f"ERROR processing audio chunk: {e}")
            import traceback
            traceback.print_exc()
    
    async def _check_wake_words(self):
        """Check audio buffer for wake words"""
        import time
        
        # Cooldown check
        current_time = time.time()
        if current_time - self._last_detection_time < self._cooldown_seconds:
            return
        
        # Get recent audio (last 2 seconds)
        buffer_array = np.array(list(self.audio_buffer)[-self.sample_rate * 2:])
        
        if len(buffer_array) < self.sample_rate * 0.5:  # Need at least 0.5 seconds
            return
        
        detected = False
        
        if self.detection_method == "openwakeword" and self.models:
            # Use openWakeWord for detection
            try:
                # Check custom models first (higher priority)
                custom_model = self.models.get("custom")
                if custom_model and not detected:
                    prediction = custom_model.predict(buffer_array)
                    for model_name, score in prediction.items():
                        # Debug logging for all scores
                        if self._debug_scores and score >= self._min_score_to_log:
                            print(f"[DEBUG] {model_name}: {score:.3f} (threshold: {self.threshold})")
                        
                        if score > self.threshold:
                            detected_wake_word = self._map_model_to_wake_word(model_name) or self.wake_words[0]
                            print(f"Wake word detected: {detected_wake_word} (custom model: {model_name}, score: {score:.2f})")
                            await self.write_event(
                                Detection(
                                    name=detected_wake_word,
                                    timestamp=current_time,
                                    model=model_name,
                                    score=score,
                                ).event()
                            )
                            self._last_detection_time = current_time
                            detected = True
                            break
                
                # Then check default models
                default_model = self.models.get("default")
                if default_model and not detected:
                    prediction = default_model.predict(buffer_array)
                    for model_name, score in prediction.items():
                        # Debug logging for all scores
                        if self._debug_scores and score >= self._min_score_to_log:
                            print(f"[DEBUG] {model_name}: {score:.3f} (threshold: {self.threshold})")
                        
                        if score > self.threshold:
                            detected_wake_word = self._map_model_to_wake_word(model_name) or self.wake_words[0]
                            print(f"Wake word detected: {detected_wake_word} (model: {model_name}, score: {score:.2f})")
                            await self.write_event(
                                Detection(
                                    name=detected_wake_word,
                                    timestamp=current_time,
                                    model=model_name,
                                    score=score,
                                ).event()
                            )
                            self._last_detection_time = current_time
                            detected = True
                            break
            except Exception as e:
                print(f"ERROR in openWakeWord prediction: {e}")
        
        elif self.detection_method == "whisper" and self.whisper_model:
            # Use Whisper for detection (fallback method)
            try:
                # Transcribe the audio
                segments, info = self.whisper_model.transcribe(
                    buffer_array,
                    beam_size=1,
                    language="en",
                    condition_on_previous_text=False,
                )
                
                # Get the transcribed text
                text = " ".join([segment.text for segment in segments]).lower()
                
                # Check if any wake word is in the text
                for wake_word in self.wake_words:
                    if wake_word.lower() in text:
                        print(f"Wake word detected via Whisper: {wake_word} (text: {text})")
                        await self.write_event(
                            Detection(
                                name=wake_word,
                                timestamp=current_time,
                                model="whisper",
                                score=0.8,  # Whisper doesn't provide confidence scores
                            ).event()
                        )
                        self._last_detection_time = current_time
                        detected = True
                        break
            except Exception as e:
                print(f"ERROR in Whisper transcription: {e}")
        
        if not detected and self.detection_method is None:
            print("WARNING: No wake word detection method available")


class MaraWakeWordServer:
    """Wyoming Protocol wake word detection server"""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 10300,
        wake_words: Optional[List[str]] = None,
        detection_method: str = "openwakeword",
        threshold: float = 0.5,
        models_dir: Optional[str] = None,
        cooldown_seconds: float = 2.0,
        debug_scores: bool = False,
        min_score_to_log: float = 0.1,
    ):
        self.host = host
        self.port = port
        self.detection_method = detection_method
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.debug_scores = debug_scores
        self.min_score_to_log = min_score_to_log
        self.server: Optional[AsyncServer] = None
        self._port = port
        self.zeroconf: Optional[Zeroconf] = None
        
        # Set up models directory
        self.models_dir = models_dir or os.path.join(os.path.dirname(__file__), "models")
        
        # Discover custom wake word models
        self.custom_models = discover_custom_wakeword_models(self.models_dir)
        
        # Combine provided wake words with discovered custom models
        default_wake_words = ["evil house", "hey homey", "hey patrick"]
        if wake_words:
            self.wake_words = wake_words
        else:
            self.wake_words = default_wake_words
        
        # Add discovered model names to wake words if not already present
        for model_name in self.custom_models.keys():
            if model_name.lower() not in [w.lower() for w in self.wake_words]:
                self.wake_words.append(model_name)
                print(f"Added discovered wake word: '{model_name}'")
    
    def _create_handler(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Create a new event handler for a client connection"""
        return MaraWakeWordEventHandler(
            reader=reader,
            writer=writer,
            wake_words=self.wake_words,
            detection_method=self.detection_method,
            threshold=self.threshold,
            models_dir=self.models_dir,
            custom_models=self.custom_models,
            cooldown_seconds=self.cooldown_seconds,
            debug_scores=self.debug_scores,
            min_score_to_log=self.min_score_to_log,
        )
    
    async def run(self):
        """Run the Wyoming Protocol server"""
        uri = f"tcp://{self.host}:{self.port}"
        print(f"Starting Mara Wake Word Wyoming Server on {uri}")
        
        # Create server
        self.server = AsyncServer.from_uri(uri)
        
        # Run server with handler factory
        await self.server.run(self._create_handler)
    
    def _get_all_ips(self):
        """Get all available IP addresses for Zeroconf advertisement"""
        import socket
        import ifaddr
        
        ips = []
        for adapter in ifaddr.get_adapters():
            for ip in adapter.ips:
                if isinstance(ip.ip, str) and not ip.ip.startswith("127."):
                    try:
                        socket.inet_aton(ip.ip)
                        ips.append(ip.ip)
                    except:
                        pass
        return ips if ips else ["127.0.0.1"]
    
    def _register_zeroconf(self):
        """Register service with Zeroconf for auto-discovery"""
        if not ZEROCONF_AVAILABLE:
            print("WARNING: Zeroconf not available, skipping service registration")
            return
        
        def register_in_thread():
            try:
                self.zeroconf = Zeroconf()
                ips = self._get_all_ips()
                
                for ip in ips:
                    info = ServiceInfo(
                        "_wyoming._tcp.local.",
                        f"Mara Wake Word.{ip}.{self._port}._wyoming._tcp.local.",
                        addresses=[socket.inet_aton(ip)],
                        port=self._port,
                        properties={
                            "name": "Mara Wake Word",
                            "version": "1.0.0",
                        },
                    )
                    self.zeroconf.register_service(info)
                    print(f"Registered Zeroconf service on {ip}:{self._port}")
            except Exception as e:
                print(f"WARNING: Zeroconf registration failed: {e}")
        
        # Run in background thread to avoid blocking
        thread = threading.Thread(target=register_in_thread, daemon=True)
        thread.start()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    default_config = {
        "server": {"host": "0.0.0.0", "port": 10300},
        "detection": {
            "method": "openwakeword",
            "threshold": 0.5,
            "cooldown_seconds": 2.0,
        },
        "wake_words": [],
        "models_dir": "./models",
        "debug": {
            "log_all_scores": False,
            "min_score_to_log": 0.1,
        },
    }
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}, using defaults")
        return default_config
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Merge with defaults
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey not in config[key]:
                        config[key][subkey] = subvalue
        
        print(f"Loaded config from: {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}, using defaults")
        return default_config


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Mara Wake Word Wyoming Server")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument(
        "--wake-words",
        nargs="+",
        default=None,
        help="Wake words to detect (auto-discovers from models/ if not specified)",
    )
    parser.add_argument(
        "--detection-method",
        choices=["openwakeword", "whisper"],
        default=None,
        help="Wake word detection method",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Detection threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Directory containing custom wake word models (default: ./models)",
    )
    parser.add_argument(
        "--debug-scores",
        action="store_true",
        help="Log all detection scores (even below threshold)",
    )
    
    args = parser.parse_args()
    
    # Load config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(script_dir, "config.yaml")
    config = load_config(config_path)
    
    # Command line args override config file
    host = args.host or config["server"]["host"]
    port = args.port or config["server"]["port"]
    wake_words = args.wake_words or config.get("wake_words") or None
    detection_method = args.detection_method
    if detection_method is None:
        dm_env = os.environ.get("WAKE_DETECTION_METHOD", "").strip().lower()
        if dm_env in ("openwakeword", "whisper"):
            detection_method = dm_env
    if detection_method is None:
        detection_method = config["detection"]["method"]
    threshold = args.threshold if args.threshold is not None else config["detection"]["threshold"]
    cooldown = config["detection"].get("cooldown_seconds", 2.0)
    models_dir = (
        args.models_dir
        or os.environ.get("WAKE_MODELS_DIR")
        or config.get("models_dir")
    )
    debug_scores = args.debug_scores or config["debug"].get("log_all_scores", False)
    min_score_to_log = config["debug"].get("min_score_to_log", 0.1)

    wake_words_env = os.environ.get("WAKE_WORDS", "").strip()
    if wake_words_env:
        wake_words = [w.strip() for w in wake_words_env.split(",") if w.strip()]
    
    # Resolve relative models_dir path
    if models_dir and not os.path.isabs(models_dir):
        models_dir = os.path.join(script_dir, models_dir)
    
    print(f"Configuration:")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Wake words: {wake_words or '(auto-discover from models)'}")
    print(f"  Detection method: {detection_method}")
    print(f"  Threshold: {threshold}")
    print(f"  Cooldown: {cooldown}s")
    print(f"  Models dir: {models_dir}")
    print(f"  Debug scores: {debug_scores}")
    
    server = MaraWakeWordServer(
        host=host,
        port=port,
        wake_words=wake_words,
        detection_method=detection_method,
        threshold=threshold,
        models_dir=models_dir,
        cooldown_seconds=cooldown,
        debug_scores=debug_scores,
        min_score_to_log=min_score_to_log,
    )
    
    # Register with Zeroconf
    server._register_zeroconf()
    
    try:
        await server.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if server.zeroconf:
            try:
                server.zeroconf.close()
            except Exception:
                pass  # Ignore errors during shutdown


if __name__ == "__main__":
    asyncio.run(main())

