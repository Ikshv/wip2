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
import re
import string
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
    from wyoming.audio import AudioChunk, AudioStart, AudioStop, AudioChunkConverter
    from wyoming.info import Info, Describe, WakeModel, WakeProgram, Attribution
    from wyoming.wake import Detect, Detection
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


WHISPER_TARGET_RATE = 16000


def _resample_linear_f32(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resample mono float32 PCM (no scipy dependency)."""
    if src_rate == dst_rate or len(samples) < 2:
        return samples.astype(np.float32, copy=False)
    new_len = max(1, int(len(samples) * dst_rate / src_rate))
    xp = np.linspace(0.0, len(samples) - 1, num=new_len, dtype=np.float64)
    y = np.interp(xp, np.arange(len(samples), dtype=np.float64), samples.astype(np.float64))
    return y.astype(np.float32)


def _normalize_transcript(s: str) -> str:
    s = s.lower()
    for ch in string.punctuation + "¿¡":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def _float32_to_int16_pcm(samples: np.ndarray) -> np.ndarray:
    """openWakeWord expects 16 kHz mono PCM as int16."""
    s = np.clip(samples.astype(np.float64, copy=False), -1.0, 1.0)
    return (s * 32767.0).astype(np.int16)


def _transcript_matches_wake_word(norm_transcript: str, wake_word: str) -> bool:
    """Match wake phrase in Whisper output (whole words for single-token names)."""
    nw = _normalize_transcript(wake_word)
    if not nw:
        return False
    if " " in nw:
        return nw in norm_transcript
    return re.search(rf"\b{re.escape(nw)}\b", norm_transcript) is not None


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
        # All audio is normalized to WHISPER_TARGET_RATE mono before buffering.
        self.audio_buffer = deque(maxlen=WHISPER_TARGET_RATE * 5)  # 5 seconds at 16 kHz
        self.sample_rate = WHISPER_TARGET_RATE
        self._pcm16_mono_16k = AudioChunkConverter(
            rate=WHISPER_TARGET_RATE, channels=1, width=2
        )
        self._debug_whisper = os.environ.get("WAKE_DEBUG_WHISPER", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self._debug_oww = os.environ.get("WAKE_DEBUG_OWW", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self._oww_debug_last_log = 0.0
        self._last_detection_time = 0.0
        self._cooldown_seconds = cooldown_seconds
        # After a hit, require scores to fall below this before another (stops one utterance
        # from firing on every chunk while openWakeWord stays above threshold).
        rms = os.environ.get("WAKE_REARM_MAX_SCORE", "").strip()
        if rms:
            try:
                self._oww_rearm_max_score = float(rms)
            except ValueError:
                self._oww_rearm_max_score = max(0.08, float(threshold) * 0.35)
        else:
            self._oww_rearm_max_score = max(0.08, float(threshold) * 0.35)
        self._oww_rearmed = True
        self._debug_scores = debug_scores
        self._min_score_to_log = min_score_to_log
        # Home Assistant sends Detect before AudioStart; must return True or HA disconnects.
        self._detect_names: Optional[List[str]] = None
        self.models_dir = models_dir or os.path.join(os.path.dirname(__file__), "models")
        self.custom_models = custom_models or {}
        
        # Initialize detection models
        self._initialize_models()

    def _detection_name_allowed(self, name: str) -> bool:
        """HA may restrict which wake word id to report (Detect names=…)."""
        if not self._detect_names:
            return True
        n = (name or "").strip().lower()
        return any(n == d.strip().lower() for d in self._detect_names if d)

    def _reset_openwakeword_models(self) -> None:
        """Reset streaming preprocessor state (call on new audio stream)."""
        for mdl in self.models.values():
            try:
                mdl.reset()
            except Exception as e:
                print(f"WARNING: openWakeWord reset: {e}")

    def _prediction_scores(self, prediction: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, score in prediction.items():
            arr = np.asarray(score, dtype=np.float64)
            out[name] = float(arr.flat[0]) if arr.size else 0.0
        return out

    async def _process_openwakeword_chunk(self, x_int16: np.ndarray) -> None:
        """Feed one chunk of 16 kHz mono int16 PCM; openWakeWord must not be reset each call."""
        import time

        if x_int16.size == 0:
            return
        x_int16 = np.ascontiguousarray(x_int16)

        current_time = time.time()

        try:
            custom = self.models.get("custom")
            default = self.models.get("default")
            global_max = 0.0
            best: Optional[tuple] = None  # (label, model_name, score)

            for label, mdl in (("custom", custom), ("default", default)):
                if mdl is None:
                    continue
                raw = mdl.predict(x_int16)
                scores = self._prediction_scores(raw)
                if scores:
                    mx = max(scores.values())
                    if mx > global_max:
                        global_max = mx
                if self._debug_scores or self._debug_oww:
                    mx = max(scores.values()) if scores else 0.0
                    if self._debug_scores and mx >= self._min_score_to_log:
                        print(f"[wake/oww:{label}] {scores} (threshold {self.threshold})")
                    elif self._debug_oww and (
                        mx >= 0.02 or current_time - self._oww_debug_last_log >= 1.0
                    ):
                        print(f"[wake/oww:{label}] {scores}")
                        self._oww_debug_last_log = current_time

                for model_name, score in scores.items():
                    if score > self.threshold and (
                        best is None or score > best[2]
                    ):
                        best = (label, model_name, score)

            if not self._oww_rearmed:
                if global_max < self._oww_rearm_max_score:
                    self._oww_rearmed = True
                return

            if best is None:
                return
            if current_time - self._last_detection_time < self._cooldown_seconds:
                return

            label, model_name, score = best
            detected = self._map_model_to_wake_word(model_name) or (
                self.wake_words[0] if self.wake_words else model_name
            )
            if not self._detection_name_allowed(detected):
                return
            print(
                f"Wake word detected: {detected} "
                f"(openWakeWord {label}/{model_name}, score={score:.3f})"
            )
            await self.write_event(
                Detection(
                    name=detected,
                    timestamp=int(current_time * 1000),
                    context={"model": model_name, "score": score},
                ).event()
            )
            self._last_detection_time = current_time
            self._oww_rearmed = False
        except Exception as e:
            print(f"ERROR in openWakeWord predict: {e}")
            import traceback

            traceback.print_exc()
    
    def _initialize_models(self):
        """Initialize wake word detection models"""
        if self.detection_method == "openwakeword" and OPENWAKEWORD_AVAILABLE:
            try:
                custom_model_paths = list(self.custom_models.values())
                if custom_model_paths:
                    try:
                        self.models["custom"] = Model(
                            wakeword_models=custom_model_paths,
                            inference_framework="onnx",
                        )
                        print(
                            "openWakeWord (ONNX): loaded custom wake model(s): "
                            f"{[os.path.basename(p) for p in custom_model_paths]}"
                        )
                    except Exception as e:
                        print(f"WARNING: Failed to load custom ONNX wake models: {e}")
                if not self.models:
                    try:
                        self.models["default"] = Model(inference_framework="onnx")
                        print(
                            "openWakeWord (ONNX): loaded default pretrained wake models "
                            "(no custom .onnx loaded from models directory)"
                        )
                    except Exception as e:
                        print(f"WARNING: Failed to load default openWakeWord models: {e}")
                        if not self.models:
                            print("Falling back to Whisper-based detection")
                            self.detection_method = "whisper" if WHISPER_AVAILABLE else None
            except Exception as e:
                print(f"ERROR initializing openWakeWord: {e}")
                self.models.clear()
                print("Falling back to Whisper-based detection")
                self.detection_method = "whisper" if WHISPER_AVAILABLE else None

        if self.detection_method == "whisper" and WHISPER_AVAILABLE:
            try:
                w_model = os.environ.get("WHISPER_MODEL", "base.en")
                w_device = os.environ.get("WHISPER_DEVICE", "cpu")
                w_compute = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
                self.whisper_model = WhisperModel(
                    w_model, device=w_device, compute_type=w_compute
                )
                print(
                    f"Loaded Whisper model for wake word detection: {w_model} "
                    f"({w_device}, {w_compute})"
                )
            except Exception as e:
                print(f"ERROR loading Whisper model: {e}")
                self.detection_method = None
                self.whisper_model = None

        if self.detection_method == "openwakeword" and not self.models:
            print("openWakeWord selected but no models loaded; trying Whisper…")
            self.detection_method = "whisper" if WHISPER_AVAILABLE else None
            if self.detection_method == "whisper":
                try:
                    w_model = os.environ.get("WHISPER_MODEL", "base.en")
                    w_device = os.environ.get("WHISPER_DEVICE", "cpu")
                    w_compute = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
                    self.whisper_model = WhisperModel(
                        w_model, device=w_device, compute_type=w_compute
                    )
                    print(
                        f"Loaded Whisper model for wake word detection: {w_model} "
                        f"({w_device}, {w_compute})"
                    )
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
            
            # Home Assistant wake pipeline: Detect → AudioStart → AudioChunk+ (see HA wyoming/wake_word.py)
            if event_type == "detect":
                try:
                    det = Detect.from_event(event)
                    self._detect_names = det.names if det else None
                    print(f"Received Detect (names={self._detect_names!r})")
                    return True
                except Exception as e:
                    print(f"WARNING: Failed to parse Detect event: {e}")
                    return True

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
                        # Fresh rate-conversion state for a new stream
                        self._pcm16_mono_16k._ratecv_state = None
                        if self.detection_method == "openwakeword" and self.models:
                            self._reset_openwakeword_models()
                            self._oww_rearmed = True
                        print(
                            f"Audio stream started: {audio_start.rate} Hz, "
                            f"{audio_start.width} byte samples, {audio_start.channels} ch "
                            f"(normalized to {WHISPER_TARGET_RATE} Hz mono for detection)"
                        )
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
                    if self.detection_method == "openwakeword" and self.models:
                        self._reset_openwakeword_models()
                        self._oww_rearmed = True
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
            w = audio_chunk.width
            ch = audio_chunk.channels
            rate = audio_chunk.rate

            if w == 2:
                # int16 PCM: resample to 16 kHz mono via Wyoming (handles stereo).
                converted = self._pcm16_mono_16k.convert(audio_chunk)
                if not converted.audio:
                    return
                audio_data = (
                    np.frombuffer(converted.audio, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            elif w == 4:
                # 32-bit float (some pipelines)
                raw = np.frombuffer(audio_chunk.audio, dtype=np.float32)
                if ch == 2:
                    raw = raw.reshape(-1, 2).mean(axis=1)
                elif ch != 1:
                    print(f"Unsupported channel count: {ch}")
                    return
                if rate != WHISPER_TARGET_RATE:
                    audio_data = _resample_linear_f32(raw, rate, WHISPER_TARGET_RATE)
                else:
                    audio_data = raw.astype(np.float32, copy=False)
            else:
                print(f"Unsupported audio width: {w}")
                return

            if self.detection_method == "openwakeword" and self.models:
                x16 = _float32_to_int16_pcm(
                    np.ascontiguousarray(audio_data, dtype=np.float32)
                )
                await self._process_openwakeword_chunk(x16)
            elif self.detection_method == "whisper":
                self.audio_buffer.extend(audio_data)
                if len(self.audio_buffer) >= WHISPER_TARGET_RATE:
                    await self._check_wake_words_whisper()
            else:
                self.audio_buffer.extend(audio_data)
        
        except Exception as e:
            print(f"ERROR processing audio chunk: {e}")
            import traceback
            traceback.print_exc()
    
    async def _check_wake_words_whisper(self):
        """Whisper-only: scan recent buffered float32 audio."""
        import time

        current_time = time.time()
        if current_time - self._last_detection_time < self._cooldown_seconds:
            return

        buffer_array = np.ascontiguousarray(
            np.array(
                list(self.audio_buffer)[-WHISPER_TARGET_RATE * 2 :], dtype=np.float32
            )
        )

        if len(buffer_array) < WHISPER_TARGET_RATE * 0.5:
            return

        detected = False

        if self.detection_method == "whisper" and self.whisper_model:
            # Use Whisper for detection (fallback method)
            try:
                hotwords = ", ".join(self.wake_words[:8]) if self.wake_words else None
                segments, info = self.whisper_model.transcribe(
                    buffer_array,
                    beam_size=1,
                    language="en",
                    condition_on_previous_text=False,
                    vad_filter=False,
                    without_timestamps=True,
                    hotwords=hotwords,
                )
                text_raw = " ".join(segment.text for segment in segments)
                norm_text = _normalize_transcript(text_raw)
                if self._debug_whisper and norm_text:
                    print(f"[wake/whisper] heard: {text_raw!r} -> normalized: {norm_text!r}")

                for wake_word in self.wake_words:
                    if _transcript_matches_wake_word(norm_text, wake_word):
                        if not self._detection_name_allowed(wake_word):
                            continue
                        print(
                            f"Wake word detected via Whisper: {wake_word} "
                            f"(text: {text_raw!r})"
                        )
                        await self.write_event(
                            Detection(
                                name=wake_word,
                                timestamp=int(current_time * 1000),
                                context={
                                    "model": "whisper",
                                    "score": 0.8,
                                },
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
        self._zeroconf_service_info: Optional[Any] = None
        
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

    def _zeroconf_announce_ips(self) -> List[str]:
        """
        IPs advertised in mDNS A records. Home Assistant uses these to open tcp://…:port.

        In Docker, container interfaces are often only 172.x; set WYOMING_ZEROCONF_IPS to your
        host's LAN address (comma-separated) so phones / other HA instances get a routable IP.
        """
        raw = os.environ.get("WYOMING_ZEROCONF_IPS", "").strip()
        if raw:
            out: List[str] = []
            for part in raw.split(","):
                p = part.strip()
                if not p:
                    continue
                try:
                    socket.inet_aton(p)
                    out.append(p)
                except OSError:
                    print(f"WARNING: Ignoring invalid WYOMING_ZEROCONF_IPS entry: {p!r}")
            if out:
                return out
            print("WARNING: WYOMING_ZEROCONF_IPS empty after parse; using auto-detected IPs")
        return self._get_all_ips()

    def _register_zeroconf(self):
        """Register _wyoming._tcp.local. the same way other Wyoming servers do (Home Assistant discovery)."""
        if not ZEROCONF_AVAILABLE:
            print("WARNING: Zeroconf not available, skipping service registration")
            return

        instance = os.environ.get("WYOMING_ZEROCONF_NAME", "Mara Wake Word").strip() or "Mara Wake Word"
        service_type = "_wyoming._tcp.local."
        # One stable DNS-SD name; all interface IPs in a single ServiceInfo (matches Mara TTS + HA manifest).
        full_name = f"{instance}.{service_type}"

        zc_result: List[Any] = [None, None]

        def register_in_thread():
            try:
                zc = Zeroconf()
                ips = self._zeroconf_announce_ips()
                addresses = [socket.inet_aton(ip) for ip in ips]
                info = ServiceInfo(
                    service_type,
                    full_name,
                    addresses=addresses,
                    port=self._port,
                    properties={"version": "1.0"},
                )
                zc.register_service(info)
                zc_result[0] = zc
                zc_result[1] = info
                print(
                    f"Zeroconf Wyoming: {full_name} port {self._port} "
                    f"addresses={[socket.inet_ntoa(a) for a in addresses]}"
                )
            except Exception as e:
                print(f"WARNING: Zeroconf registration failed: {e}")

        thread = threading.Thread(target=register_in_thread, daemon=True)
        thread.start()
        thread.join(timeout=3.0)
        self.zeroconf = zc_result[0]
        self._zeroconf_service_info = zc_result[1]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    default_config = {
        "server": {"host": "0.0.0.0", "port": 10300},
        "detection": {
            "method": "openwakeword",
            "threshold": 0.2,
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
    tw_env = os.environ.get("WAKE_THRESHOLD", "").strip()
    if tw_env:
        try:
            threshold = float(tw_env)
        except ValueError:
            print(f"WARNING: Ignoring invalid WAKE_THRESHOLD={tw_env!r}")
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
                if server._zeroconf_service_info is not None:
                    server.zeroconf.unregister_service(server._zeroconf_service_info)
                server.zeroconf.close()
            except Exception:
                pass  # Ignore errors during shutdown


if __name__ == "__main__":
    asyncio.run(main())

