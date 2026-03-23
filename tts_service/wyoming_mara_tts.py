#!/usr/bin/env python3
"""
Wyoming Protocol TTS Server for Mara
Integrates Piper TTS with optional RVC voice conversion
"""
import asyncio
import os
import sys
import socket
import numpy as np
import soundfile as sf
import tempfile
import subprocess
import wave
import math
from typing import Optional, Tuple

try:
    from zeroconf import ServiceInfo, Zeroconf
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False

try:
    from wyoming.server import AsyncServer, AsyncEventHandler
    from wyoming.tts import Synthesize
    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.info import Info, Describe, TtsVoice, TtsProgram
    from wyoming.ping import Ping, Pong
    from wyoming.error import Error
except ImportError:
    print("ERROR: Wyoming dependencies not installed.")
    print("Install with: pip install wyoming")
    sys.exit(1)

from piper.voice import PiperVoice
import glob
import re


def discover_voices(models_dir: str) -> dict:
    """
    Discover available Piper voices from a models directory.
    Returns a dict mapping short names to full model paths.
    
    Example: {'patrick': '/path/to/models/en_US-patrick-medium.onnx',
              'bobby': '/path/to/models/en_US-bobby-medium.onnx'}
    """
    voices = {}
    
    if not os.path.isdir(models_dir):
        print(f"Models directory not found: {models_dir}")
        return voices
    
    # Find all .onnx files that have matching .json config files
    for onnx_path in glob.glob(os.path.join(models_dir, "*.onnx")):
        json_path = onnx_path + ".json"
        if os.path.exists(json_path):
            # Extract voice name from filename: en_US-patrick-medium.onnx -> patrick
            filename = os.path.basename(onnx_path)
            # Pattern: {lang}_{region}-{name}-{quality}.onnx
            match = re.match(r'^[a-z]{2}_[A-Z]{2}-([a-zA-Z0-9_]+)-\w+\.onnx$', filename)
            if match:
                voice_name = match.group(1).lower()
                voices[voice_name] = onnx_path
                print(f"Discovered voice: {voice_name} -> {filename}")
            else:
                # Fallback: use filename without extension as voice name
                voice_name = filename.replace('.onnx', '').lower()
                voices[voice_name] = onnx_path
                print(f"Discovered voice (fallback): {voice_name} -> {filename}")
    
    return voices


class MaraTTSEventHandler(AsyncEventHandler):
    """Event handler for Wyoming TTS requests"""
    
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        voice_name: str,
        use_rvc: bool,
        rvc_dir: Optional[str],
        rvc_model: Optional[str],
        rvc_index: Optional[str],
        models_dir: Optional[str] = None,
    ):
        super().__init__(reader, writer)
        self.voice_name = voice_name
        self.use_rvc = use_rvc
        self.rvc_dir = rvc_dir
        self.rvc_model = rvc_model
        self.rvc_index = rvc_index
        self.voice: Optional[PiperVoice] = None
        self.models_dir = models_dir or os.path.join(os.path.dirname(__file__), "models")
        self.discovered_voices = discover_voices(self.models_dir)
    
    async def disconnect(self):
        """Handle client disconnection"""
        print("Client disconnected (this is normal when client closes connection)")
        await super().disconnect()
    
    async def handle_event(self, event):
        """Handle incoming Wyoming Protocol events"""
        try:
            # Check the event type field
            event_type = getattr(event, 'type', None)
            print(f"DEBUG: Received event type: {event_type}")
            
            # Handle ping events first (keepalive) - critical for connection stability
            if event_type == "ping":
                try:
                    # Always respond to ping, even if from_event fails
                    pong_event = Pong().event()
                    await self.write_event(pong_event)
                    print("✓ Responded to ping with pong")
                    return True
                except Exception as e:
                    print(f"ERROR: Failed to respond to ping: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            
            # Handle describe events
            if event_type == "describe" or event_type is None:
                try:
                    describe = Describe.from_event(event)
                    if describe is not None:
                        print("Received Describe event")
                        await self.handle_describe(describe)
                        return True
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"Failed to convert to Describe: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Handle synthesize events
            if event_type == "synthesize" or event_type is None:
                try:
                    synthesize = Synthesize.from_event(event)
                    if synthesize is not None:
                        text_preview = synthesize.text[:50] if synthesize.text else 'empty'
                        print(f"Received Synthesize event: {text_preview}")
                        await self.handle_synthesize(synthesize)
                        return True
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"Failed to convert to Synthesize: {e}")
                    import traceback
                    traceback.print_exc()
            
            # If we can't convert it, log what we got
            if event_type not in ["ping"]:  # Don't log ping events
                print(f"Unhandled event - type: {event_type}, class: {type(event).__name__}")
                if hasattr(event, 'data'):
                    print(f"Event data: {event.data}")
            return False
        except Exception as e:
            print(f"ERROR in handle_event: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def load_voice(self, voice_name: Optional[str] = None):
        """Load Piper voice model"""
        if voice_name is None:
            voice_name = self.voice_name
        
        # Only reload if voice changed
        current_voice_name = getattr(self, '_current_voice_name', None)
        if self.voice is None or voice_name != current_voice_name:
            try:
                print(f"Loading voice: {voice_name}")
                
                # First, check if this is a short name that we've discovered
                if voice_name in self.discovered_voices:
                    onnx_path = self.discovered_voices[voice_name]
                    print(f"Loading from models dir: {onnx_path}")
                    self.voice = PiperVoice.load(onnx_path)
                    self._current_voice_name = voice_name
                    print(f"Voice loaded successfully (sample rate: {self.voice.config.sample_rate} Hz)")
                    return
                
                # Try loading from models directory by full name
                models_onnx = os.path.join(self.models_dir, f"{voice_name}.onnx")
                if os.path.exists(models_onnx) and os.path.exists(models_onnx + ".json"):
                    print(f"Loading from models dir: {models_onnx}")
                    self.voice = PiperVoice.load(models_onnx)
                    self._current_voice_name = voice_name
                    print(f"Voice loaded successfully (sample rate: {self.voice.config.sample_rate} Hz)")
                    return
                
                # Try loading from current directory
                if os.path.exists(f"{voice_name}.onnx") and os.path.exists(f"{voice_name}.onnx.json"):
                    self.voice = PiperVoice.load(f"{voice_name}.onnx")
                    self._current_voice_name = voice_name
                    print(f"Voice loaded successfully (sample rate: {self.voice.config.sample_rate} Hz)")
                    return
                
                # Try loading by name (will search standard locations)
                self.voice = PiperVoice.load(voice_name)
                self._current_voice_name = voice_name
                print(f"Voice loaded successfully (sample rate: {self.voice.config.sample_rate} Hz)")
                
            except Exception as e:
                print(f"Voice not found: {e}")
                print(f"Available voices in models dir: {list(self.discovered_voices.keys())}")
                print(f"Models directory: {self.models_dir}")
                raise
    
    async def apply_rvc(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Apply RVC voice conversion"""
        if not self.use_rvc or not os.path.exists(self.rvc_dir):
            return audio, sr
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_in:
            in_wav = f_in.name
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_out:
            out_wav = f_out.name
        
        try:
            sf.write(in_wav, audio, sr)
            python_path = os.path.join(self.rvc_dir, ".venv/bin/python")
            convert_script = os.path.join(self.rvc_dir, "convert_rvc.py")
            
            if not os.path.exists(python_path) or not os.path.exists(convert_script):
                return audio, sr
            
            cmd = [python_path, convert_script, "--model", self.rvc_model,
                   "--in", in_wav, "--out", out_wav, "--f0", "harvest",
                   "--key", "0", "--device", "mps"]
            if os.path.exists(self.rvc_index):
                cmd += ["--index", self.rvc_index]
            
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc.wait()
            
            if proc.returncode == 0:
                data, sr2 = sf.read(out_wav, dtype='float32')
                return data, sr2
            return audio, sr
        except Exception as e:
            print(f"RVC error: {e}")
            return audio, sr
        finally:
            for f in [in_wav, out_wav]:
                try:
                    if os.path.exists(f):
                        os.unlink(f)
                except:
                    pass
    
    async def handle_synthesize(self, msg: Synthesize) -> None:
        """Handle TTS synthesis request"""
        text = msg.text.strip()
        if not text:
            return
        
        # Refresh discovered voices (hot-reload support)
        self.discovered_voices = discover_voices(self.models_dir)
        
        # Check if a specific voice was requested
        requested_voice = self.voice_name  # Default to configured voice
        if msg.voice and msg.voice.name:
            voice_name = msg.voice.name
            # Check if it's a discovered voice (short name like "patrick", "bobby")
            if voice_name in self.discovered_voices:
                requested_voice = voice_name  # Use short name, load_voice will resolve it
            elif voice_name.endswith("-rvc") and voice_name[:-4] in self.discovered_voices:
                # Handle RVC variant - use the base voice
                requested_voice = voice_name[:-4]
            else:
                # Use as-is (could be a full voice name like "en_US-patrick-medium")
                requested_voice = voice_name
            print(f"Voice requested: {msg.voice.name} -> {requested_voice}")
        
        # Load the requested voice (or default)
        await self.load_voice(requested_voice)
        print(f"Synthesizing: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        try:
            # Use temporary WAV file approach (matches official implementation)
            with tempfile.NamedTemporaryFile(mode="wb+", suffix=".wav", delete=False) as output_file:
                output_path = output_file.name
            
            try:
                # Synthesize to WAV file
                wav_writer = wave.open(output_path, "wb")
                with wav_writer:
                    if hasattr(self.voice, 'synthesize_wav'):
                        # Use synthesize_wav if available (official API)
                        from piper import SynthesisConfig
                        syn_config = SynthesisConfig()
                        self.voice.synthesize_wav(text, wav_writer, syn_config)
                    else:
                        # Fallback: use speak_text and write WAV manually
                        audio = self.voice.speak_text(text)
                        sr = self.voice.config.sample_rate
                        
                        # Apply RVC if enabled
                        if self.use_rvc:
                            audio, sr = await self.apply_rvc(audio, sr)
                        
                        # Write as WAV
                        wav_writer.setnchannels(1)  # Mono
                        wav_writer.setsampwidth(2)  # 16-bit
                        wav_writer.setframerate(sr)
                        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                        wav_writer.writeframes(audio_int16.tobytes())
                
                # Read WAV file and send chunks
                wav_file = wave.open(output_path, "rb")
                with wav_file:
                    rate = wav_file.getframerate()
                    width = wav_file.getsampwidth()
                    channels = wav_file.getnchannels()
                    
                    await self.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
                    
                    # Read all audio frames
                    audio_bytes = wav_file.readframes(wav_file.getnframes())
                    
                    # Send in chunks (1024 samples per chunk, matching official)
                    bytes_per_sample = width * channels
                    samples_per_chunk = 1024
                    bytes_per_chunk = bytes_per_sample * samples_per_chunk
                    num_chunks = int(math.ceil(len(audio_bytes) / bytes_per_chunk))
                    
                    for i in range(num_chunks):
                        offset = i * bytes_per_chunk
                        chunk = audio_bytes[offset : offset + bytes_per_chunk]
                        await self.write_event(
                            AudioChunk(
                                audio=chunk,
                                rate=rate,
                                width=width,
                                channels=channels,
                            ).event()
                        )
                
                await self.write_event(AudioStop().event())
                print("Synthesis complete")
            finally:
                # Clean up temp file
                try:
                    if os.path.exists(output_path):
                        os.unlink(output_path)
                except:
                    pass
        except Exception as e:
            print(f"ERROR during synthesis: {e}")
            import traceback
            traceback.print_exc()
            try:
                # Send error event to client
                await self.write_event(Error(text=str(e), code=e.__class__.__name__).event())
                await self.write_event(AudioStop().event())
            except Exception as e2:
                print(f"ERROR sending error event: {e2}")
    
    async def handle_describe(self, msg: Describe) -> None:
        """Handle info/describe request"""
        from wyoming.info import Attribution, TtsProgram
        
        print("Handling Describe request...")
        
        # Re-discover voices in case new models were added
        self.discovered_voices = discover_voices(self.models_dir)
        
        # Create voice list dynamically from discovered voices
        voices = []
        
        for voice_name in sorted(self.discovered_voices.keys()):
            voices.append(TtsVoice(
                name=voice_name,
                description=f"{voice_name.title()} (Standard)",
                attribution=Attribution(name="Piper TTS", url="https://github.com/rhasspy/piper"),
                installed=True,
                version="1.0.0",
                languages=["en", "en-US"]  # Support both language codes
            ))
            
            # Add RVC variant if RVC is enabled
            if self.use_rvc:
                voices.append(TtsVoice(
                    name=f"{voice_name}-rvc",
                    description=f"{voice_name.title()} (RVC Enhanced)",
                    attribution=Attribution(name="Piper TTS + RVC", url="https://github.com/rhasspy/piper"),
                    installed=True,
                    version="1.0.0",
                    languages=["en", "en-US"]
                ))
        
        print(f"Discovered {len(self.discovered_voices)} voice(s): {list(self.discovered_voices.keys())}")
        
        try:
            # Wrap voices in TtsProgram (required by Wyoming Protocol)
            tts_program = TtsProgram(
                name="mara-piper",
                description="Mara TTS with Piper and optional RVC",
                attribution=Attribution(name="Mara", url="https://github.com/rhasspy/piper"),
                installed=True,
                voices=voices,
                version="1.0.0",
                supports_synthesize_streaming=False,
            )
            
            info = Info(tts=[tts_program])
            event = info.event()
            print(f"Sending Info event with {len(voices)} voice(s) in TtsProgram")
            await self.write_event(event)
            print("Info event sent successfully")
        except Exception as e:
            print(f"ERROR sending Info event: {e}")
            import traceback
            traceback.print_exc()


class MaraTTSServer:
    """Wyoming Protocol TTS server with Piper and RVC support"""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 10200,
        voice_name: Optional[str] = None,
        use_rvc: bool = False,
        rvc_dir: Optional[str] = None,
        rvc_model: Optional[str] = None,
        rvc_index: Optional[str] = None,
        models_dir: Optional[str] = None,
    ):
        uri = f"tcp://{host}:{port}"
        self.server = AsyncServer.from_uri(uri)
        self._port = port  # Store port for easy access
        
        # Models directory (defaults to ./models relative to script)
        self.models_dir = models_dir or os.getenv("MODELS_DIR", os.path.join(os.path.dirname(__file__), "models"))
        
        # Discover available voices
        self.discovered_voices = discover_voices(self.models_dir)
        
        # Voice configuration - default to first discovered voice or fallback
        if voice_name:
            self.voice_name = voice_name
        elif os.getenv("PIPER_VOICE"):
            self.voice_name = os.getenv("PIPER_VOICE")
        elif self.discovered_voices:
            # Use first discovered voice as default
            self.voice_name = list(self.discovered_voices.keys())[0]
        else:
            self.voice_name = "en_US-patrick-medium"
        
        self.use_rvc = use_rvc or (os.getenv("USE_RVC", "false").lower() == "true")
        self.rvc_dir = rvc_dir or os.getenv("RVC_REPO", "/Users/you/Desktop/rvc")
        self.rvc_model = rvc_model or os.getenv(
            "RVC_MODEL",
            "/Users/isaacshvartsman/Downloads/Yajirobe-DBSparkingZero/Yajirobe-DBSparkingZero_160e_2720s.pth"
        )
        self.rvc_index = rvc_index or os.getenv(
            "RVC_INDEX",
            "/Users/isaacshvartsman/Downloads/Yajirobe-DBSparkingZero/Yajirobe-DBSparkingZero.index"
        )
        
        print(f"Initializing Mara TTS Server on {host}:{port}")
        print(f"Models directory: {self.models_dir}")
        print(f"Discovered voices: {list(self.discovered_voices.keys())}")
        print(f"Default voice: {self.voice_name}")
        print(f"RVC: {'Enabled' if self.use_rvc else 'Disabled'}")
    
    def _create_handler(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> MaraTTSEventHandler:
        """Create a new event handler for a client connection"""
        client_addr = writer.get_extra_info('peername')
        print(f"New client connected from: {client_addr}")
        return MaraTTSEventHandler(
            reader,
            writer,
            self.voice_name,
            self.use_rvc,
            self.rvc_dir,
            self.rvc_model,
            self.rvc_index,
            self.models_dir,
        )
    
    async def run(self):
        """Run the Wyoming Protocol server"""
        # Start Zeroconf advertising for auto-discovery (in background thread)
        zeroconf = None
        service_info = None
        if ZEROCONF_AVAILABLE:
            try:
                import threading
                
                def register_zeroconf():
                    """Register Zeroconf service in a separate thread"""
                    try:
                        zc = Zeroconf()
                        # Get all IP addresses for advertising
                        ips = self._get_all_ips()
                        addresses = [socket.inet_aton(ip) for ip in ips]
                        
                        info = ServiceInfo(
                            "_wyoming._tcp.local.",
                            "Mara TTS._wyoming._tcp.local.",
                            addresses=addresses,
                            port=self._get_port(),
                            properties={"version": "1.0"},
                        )
                        zc.register_service(info)
                        return zc, info
                    except Exception as e:
                        print(f"WARNING: Zeroconf registration failed: {e}")
                        return None, None
                
                # Run Zeroconf registration in a separate thread
                zc_result = [None, None]
                def zc_thread():
                    zc_result[0], zc_result[1] = register_zeroconf()
                
                thread = threading.Thread(target=zc_thread, daemon=True)
                thread.start()
                thread.join(timeout=2)  # Wait up to 2 seconds
                
                zeroconf, service_info = zc_result
                
                if zeroconf:
                    ips = self._get_all_ips()
                    print(f"\n{'='*60}")
                    print(f"Mara TTS Server running (auto-discovery enabled)")
                    print(f"Advertising on IPs: {', '.join(ips)}")
                    print(f"Port: {self._get_port()}")
                    print(f"Manual address: tcp://{self._get_local_ip()}:{self._get_port()}")
                    print(f"{'='*60}\n")
                else:
                    print(f"\n{'='*60}")
                    print(f"Mara TTS Server running (auto-discovery failed)")
                    print(f"Manual address: tcp://{self._get_local_ip()}:{self._get_port()}")
                    print(f"{'='*60}\n")
            except Exception as e:
                print(f"WARNING: Zeroconf setup failed: {e}")
                print(f"Server running on: tcp://{self._get_local_ip()}:{self._get_port()}")
        else:
            print(f"\n{'='*60}")
            print(f"Mara TTS Server running (auto-discovery disabled)")
            print(f"Manual address: tcp://{self._get_local_ip()}:{self._get_port()}")
            print(f"{'='*60}\n")
        
        try:
            await self.server.run(self._create_handler)
        finally:
            if zeroconf and service_info:
                try:
                    def unregister():
                        zeroconf.unregister_service(service_info)
                        zeroconf.close()
                    threading.Thread(target=unregister, daemon=True).start()
                except:
                    pass
    
    def _get_local_ip(self) -> str:
        """Get local IP address for display"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def _get_all_ips(self) -> list:
        """Get all local IP addresses for Zeroconf"""
        ips = []
        try:
            import ifaddr
            adapters = ifaddr.get_adapters()
            for adapter in adapters:
                for ip in adapter.ips:
                    if ip.is_IPv4 and not ip.ip.startswith("127."):
                        ips.append(ip.ip)
        except:
            # Fallback: use the detected IP
            ip = self._get_local_ip()
            if ip != "127.0.0.1":
                ips.append(ip)
        return ips if ips else [self._get_local_ip()]
    
    def _get_port(self) -> int:
        """Get server port"""
        # Extract port from the URI we created
        # We stored it in __init__, so let's track it
        if hasattr(self, '_port'):
            return self._port
        # Fallback: try to extract from server object
        uri_str = str(self.server)
        if ":" in uri_str:
            try:
                port = int(uri_str.split(":")[-1].split(">")[0])
                return port
            except:
                pass
        return 10200


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mara TTS Wyoming Protocol Server")
    parser.add_argument(
        "--host",
        default=os.getenv("WYOMING_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("WYOMING_PORT", "10200")),
        help="Port to bind to (default: 10200)"
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Piper voice name (overrides PIPER_VOICE env var)"
    )
    parser.add_argument(
        "--rvc",
        action="store_true",
        help="Enable RVC voice conversion"
    )
    parser.add_argument(
        "--rvc-dir",
        default=None,
        help="RVC repository directory"
    )
    parser.add_argument(
        "--rvc-model",
        default=None,
        help="RVC model path"
    )
    parser.add_argument(
        "--rvc-index",
        default=None,
        help="RVC index path"
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Directory containing Piper voice models (default: ./models)"
    )
    
    args = parser.parse_args()
    
    server = MaraTTSServer(
        host=args.host,
        port=args.port,
        voice_name=args.voice,
        use_rvc=args.rvc,
        rvc_dir=args.rvc_dir,
        rvc_model=args.rvc_model,
        rvc_index=args.rvc_index,
        models_dir=args.models_dir,
    )
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

