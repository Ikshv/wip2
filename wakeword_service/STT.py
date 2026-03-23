#!/usr/bin/env python3
# Wake word via STT (no dedicated model):
# - Continuously transcribe short audio windows with Whisper tiny.en
# - If window contains a wake phrase ("hey homey", "hi homey", "ok/okay homey", "oh homey"),
#   play a beep, then capture the command hands-free (VAD), and transcribe with small.en.

import os  # <-- NEW: set env before importing llm
# ----- Ensure model + context persistence are set BEFORE importing llm -----
os.environ.setdefault("MARA_CONTEXT_FILE", os.path.expanduser("~/.mara_ctx.json"))
os.environ.setdefault("MARA_MODEL", "llama2-uncensored:latest")
os.environ.setdefault("MARA_KEEP_ALIVE", "5m")
os.environ.setdefault("MARA_SESSION", "mic-session-1")  # default session

import argparse, time, queue, sys, math, re
from collections import deque

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

from llm import reply as llm_reply                    # <-- imports AFTER env is set
from TTS import say, say_with_rvc

# ----------------------------- Config -----------------------------
RATE = 16000
CHANNELS = 1
BLOCK_MS = 20                                # 10/20/30ms valid for VAD
BLOCK_FRAMES = int(RATE * BLOCK_MS / 1000)

# Wake detection window/step (balance CPU vs latency)
WAKE_WINDOW_S = 1.4                          # analyze ~1.4s window
WAKE_STEP_S   = 0.40                         # re-run STT every 400 ms

# VAD settings
VAD_AGGR_WAKE = 2                            # waiting for wake
VAD_AGGR_CMD  = 1                            # capturing the command
SILENCE_TAIL_MS = 1200                       # end utterance after this much silence

# Gain/normalization
TARGET_RMS = 0.05                            # ~-26 dBFS
MAX_GAIN   = 6.0

# Cooldowns
WAKE_COOLDOWN_S    = 2.0
WAKE_REARM_DELAY_S = 0.6                     # short pause after wake before capture

# Whisper models
ASR_MODEL_WAKE = "tiny.en"                   # lightweight for wake detection
ASR_MODEL_CMD  = "small.en"                  # quality for commands

INITIAL_PROMPT = (
    "This is a home assistant named homey for smart home control, IoT, lighting, climate, and reminders."
)

# Wake phrases (include variants)
WAKE_PHRASES = [
    "evil house"
]

# ===== Voice interrupt config =====
INTERRUPT_PHRASES = ["stop"]

def contains_interrupt(text: str) -> bool:
    t = normalize_text(text)
    return any(normalize_text(p) in t for p in INTERRUPT_PHRASES)

# Use the same session id across this mic session so memory persists
SESSION_ID = os.getenv("MARA_SESSION", "mic-session-1")  # <-- NEW

# --- Async audio playback (so we can listen while talking) ---
import threading
import sounddevice as sd

def tts_play_async_pcm(audio_float32: np.ndarray, sr: int):
    """Play PCM audio in a background thread. Returns stop() function."""
    def _run():
        try:
            sd.play(audio_float32, sr)
            sd.wait()
        except Exception:
            pass
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    def stop():
        try:
            sd.stop()
        except Exception:
            pass
    return stop

def synthesize_tts(text: str):
    """
    Try Piper (Python API) to get PCM we can interrupt.
    If Piper is unavailable, fall back to macOS 'say' subprocess (also interruptible).
    Returns tuple (mode, handle):
        - ("pcm", (audio_float32, sample_rate))  -> play with tts_play_async_pcm
        - ("proc", subprocess.Popen)             -> kill on interrupt
    """
    if not text:
        return None
    # Try Piper first (best quality, full control)
    try:
        from piper.voice import PiperVoice
        voice_name = os.getenv("PIPER_VOICE", "en_US-amy-medium")
        try:
            voice = PiperVoice.load(voice_name)
        except Exception:
            PiperVoice.download_voice(voice_name)
            voice = PiperVoice.load(voice_name)
        audio = voice.speak_text(text)  # float32 PCM
        return ("pcm", (audio, voice.config.sample_rate))
    except Exception:
        pass

    # Fallback: macOS 'say' (spawn a process we can terminate)
    try:
        import subprocess
        proc = subprocess.Popen(["say", text])
        return ("proc", proc)
    except Exception:
        return None

# ------------------------- Utility DSP ---------------------------
def agc_preprocess(float32_audio, target_rms=TARGET_RMS, max_gain=MAX_GAIN):
    rms = max(1e-8, max(1e-8, float((float32_audio**2).mean())**0.5))
    gain = min(max_gain, max(1.0, target_rms / rms))
    x = float32_audio * gain
    return np.tanh(x)  # soft clip

def int16_to_float32(b):
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0

def chunk_bytes_to_float32(chunks):
    return int16_to_float32(b"".join(chunks))

def now_ms():
    return int(time.time() * 1000)

# ---------------------- Keyword matching -------------------------
def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def contains_wake_phrase(text: str, targets=WAKE_PHRASES) -> bool:
    t = normalize_text(text)
    for p in targets:
        p2 = normalize_text(p)
        if p2 and p2 in t:
            return True
    # tiny fuzzy tolerance around "homey"
    if "homey" in t or re.search(r"\bma?ra\b", t):
        return True
    return False

# ----------------------------- Beep -------------------------------
def beep():
    try:
        dur = 0.12
        f = 880.0
        t = np.linspace(0, dur, int(RATE * dur), False)
        tone = 0.15 * np.sin(2 * np.pi * f * t)
        sd.play(tone.astype(np.float32), RATE)
        sd.wait()
    except Exception:
        pass

# ---------------------- Command capture + ASR ---------------------
def capture_and_transcribe_command(q, vad_cmd, asr_cmd, preroll_deque):
    """Record until silence, then transcribe with high-quality model."""
    # include a bit of pre-roll so we don't cut first word
    buf = list(preroll_deque)
    last_voice = now_ms()

    while True:
        try:
            b = q.get(timeout=0.25)
        except queue.Empty:
            if now_ms() - last_voice > SILENCE_TAIL_MS:
                break
            else:
                continue

        buf.append(b)
        if vad_cmd.is_speech(b, RATE):
            last_voice = now_ms()
        if now_ms() - last_voice > SILENCE_TAIL_MS:
            break

    audio = chunk_bytes_to_float32(buf)
    audio = agc_preprocess(audio)
    print("🧠 Transcribing…")
    segs, info = asr_cmd.transcribe(
        audio,
        language="en",
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=[0.0, 0.2, 0.4],
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.65,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        # initial_prompt=INITIAL_PROMPT,
    )
    return "".join(s.text for s in segs).strip()

# --------------------------- Main App ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, help="Input device index (see sounddevice.query_devices())")
    args = ap.parse_args()

    if args.device is not None:
        sd.default.device = (args.device, None)

    print("Loading Whisper models…")
    asr_wake = WhisperModel(ASR_MODEL_WAKE, compute_type="auto")
    asr_cmd  = WhisperModel(ASR_MODEL_CMD,  compute_type="auto")

    vad_wake = webrtcvad.Vad(VAD_AGGR_WAKE)
    vad_cmd  = webrtcvad.Vad(VAD_AGGR_CMD)

    q = queue.Queue()
    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    print("Mic stream starting… (say 'Hey homey')")
    with sd.RawInputStream(
        samplerate=RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=BLOCK_FRAMES,
        callback=callback,
    ):
        # Ring buffers for wake detection and pre-roll
        from math import ceil
        preroll_ms = 350
        preroll_blocks = ceil(preroll_ms / BLOCK_MS)
        preroll = deque(maxlen=preroll_blocks)

        sliding = deque(maxlen=int((RATE * WAKE_WINDOW_S) / (BLOCK_FRAMES)))
        last_trigger_ts = 0.0
        last_stt_ts = 0.0
        listening_for_wake = True

        print("⏳ Waiting for wake word…")
        
        while True:
            b = q.get()
            preroll.append(b)
            sliding.append(b)

            # Only try STT every WAKE_STEP_S
            if (time.time() - last_stt_ts) < WAKE_STEP_S:
                continue
            last_stt_ts = time.time()

            # Quick VAD gate: only STT if there's speech in the window
            if not sliding:
                continue
            if not any(vad_wake.is_speech(ch, RATE) for ch in sliding):
                continue
            if not listening_for_wake:
                continue

            # Don’t trigger twice too fast
            if (time.time() - last_trigger_ts) < WAKE_COOLDOWN_S:
                continue

            # Run tiny ASR on the current window
            win_bytes = b"".join(list(sliding))
            audio = int16_to_float32(win_bytes)
            audio = agc_preprocess(audio)

            segs, _ = asr_wake.transcribe(
                audio, language="en", task="transcribe",
                beam_size=1, temperature=0.0, vad_filter=True,
                condition_on_previous_text=False
            )
            txt = "".join(s.text for s in segs).strip()
            if not txt:
                continue
            # print(f"[wake window]: {txt}")  # debug

            if contains_wake_phrase(txt):
                last_trigger_ts = time.time()
                print((f"🔔 Wake detected (\"{txt}\") — listening…") if callable(print) else f"🔔 Wake detected (\"{txt}\") — listening…")

                # 🔧 Flush buffers so leftover "evil house" audio can't re-trigger
                sliding.clear()
                preroll.clear()
                while not q.empty():
                    q.get_nowait()

                beep()
                time.sleep(WAKE_REARM_DELAY_S)

                text = capture_and_transcribe_command(q, vad_cmd, asr_cmd, preroll)
                print("You said:", text or "[no speech detected]")
                print("⏳ Waiting for wake word…")

                if text:
                    # ---- LLM with session memory ----
                    resp = llm_reply(text, SESSION_ID)    # <-- pass session id
                    print("Mara:", resp)

                    # Synthesize speech (Piper PCM preferred; 'say' process as fallback)
                    tts_obj = synthesize_tts(resp)
                    if not tts_obj:
                        # nothing we can speak; just move on
                        last_trigger_ts = time.time()
                        listening_for_wake = True
                        continue

                    # Allow detector while speaking (so we can capture the interrupt phrase)
                    listening_for_wake = True

                    from math import ceil
                    speaking_started = time.time()

                    mode, payload = tts_obj
                    stop_tts = None
                    proc = None

                    if mode == "pcm":
                        audio, sr = payload
                        stop_tts = tts_play_async_pcm(audio, sr)
                    elif mode == "proc":
                        proc = payload  # Popen handle

                    # Clear windows so we don't mis-hear previous command
                    sliding.clear()
                    preroll.clear()
                    while not q.empty():
                        try: q.get_nowait()
                        except queue.Empty: break

                    # --- while speaking, scan for interrupt phrases with tiny ASR ---
                    while True:
                        # Determine if still speaking
                        still_speaking = False
                        if mode == "pcm":
                            still_speaking = True  # assume yes; thread ends when audio finishes
                        else:
                            still_speaking = (proc.poll() is None)

                        # Fill sliding window from mic
                        try:
                            b = q.get(timeout=0.05)
                            sliding.append(b)
                        except queue.Empty:
                            pass

                        # STT step: only if VAD sees speech in the window
                        if sliding and any(vad_wake.is_speech(ch, RATE) for ch in sliding):
                            if (time.time() - last_stt_ts) >= WAKE_STEP_S:
                                last_stt_ts = time.time()
                                win_bytes = b"".join(list(sliding))
                                audio_win = agc_preprocess(int16_to_float32(win_bytes))
                                segs, _ = asr_wake.transcribe(
                                    audio_win, language="en", beam_size=1, temperature=0.0,
                                    vad_filter=True, condition_on_previous_text=False
                                )
                                heard = "".join(s.text for s in segs).strip()
                                if heard and contains_interrupt(heard):
                                    print("⛔ Voice interrupt detected — stopping speech.")
                                    if mode == "pcm" and stop_tts:
                                        stop_tts()
                                    if mode == "proc" and proc and proc.poll() is None:
                                        try:
                                            proc.terminate()
                                        except Exception:
                                            pass
                                    # Drain and immediately capture the new utterance
                                    sliding.clear(); preroll.clear()
                                    while not q.empty():
                                        try: q.get_nowait()
                                        except queue.Empty: break

                                    print("🎙️  Listening after interrupt…")
                                    text2 = capture_and_transcribe_command(q, vad_cmd, asr_cmd, preroll)
                                    print("You said:", text2 or "[no speech detected]")
                                    if text2:
                                        # ---- LLM with session memory (same session id) ----
                                        resp2 = llm_reply(text2, SESSION_ID)   # <-- pass session id
                                        print("Mara:", resp2)
                                        # (optional) speak the new reply without allowing recursive interrupt,
                                        # or call the same block again if you want nested interrupts
                                        t2 = synthesize_tts(resp2)
                                        if t2:
                                            m2, p2 = t2
                                            if m2 == "pcm":
                                                stop2 = tts_play_async_pcm(p2[0], p2[1])
                                                time.sleep(0.2)
                                                stop2()
                                            else:
                                                import time as _t
                                                _t.sleep(0.2)
                                    last_trigger_ts = time.time()
                                    break  # exit speaking loop

                        # Safety: leave if speech likely done
                        if (time.time() - speaking_started) > 30:
                            # stop leftover audio
                            if mode == "pcm" and stop_tts:
                                try: stop_tts()
                                except Exception: pass
                            if mode == "proc" and proc and proc.poll() is None:
                                try: proc.terminate()
                                except Exception: pass
                            break

                    # Done speaking or interrupted
                    time.sleep(0.2)   # small cushion
                    last_trigger_ts = time.time()
                    listening_for_wake = True
                    print("⏳ Waiting for wake word…")
                    # say_with_rvc("")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")