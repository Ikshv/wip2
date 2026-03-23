# tts.py — Piper 1 (GPL) via Python API with a macOS fallback
import os
import os, subprocess, soundfile as sf, numpy as np, tempfile

def say(text: str):
    if not text:
        return
    # Try Piper first
    try:
        from piper.voice import PiperVoice
        import sounddevice as sd

        voice_name = os.getenv("PIPER_VOICE", "en_US-patrick-medium")
        try:
            voice = PiperVoice.load(voice_name)
        except Exception:
            # first run: fetch voice automatically
            PiperVoice.download_voice(voice_name)
            voice = PiperVoice.load(voice_name)

        # Simple: generate PCM and play
        audio = voice.speak_text(text)  # returns float32 PCM at voice.config.sample_rate
        sr = voice.config.sample_rate
        # ✅ Save audio to file
        sf.write("output.wav", audio, sr)
        print("Saved to output.wav")
        sd.play(audio, voice.config.sample_rate)
        sd.wait()
        return
    except Exception as e:
        # Fallback to macOS built-in TTS if Piper not ready
        try:
            import subprocess
            subprocess.run(["say", text], check=False)
        except Exception:
            print(f"[TTS fallback only] {text}")


def say_with_rvc(text: str):
    from piper.voice import PiperVoice
    import sounddevice as sd

    voice = PiperVoice.load(os.getenv("PIPER_VOICE", "en_US-patrick-medium"))
    pcm = voice.speak_text(text)
    sr = voice.config.sample_rate

    # write neutral wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
        in_wav = f_in.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        out_wav = f_out.name

    sf.write(in_wav, pcm, sr)

    # call the RVC converter
    rvc_dir   = os.getenv("RVC_REPO", "/Users/you/Desktop/rvc")
    model_pth = os.getenv("RVC_MODEL", "/Users/isaacshvartsman/Downloads/Yajirobe-DBSparkingZero/Yajirobe-DBSparkingZero_160e_2720s.pth")
    index_pth = os.getenv("RVC_INDEX", "/Users/isaacshvartsman/Downloads/Yajirobe-DBSparkingZero/Yajirobe-DBSparkingZero.index")
    cmd = [
        os.path.join(rvc_dir, ".venv/bin/python"),
        os.path.join(rvc_dir, "convert_rvc.py"),
        "--model", model_pth,
        "--in", in_wav,
        "--out", out_wav,
        "--f0", "harvest",
        "--key", "0",
        "--device", "mps",
    ]
    if os.path.exists(index_pth):
        cmd += ["--index", index_pth]

    subprocess.run(cmd, check=True)

    # play converted
    data, sr2 = sf.read(out_wav, dtype="float32")
    sd.play(data, sr2)
    sd.wait()

    # cleanup
    try:
        os.remove(in_wav)
        os.remove(out_wav)
    except Exception:
        pass
