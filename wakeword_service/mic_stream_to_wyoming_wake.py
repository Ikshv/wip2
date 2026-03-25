#!/usr/bin/env python3
"""
Stream the default microphone to a Wyoming wake-word server over TCP.

Use this to verify wake detection without Home Assistant.

If bare ``pip`` errors with "activated virtualenv (required)", use a venv:

  cd wakeword_service
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements_mic_client.txt

Or run without activating: ``.venv/bin/python mic_stream_to_wyoming_wake.py ...``

macOS: if sounddevice fails to load, install PortAudio: ``brew install portaudio``

Run one command per line (do not paste comment lines starting with # into the shell).

Examples (wake service on localhost, Docker publishing 10300):

  .venv/bin/python mic_stream_to_wyoming_wake.py --uri tcp://127.0.0.1:10300 --describe --ping

Remote host:

  .venv/bin/python mic_stream_to_wyoming_wake.py --uri tcp://192.168.1.50:10300

Options: --describe --ping --verbose --list-devices --device <index>
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import queue
import sys
import threading
from typing import Any, Optional


def _import_sounddevice():
    try:
        import sounddevice as sd

        return sd
    except ImportError:
        print(
            "Missing sounddevice. Install with:\n"
            "  pip install sounddevice\n"
            "On macOS you may need: brew install portaudio",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _import_wyoming():
    try:
        from wyoming.audio import AudioChunk, AudioStart, AudioStop
        from wyoming.client import AsyncClient
        from wyoming.info import Describe
        from wyoming.ping import Ping
        from wyoming.wake import Detection

        return AudioChunk, AudioStart, AudioStop, AsyncClient, Describe, Ping, Detection
    except ImportError as e:
        print(f"Missing wyoming: {e}\n  pip install wyoming", file=sys.stderr)
        raise SystemExit(1)


TARGET_RATE = 16000
SAMPLE_WIDTH = 2  # int16
CHANNELS = 1


def _resample_linear_int16(mono_i16: "np.ndarray", src_rate: int, dst_rate: int) -> "np.ndarray":
    import numpy as np

    if src_rate == dst_rate or mono_i16.size < 2:
        return mono_i16.astype(np.int16, copy=False)
    f32 = mono_i16.astype(np.float32) / 32768.0
    new_len = max(1, int(len(f32) * dst_rate / src_rate))
    xp = np.linspace(0.0, len(f32) - 1, num=new_len, dtype=np.float64)
    y = np.interp(xp, np.arange(len(f32), dtype=np.float64), f32.astype(np.float64))
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).astype(np.int16)


def _run_mic_input_stream(
    sd: Any,
    *,
    device: Optional[int],
    in_rate: int,
    chunk_samples: int,
    pcm_queue: "queue.Queue[Optional[bytes]]",
    close_event: threading.Event,
) -> None:
    """
    One continuous PortAudio input stream (avoids open/close per chunk — macOS
    mic indicator stays solid instead of flashing).
    """
    import numpy as np

    acc = np.zeros((0,), dtype=np.int16)
    chunk_bytes = chunk_samples * SAMPLE_WIDTH
    blocksize = max(128, int(in_rate * 0.02))

    def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        nonlocal acc
        if close_event.is_set():
            raise sd.CallbackStop
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        # float32 from PortAudio (default); int16-only streams are rarer on macOS
        mono = np.squeeze(indata.astype(np.float32, copy=False))
        if mono.ndim > 1:
            mono = mono.mean(axis=1)
        mono = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
        if in_rate != TARGET_RATE:
            mono = _resample_linear_int16(mono, in_rate, TARGET_RATE)
        if mono.size:
            acc = np.concatenate((acc, mono))
        while acc.size >= chunk_samples:
            piece = acc[:chunk_samples]
            acc = acc[chunk_samples:]
            data = piece.tobytes()
            try:
                pcm_queue.put_nowait(data)
            except queue.Full:
                try:
                    pcm_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    pcm_queue.put_nowait(data)
                except queue.Full:
                    pass

    with sd.InputStream(
        samplerate=in_rate,
        channels=CHANNELS,
        dtype="float32",
        device=device,
        blocksize=blocksize,
        callback=callback,
    ):
        close_event.wait()


async def _reader_loop(
    client: Any,
    stop: asyncio.Event,
    verbose: bool,
    Detection: Any,
) -> None:
    while not stop.is_set():
        try:
            event = await asyncio.wait_for(client.read_event(), timeout=0.25)
        except asyncio.TimeoutError:
            continue
        if event is None:
            if verbose:
                print("Server closed connection", file=sys.stderr)
            stop.set()
            return
        _print_server_event(event, verbose, Detection)


def _print_server_event(event: Any, verbose: bool, Detection: Any) -> None:
    if event.type == "detection":
        try:
            d = Detection.from_event(event)
            extra = []
            ctx = (event.data or {}).get("context") or {}
            if isinstance(ctx, dict):
                for k in ("model", "score"):
                    if k in ctx:
                        extra.append(f"{k}={ctx[k]!r}")
            tail = f" ({', '.join(extra)})" if extra else ""
            print(f"*** wake detection: {d.name!r}{tail}")
        except Exception:
            print(f"*** detection (raw): {event.type} {event.data}", file=sys.stderr)
        return
    if event.type == "info":
        print("Received info (describe OK)")
        if verbose and event.data:
            print(event.data)
        return
    if event.type == "pong":
        print("pong")
        return
    if verbose:
        print(f"<- {event.type} {event.data or ''}")


async def _run(args: argparse.Namespace) -> None:
    sd = _import_sounddevice()
    if args.list_devices:
        print(sd.query_devices())
        return

    (
        AudioChunk,
        AudioStart,
        AudioStop,
        AsyncClient,
        Describe,
        Ping,
        Detection,
    ) = _import_wyoming()

    client = AsyncClient.from_uri(args.uri)
    await client.connect()
    stop = asyncio.Event()
    reader = asyncio.create_task(
        _reader_loop(client, stop, args.verbose, Detection)
    )

    try:
        if args.ping:
            await client.write_event(Ping().event())
            await asyncio.sleep(0.2)

        if args.describe:
            await client.write_event(Describe().event())
            await asyncio.sleep(0.5)

        await client.write_event(
            AudioStart(rate=TARGET_RATE, width=SAMPLE_WIDTH, channels=CHANNELS).event()
        )

        loop = asyncio.get_running_loop()
        chunk_samples = max(160, int(TARGET_RATE * args.chunk_ms / 1000.0))

        pcm_queue: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=128)
        mic_close = threading.Event()
        stream_future = loop.run_in_executor(
            None,
            lambda: _run_mic_input_stream(
                sd,
                device=args.device,
                in_rate=args.samplerate,
                chunk_samples=chunk_samples,
                pcm_queue=pcm_queue,
                close_event=mic_close,
            ),
        )

        await asyncio.sleep(0.15)
        if stream_future.done():
            exc = stream_future.exception()
            if exc is not None:
                raise RuntimeError(f"microphone stream failed to start: {exc}") from exc

        print(
            f"Streaming mic → {args.uri} at {TARGET_RATE} Hz mono int16 "
            f"(~{chunk_samples} samples/chunk, continuous capture). Ctrl+C to stop.",
            file=sys.stderr,
        )

        chunks_sent = 0
        stall_warned = False
        import numpy as np

        chunks_per_sec = max(1, int(TARGET_RATE / chunk_samples))
        try:
            while not stop.is_set():
                if stream_future.done():
                    exc = stream_future.exception()
                    if exc is not None:
                        raise RuntimeError(f"microphone stream ended with error: {exc}") from exc
                    break
                try:
                    pcm = await asyncio.wait_for(
                        asyncio.to_thread(pcm_queue.get),
                        timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    if chunks_sent == 0 and not stall_warned:
                        print(
                            "[mic] no audio yet — check mic permission, default input, "
                            "or run with --list-devices",
                            file=sys.stderr,
                        )
                        stall_warned = True
                    continue
                if pcm is None:
                    break
                chunks_sent += 1
                if args.meter and chunks_sent % chunks_per_sec == 0:
                    peak = float(np.max(np.abs(np.frombuffer(pcm, dtype=np.int16))))
                    print(f"[mic] level (int16 max abs in chunk): {peak:.0f}", file=sys.stderr)
                await client.write_event(
                    AudioChunk(
                        rate=TARGET_RATE,
                        width=SAMPLE_WIDTH,
                        channels=CHANNELS,
                        audio=pcm,
                    ).event()
                )
        finally:
            mic_close.set()
            try:
                pcm_queue.put_nowait(None)
            except Exception:
                pass
            try:
                await stream_future
            except Exception as e:
                print(f"[mic] stream thread: {e}", file=sys.stderr)

    except asyncio.CancelledError:
        raise
    except KeyboardInterrupt:
        print("\nStopping…", file=sys.stderr)
    finally:
        stop.set()
        try:
            await client.write_event(AudioStop().event())
        except Exception:
            pass
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader
        await client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream microphone PCM to a Wyoming wake-word server (TCP)."
    )
    parser.add_argument(
        "--uri",
        default="tcp://127.0.0.1:10300",
        help="Wyoming server URI (default: tcp://127.0.0.1:10300)",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=TARGET_RATE,
        help=f"Capture sample rate before resampling to {TARGET_RATE} Hz (default: {TARGET_RATE})",
    )
    parser.add_argument(
        "--chunk-ms",
        type=float,
        default=40.0,
        help="Approximate duration of each audio-chunk in ms (default: 40)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="sounddevice input device index (default: system default)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print sounddevice device list and exit",
    )
    parser.add_argument("--ping", action="store_true", help="Send ping before streaming")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Send describe before streaming (prints info response)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log all incoming Wyoming event types",
    )
    parser.add_argument(
        "--meter",
        action="store_true",
        help="Print approximate mic level to stderr ~1x/sec (sanity check for silence)",
    )
    args = parser.parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
