#!/usr/bin/env python3
"""
Transcribe audio using Whisper and slice into sentence segments.
"""
import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from faster_whisper import WhisperModel
from pydub import AudioSegment
from tqdm import tqdm


def transcribe_audio(
    audio_path: str,
    model_size: str = "medium",
    language: str = "en",
    device: str = "auto"
) -> List[Dict[str, Any]]:
    """
    Transcribe audio file using faster-whisper.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large-v3)
        language: Language code
        device: Device to use (auto, cpu, cuda)
        
    Returns:
        List of segments with text, start, and end times
    """
    print(f"Loading Whisper model: {model_size}")
    
    # Determine compute type based on device
    if device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "cpu"  # MPS not fully supported yet
            compute_type = "int8"
        else:
            device = "cpu"
            compute_type = "int8"
    else:
        compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"Using device: {device}, compute type: {compute_type}")
    
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    print(f"Transcribing: {audio_path}")
    segments, info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        vad_filter=True,  # Filter out non-speech
        vad_parameters=dict(
            min_silence_duration_ms=500,  # Minimum silence to split
            speech_pad_ms=200,  # Padding around speech
        )
    )
    
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    
    results = []
    print("\n" + "="*60)
    print("TRANSCRIPTION")
    print("="*60)
    
    segment_count = 0
    for segment in segments:
        segment_count += 1
        text = segment.text.strip()
        duration = segment.end - segment.start
        
        # Print each transcribed segment
        print(f"\n[{segment_count}] {segment.start:.2f}s - {segment.end:.2f}s ({duration:.2f}s)")
        print(f"    \"{text}\"")
        
        results.append({
            "text": text,
            "start": segment.start,
            "end": segment.end,
            "words": [
                {"word": w.word, "start": w.start, "end": w.end}
                for w in (segment.words or [])
            ]
        })
    
    print(f"\n{'='*60}")
    print(f"Total segments transcribed: {segment_count}")
    print("="*60 + "\n")
    
    return results


def slice_audio(
    audio_path: str,
    segments: List[Dict[str, Any]],
    output_dir: str,
    min_duration: float = 0.5,
    max_duration: float = 15.0,
    padding_ms: int = 100
) -> List[Dict[str, Any]]:
    """
    Slice audio file based on transcription segments.
    
    Args:
        audio_path: Path to source audio file
        segments: List of transcription segments
        output_dir: Directory to save sliced audio
        min_duration: Minimum clip duration in seconds
        max_duration: Maximum clip duration in seconds
        padding_ms: Padding to add before/after each clip in milliseconds
        
    Returns:
        List of metadata for each clip
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading audio: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    
    # Get base name for clips
    base_name = Path(audio_path).stem
    
    clips_metadata = []
    clip_index = 0
    
    print(f"\n{'='*60}")
    print("SLICING AUDIO")
    print("="*60)
    print(f"Processing {len(segments)} segments...")
    print(f"Min duration: {min_duration}s, Max duration: {max_duration}s")
    print(f"Padding: {padding_ms}ms\n")
    
    skipped_short = 0
    skipped_long = 0
    skipped_empty = 0
    
    for i, segment in enumerate(segments):
        duration = segment["end"] - segment["start"]
        text = segment["text"].strip()
        
        # Skip clips that are too short
        if duration < min_duration:
            print(f"  [SKIP] Segment {i+1}: Too short ({duration:.2f}s < {min_duration}s)")
            skipped_short += 1
            continue
            
        # Skip clips that are too long
        if duration > max_duration:
            print(f"  [SKIP] Segment {i+1}: Too long ({duration:.2f}s > {max_duration}s)")
            skipped_long += 1
            continue
        
        # Skip empty or very short text
        if len(text) < 2:
            print(f"  [SKIP] Segment {i+1}: Empty or too short text")
            skipped_empty += 1
            continue
        
        # Calculate start/end with padding
        start_ms = max(0, int(segment["start"] * 1000) - padding_ms)
        end_ms = min(len(audio), int(segment["end"] * 1000) + padding_ms)
        
        # Extract clip
        clip = audio[start_ms:end_ms]
        
        # Normalize audio levels
        clip = clip.normalize()
        
        # Generate filename
        clip_filename = f"{base_name}_{clip_index:04d}.wav"
        clip_path = output_path / clip_filename
        
        # Export as WAV (22050 Hz mono for Piper)
        clip = clip.set_frame_rate(22050).set_channels(1)
        clip.export(clip_path, format="wav")
        
        # Print saved clip info
        print(f"  [SAVED] {clip_filename} ({duration:.2f}s)")
        print(f"          \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        
        # Store metadata
        clips_metadata.append({
            "file": clip_filename,
            "text": text,
            "duration": duration,
            "original_start": segment["start"],
            "original_end": segment["end"]
        })
        
        clip_index += 1
    
    print(f"\n{'='*60}")
    print("SLICING COMPLETE")
    print("="*60)
    print(f"Clips saved: {clip_index}")
    print(f"Skipped (too short): {skipped_short}")
    print(f"Skipped (too long): {skipped_long}")
    print(f"Skipped (empty text): {skipped_empty}")
    print("="*60 + "\n")
    
    return clips_metadata


def save_metadata(metadata: List[Dict[str, Any]], output_path: str):
    """Save metadata to JSON and CSV formats."""
    output_path = Path(output_path)
    
    # Save as JSON
    json_path = output_path / "metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata JSON: {json_path}")
    
    # Save as CSV (Piper format: filename|text)
    csv_path = output_path / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        for item in metadata:
            # Escape pipes in text
            text = item["text"].replace("|", " ")
            f.write(f"{item['file']}|{text}\n")
    print(f"Saved metadata CSV: {csv_path}")
    
    # Print summary
    total_duration = sum(item["duration"] for item in metadata)
    print(f"\n=== Summary ===")
    print(f"Total clips: {len(metadata)}")
    print(f"Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Average clip duration: {total_duration/len(metadata):.1f} seconds" if metadata else "N/A")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe and slice audio for TTS training")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--output-dir", "-o", default="output/sliced", help="Output directory for clips")
    parser.add_argument("--model", "-m", default="medium", help="Whisper model size")
    parser.add_argument("--language", "-l", default="en", help="Language code")
    parser.add_argument("--min-duration", type=float, default=0.5, help="Minimum clip duration (seconds)")
    parser.add_argument("--max-duration", type=float, default=15.0, help="Maximum clip duration (seconds)")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("VOICE TRAINER - Transcribe & Slice")
    print("="*60)
    print(f"Input file: {args.audio_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Whisper model: {args.model}")
    print(f"Language: {args.language}")
    print(f"Duration range: {args.min_duration}s - {args.max_duration}s")
    print("="*60 + "\n")
    
    # Transcribe
    segments = transcribe_audio(
        args.audio_path,
        model_size=args.model,
        language=args.language,
        device=args.device
    )
    
    # Save raw transcription
    transcript_dir = Path("output/transcripts")
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / f"{Path(args.audio_path).stem}_transcript.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"Saved raw transcript: {transcript_path}")
    
    # Slice audio
    clips_metadata = slice_audio(
        args.audio_path,
        segments,
        args.output_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    
    # Save metadata
    save_metadata(clips_metadata, args.output_dir)


if __name__ == "__main__":
    main()
