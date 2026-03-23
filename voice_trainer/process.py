#!/usr/bin/env python3
"""
Main entry point: Download, transcribe, and slice a YouTube video in one command.
"""
import argparse
import sys
from pathlib import Path

from download import download_audio
from transcribe_and_slice import transcribe_audio, slice_audio, save_metadata


def process_video(
    url: str,
    output_name: str = None,
    model_size: str = "medium",
    language: str = "en",
    min_duration: float = 0.5,
    max_duration: float = 15.0,
    device: str = "auto"
):
    """
    Full pipeline: Download → Transcribe → Slice
    
    Args:
        url: YouTube video URL
        output_name: Name for output folder (defaults to video title)
        model_size: Whisper model size
        language: Language code for transcription
        min_duration: Minimum clip duration
        max_duration: Maximum clip duration
        device: Device for Whisper (auto, cpu, cuda)
    """
    print("=" * 60)
    print("VOICE TRAINER - YouTube to Training Data Pipeline")
    print("=" * 60)
    
    # Step 1: Download
    print("\n[1/3] Downloading audio...")
    audio_path = download_audio(url, "input")
    
    if output_name is None:
        output_name = Path(audio_path).stem
    
    # Clean output name for folder
    output_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in output_name)
    output_dir = Path("output") / output_name
    
    # Step 2: Transcribe
    print("\n[2/3] Transcribing audio...")
    segments = transcribe_audio(
        audio_path,
        model_size=model_size,
        language=language,
        device=device
    )
    
    # Save raw transcript
    transcript_dir = output_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    transcript_path = transcript_dir / "full_transcript.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"Saved transcript: {transcript_path}")
    
    # Also save readable text version
    text_path = transcript_dir / "full_transcript.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")
    print(f"Saved readable transcript: {text_path}")
    
    # Step 3: Slice
    print("\n[3/3] Slicing audio into clips...")
    sliced_dir = output_dir / "sliced"
    clips_metadata = slice_audio(
        audio_path,
        segments,
        str(sliced_dir),
        min_duration=min_duration,
        max_duration=max_duration
    )
    
    # Save metadata
    save_metadata(clips_metadata, str(sliced_dir))
    
    # Final summary
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - transcripts/  : Full transcription")
    print(f"  - sliced/       : Audio clips + metadata.csv")
    print(f"\nNext steps:")
    print(f"  1. Review clips in {sliced_dir}/")
    print(f"  2. Remove any bad clips (noise, wrong speaker, etc.)")
    print(f"  3. Edit metadata.csv if needed")
    print(f"  4. Use for Piper training!")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download, transcribe, and slice YouTube videos for TTS training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process.py "https://youtube.com/watch?v=xxxxx"
  python process.py "https://youtube.com/watch?v=xxxxx" --name "goku_clips"
  python process.py "https://youtube.com/watch?v=xxxxx" --model large-v3 --language ja
        """
    )
    
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--name", "-n", help="Output folder name (defaults to video title)")
    parser.add_argument("--model", "-m", default="medium", 
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--language", "-l", default="en", help="Language code (default: en)")
    parser.add_argument("--min-duration", type=float, default=0.5, 
                        help="Minimum clip duration in seconds (default: 0.5)")
    parser.add_argument("--max-duration", type=float, default=15.0,
                        help="Maximum clip duration in seconds (default: 15.0)")
    parser.add_argument("--device", default="auto", 
                        choices=["auto", "cpu", "cuda"],
                        help="Device for Whisper inference (default: auto)")
    
    args = parser.parse_args()
    
    process_video(
        url=args.url,
        output_name=args.name,
        model_size=args.model,
        language=args.language,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        device=args.device
    )


if __name__ == "__main__":
    main()
