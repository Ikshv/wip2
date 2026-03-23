#!/usr/bin/env python3
"""
Merge multiple processed datasets into a single training dataset.
Run this before training to combine all your voice clips.
"""
import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime


def merge_datasets(
    output_dir: str,
    input_dirs: list = None,
    voice_name: str = "goku"
):
    """
    Merge multiple sliced datasets into one training-ready dataset.
    
    Args:
        output_dir: Where to put the merged dataset
        input_dirs: List of directories to merge (auto-detect if None)
        voice_name: Name for the voice (used in output filenames)
    """
    output_path = Path(output_dir)
    wavs_dir = output_path / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect input directories if not specified
    if input_dirs is None:
        base_output = Path("output")
        input_dirs = []
        for d in base_output.iterdir():
            if d.is_dir() and (d / "sliced" / "metadata.json").exists():
                input_dirs.append(str(d / "sliced"))
        print(f"Auto-detected {len(input_dirs)} datasets:")
        for d in input_dirs:
            print(f"  - {d}")
    
    all_metadata = []
    clip_index = 0
    total_duration = 0
    
    print(f"\nMerging datasets into: {output_path}")
    print("=" * 60)
    
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        metadata_path = input_path / "metadata.json"
        
        if not metadata_path.exists():
            print(f"  [SKIP] No metadata.json in {input_dir}")
            continue
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        dataset_name = input_path.parent.name
        print(f"\n  Processing: {dataset_name}")
        print(f"  Clips: {len(metadata)}")
        
        for item in metadata:
            old_filename = item["file"]
            old_path = input_path / old_filename
            
            if not old_path.exists():
                print(f"    [SKIP] File not found: {old_filename}")
                continue
            
            # Create new standardized filename
            new_filename = f"{voice_name}_{clip_index:05d}.wav"
            new_path = wavs_dir / new_filename
            
            # Copy file
            shutil.copy2(old_path, new_path)
            
            # Update metadata
            all_metadata.append({
                "file": new_filename,
                "text": item["text"],
                "duration": item["duration"],
                "source_dataset": dataset_name,
                "original_file": old_filename
            })
            
            total_duration += item["duration"]
            clip_index += 1
    
    # Save merged metadata as JSON
    json_path = output_path / "metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    
    # Save as CSV (LJSpeech format for Piper)
    csv_path = output_path / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        for item in all_metadata:
            # LJSpeech format: filename_without_ext|text|text
            basename = item["file"].replace(".wav", "")
            text = item["text"].replace("|", " ")
            f.write(f"{basename}|{text}|{text}\n")
    
    # Save training info
    info = {
        "voice_name": voice_name,
        "total_clips": len(all_metadata),
        "total_duration_seconds": total_duration,
        "total_duration_minutes": total_duration / 60,
        "source_datasets": list(set(item["source_dataset"] for item in all_metadata)),
        "created_at": datetime.now().isoformat(),
    }
    info_path = output_path / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"Total clips: {len(all_metadata)}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"\nOutput files:")
    print(f"  - {wavs_dir}/ ({len(all_metadata)} WAV files)")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print(f"  - {info_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Merge voice datasets for training")
    parser.add_argument(
        "--output", "-o",
        default="training/dataset",
        help="Output directory for merged dataset"
    )
    parser.add_argument(
        "--name", "-n",
        default="goku",
        help="Voice name (used in filenames)"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input directories to merge (auto-detect if not specified)"
    )
    
    args = parser.parse_args()
    
    merge_datasets(
        output_dir=args.output,
        input_dirs=args.inputs if args.inputs else None,
        voice_name=args.name
    )


if __name__ == "__main__":
    main()
