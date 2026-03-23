#!/usr/bin/env python3
"""
Download audio from YouTube videos using yt-dlp
"""
import os
import subprocess
import sys
from pathlib import Path


def download_audio(url: str, output_dir: str = "input") -> str:
    """
    Download audio from a YouTube video.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the audio file
        
    Returns:
        Path to the downloaded audio file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Output template - use video title as filename
    output_template = str(output_path / "%(title)s.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "wav",  # Convert to WAV for processing
        "--audio-quality", "0",  # Best quality
        "-o", output_template,
        "--no-playlist",  # Don't download playlists
        "--cookies-from-browser", "chrome",  # Use Chrome cookies for auth
        "--remote-components", "ejs:github",  # Download JS challenge solver
        "--print", "after_move:filepath",  # Print the final filepath
        url
    ]
    
    print(f"Downloading audio from: {url}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Get the output filepath from stdout
        filepath = result.stdout.strip().split('\n')[-1]
        print(f"Downloaded: {filepath}")
        return filepath
    except subprocess.CalledProcessError as e:
        print(f"Error downloading: {e.stderr}")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python download.py <youtube_url> [output_dir]")
        sys.exit(1)
    
    url = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "input"
    
    filepath = download_audio(url, output_dir)
    print(f"\nAudio saved to: {filepath}")


if __name__ == "__main__":
    main()
