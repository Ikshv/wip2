"""Fetch melspectrogram + embedding ONNX files missing from the PyPI openwakeword wheel."""
from __future__ import annotations

import pathlib
import urllib.request

import openwakeword

def main() -> None:
    models_dir = pathlib.Path(openwakeword.__file__).resolve().parent / "resources" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    base = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
    for name in ("melspectrogram.onnx", "embedding_model.onnx"):
        dest = models_dir / name
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"skip existing {dest}")
            continue
        url = f"{base}/{name}"
        print(f"download {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    main()
