"""
Web UI API for managing Piper voice models (.onnx + .onnx.json) on a shared volume.
"""
from __future__ import annotations

import glob
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/data/models")).resolve()

# Piper bundle: voice.onnx + voice.onnx.json — basename must match this pattern
SAFE_STEM = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
SHORT_NAME = re.compile(r"^[a-z]{2}_[A-Z]{2}-([a-zA-Z0-9_]+)-\w+\.onnx$")

app = FastAPI(title="Mara Piper model manager", version="1.0.0")

_static = Path(__file__).resolve().parent / "static"


def _stem_from_onnx_filename(name: str) -> str:
    if not name.endswith(".onnx"):
        raise HTTPException(status_code=400, detail="ONNX file must end with .onnx")
    stem = name[:-5]
    if not SAFE_STEM.match(stem):
        raise HTTPException(status_code=400, detail="Invalid ONNX filename")
    return stem


def _short_label(onnx_basename: str) -> str:
    m = SHORT_NAME.match(onnx_basename)
    if m:
        return m.group(1).lower()
    return Path(onnx_basename).stem.lower()


def list_models() -> list[dict]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for onnx_path in sorted(glob.glob(str(MODELS_DIR / "*.onnx"))):
        p = Path(onnx_path)
        json_path = p.with_suffix(".onnx.json")
        if not json_path.is_file():
            continue
        st = p.stat()
        jst = json_path.stat()
        name = p.name
        out.append(
            {
                "id": p.stem,
                "short_name": _short_label(name),
                "onnx": name,
                "json": json_path.name,
                "onnx_bytes": st.st_size,
                "json_bytes": jst.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return out


@app.get("/health")
def health():
    return {"status": "ok", "models_dir": str(MODELS_DIR)}


@app.get("/api/models")
def api_models():
    return {"models": list_models(), "models_dir": str(MODELS_DIR)}


@app.delete("/api/models/{model_id}")
def api_delete(model_id: str):
    stem = model_id.strip()
    if not stem or ".." in stem or "/" in stem or "\\" in stem:
        raise HTTPException(status_code=400, detail="Invalid model id")
    if not SAFE_STEM.match(stem):
        raise HTTPException(status_code=400, detail="Invalid model id")

    onnx = MODELS_DIR / f"{stem}.onnx"
    cfg = MODELS_DIR / f"{stem}.onnx.json"
    if not onnx.is_file():
        raise HTTPException(status_code=404, detail="Model not found")
    onnx.unlink()
    if cfg.is_file():
        cfg.unlink()
    return {"deleted": stem}


@app.post("/api/models/upload")
async def api_upload(
    onnx: UploadFile = File(...),
    config_json: UploadFile = File(..., description="Piper config: name.onnx.json"),
):
    if not onnx.filename:
        raise HTTPException(status_code=400, detail="Missing ONNX filename")
    if not config_json.filename:
        raise HTTPException(status_code=400, detail="Missing JSON filename")

    stem = _stem_from_onnx_filename(onnx.filename)
    expected_json = f"{stem}.onnx.json"
    if config_json.filename != expected_json:
        raise HTTPException(
            status_code=400,
            detail=f"JSON file must be named {expected_json} (matching the ONNX stem)",
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = MODELS_DIR / f"{stem}.onnx"
    json_path = MODELS_DIR / f"{stem}.onnx.json"

    data_onnx = await onnx.read()
    data_json = await config_json.read()

    if len(data_onnx) < 1024:
        raise HTTPException(status_code=400, detail="ONNX file seems too small")
    if len(data_json) < 10:
        raise HTTPException(status_code=400, detail="JSON config seems invalid")

    tmp_onnx = onnx_path.with_suffix(onnx_path.suffix + ".tmp")
    tmp_json = json_path.with_suffix(json_path.suffix + ".tmp")
    try:
        tmp_onnx.write_bytes(data_onnx)
        tmp_json.write_bytes(data_json)
        tmp_onnx.replace(onnx_path)
        tmp_json.replace(json_path)
    except OSError as e:
        for p in (tmp_onnx, tmp_json):
            if p.is_file():
                p.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {"ok": True, "id": stem, "models": list_models()}


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = Path(__file__).resolve().parent / "static" / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail="UI not found")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


# Mount static assets after routes so `/` and `/api/*` are not shadowed.
if _static.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")
