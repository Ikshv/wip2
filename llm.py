# llm.py — Ollama /api/generate with context memory (no /api/chat fallbacks)
import os, json, requests
from pathlib import Path
from typing import Optional

BASE   = os.getenv("OLLAMA_BASE", "http://localhost:11434")
MODEL  = os.getenv("MARA_MODEL", "llama2-uncensored:latest")

# at top of llm.py
import json, os, pathlib

def load_system():
    template = pathlib.Path("mara.tmpl").read_text()
    cfg = {
        "NAME": os.getenv("MARA_NAME", "Mara"),
        "ALT_NAME": os.getenv("MARA_ALT_NAME", "Mara"),
        "TONE": os.getenv("MARA_TONE", "fun, upbeat, a bit cheeky"),
        "EMOJIS": os.getenv("MARA_EMOJIS", "auto"),
        "HUMOR": os.getenv("MARA_HUMOR", "2"),        # 0..3
        "PROFANITY": os.getenv("MARA_PROFANITY", "mild"),
        "FLIRT": os.getenv("MARA_FLIRT", "off"),       # off/soft
    }
    return template.format(**cfg)

SYSTEM = load_system()

KEEP_ALIVE = os.getenv("MARA_KEEP_ALIVE", "5m")
OPTIONS = {
    "num_ctx": int(os.getenv("MARA_NUM_CTX", "2048")),
    "num_predict": int(os.getenv("MARA_NUM_PREDICT", "128")),
    "temperature": float(os.getenv("MARA_TEMPERATURE", "0.2")),
}

CTX_FILE = Path(os.path.expanduser(os.getenv("MARA_CONTEXT_FILE", "~/.mara_ctx.json")))
SESSION  = requests.Session()

def _load_ctx(session_id: str) -> Optional[list]:
    try:
        if CTX_FILE.exists():
            db = json.loads(CTX_FILE.read_text())
            return db.get(MODEL, {}).get(session_id)
    except Exception:
        pass
    return None

def _save_ctx(session_id: str, context: list):
    try:
        db = {}
        if CTX_FILE.exists():
            db = json.loads(CTX_FILE.read_text())
        db.setdefault(MODEL, {})[session_id] = context
        CTX_FILE.parent.mkdir(parents=True, exist_ok=True)
        CTX_FILE.write_text(json.dumps(db))
    except Exception:
        pass

def reply(user_text: str, session_id: str = "default") -> str:
    prompt = f"{SYSTEM}\nUser: {user_text}\nAssistant:"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": OPTIONS,
    }
    ctx = _load_ctx(session_id)
    if ctx:
        payload["context"] = ctx

    try:
        r = SESSION.post(f"{BASE}/api/generate", json=payload, timeout=300)
        r.raise_for_status()
    except requests.ConnectionError as ce:
        raise RuntimeError(f"Cannot reach Ollama at {BASE}. Is `ollama serve` running?") from ce
    except requests.HTTPError as he:
        # If 404 here, it's almost always wrong model tag or very old server build.
        code = he.response.status_code if he.response is not None else "?"
        raise RuntimeError(
            f"Ollama returned {code} for /api/generate using model '{MODEL}'. "
            "Make sure the tag exists (see `ollama list`) and is compatible."
        ) from he

    data = r.json()
    new_ctx = data.get("context")
    if new_ctx:
        _save_ctx(session_id, new_ctx)

    return (data.get("response") or "").strip()

if __name__ == "__main__":
    import sys
    txt = " ".join(sys.argv[1:]) or "hello there"
    sid = os.getenv("MARA_SESSION", "default")
    print(reply(txt, session_id=sid))