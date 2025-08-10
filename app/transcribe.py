import os
from typing import Optional
from openai import OpenAI
from .config import OPENAI_API_KEY

def transcribe_audio(file_path: str) -> Optional[str]:
    key = OPENAI_API_KEY
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    # Prefer newer small transcription models if available; fallback to whisper-1
    model = "gpt-4o-mini-transcribe"
    try:
        with open(file_path, "rb") as f:
            resp = client.audio.transcriptions.create(model=model, file=f)
        return (resp.text or "").strip()
    except Exception:
        try:
            with open(file_path, "rb") as f:
                resp = client.audio.transcriptions.create(model="whisper-1", file=f)
            return (resp.text or "").strip()
        except Exception:
            return None
