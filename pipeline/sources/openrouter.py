"""OpenRouter model list - used only as a name dictionary.

Each entry carries a display name and, for open-weight models, the Hugging Face
repo id. That gives extra spellings for the name join ("Meta: Llama 4 Maverick"
-> meta-llama/Llama-4-Maverick-17B-128E-Instruct). No scores or prices are taken.
"""
from __future__ import annotations

from ..http import get_json

URL = "https://openrouter.ai/api/v1/models"


def parse(payload: dict) -> list[dict]:
    out = []
    for m in (payload or {}).get("data") or []:
        hf = m.get("hugging_face_id")
        if hf and m.get("name"):
            out.append({"hf_id": hf, "name": m["name"], "slug": m.get("id") or ""})
    return out


def load() -> list[dict]:
    rows = parse(get_json(URL))
    if not rows:
        raise RuntimeError("OpenRouter returned no entries with a Hugging Face id")
    return rows
