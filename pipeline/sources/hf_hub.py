"""Hugging Face Hub - parameter counts and MoE layout, straight from the weights.

GET /api/models/{repo} returns, among other things:
    safetensors: {"parameters": {"BF16": 30532122624}, "total": 30532122624}
    createdAt, tags, cardData.license, gated
and /{repo}/resolve/main/config.json holds the architecture settings used to
detect mixture-of-experts models and estimate their active parameters.

Public repos need no token; set HF_TOKEN (read-only) to get higher rate limits
and to read config.json from gated repos you have accepted.
"""
from __future__ import annotations

import os
import time
import urllib.parse

from ..http import FetchError, get_json

API = "https://huggingface.co/api/models"


def _headers() -> dict:
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def pause() -> None:
    """Stay under the Hub's request limits (lower for anonymous callers)."""
    time.sleep(0.3 if os.environ.get("HF_TOKEN") else 0.65)


def model_info(repo: str) -> dict:
    """Return {'total', 'created', 'licence', 'tags', 'gated', 'model_type'} for one repo. Raises FetchError."""
    d = get_json(f"{API}/{urllib.parse.quote(repo, safe='/')}", headers=_headers(), timeout=40, retries=1)
    st = d.get("safetensors") or {}
    total = st.get("total")
    if not total and isinstance(st.get("parameters"), dict):
        total = sum(v for v in st["parameters"].values() if isinstance(v, (int, float)))
    if not total:
        total = (d.get("gguf") or {}).get("total")
    card = d.get("cardData") or {}
    licence = card.get("license")
    if isinstance(licence, list):
        licence = licence[0] if licence else None
    cfg = d.get("config") if isinstance(d.get("config"), dict) else {}
    return {
        "model_type": cfg.get("model_type") or " ".join(cfg.get("architectures") or []),
        "total": int(total) if total else None,
        "created": (d.get("createdAt") or "")[:10] or None,
        "licence": licence,
        "tags": d.get("tags") or [],
        "gated": bool(d.get("gated")),
    }


def config(repo: str) -> dict | None:
    try:
        d = get_json(
            f"https://huggingface.co/{urllib.parse.quote(repo, safe='/')}/resolve/main/config.json",
            headers=_headers(),
            timeout=40,
            retries=1,
        )
        return d if isinstance(d, dict) else None
    except FetchError:
        return None


def discover(author: str, limit: int = 40, sort: str = "downloads") -> list[dict]:
    """Most-downloaded text-generation repos published by one organisation.

    Finds releases that models.dev has not catalogued yet (it leans towards models
    served by API providers, so small local models can be missing).
    """
    q = urllib.parse.urlencode(
        {"author": author, "pipeline_tag": "text-generation", "sort": sort, "direction": "-1", "limit": str(limit)}
    )
    rows = get_json(f"{API}?{q}", headers=_headers(), timeout=40, retries=1)
    out = []
    for d in rows if isinstance(rows, list) else []:
        rid = d.get("id") or d.get("modelId")
        if not rid:
            continue
        tags = d.get("tags") or []
        # Skip derivatives: quantisations, fine-tunes, adapters and merges of another repo.
        if any(t.startswith(("base_model:quantized:", "base_model:finetune:", "base_model:adapter:", "base_model:merge:")) for t in tags):
            continue
        out.append({"hf_id": rid, "created": (d.get("createdAt") or "")[:10] or None, "tags": tags})
    return out
