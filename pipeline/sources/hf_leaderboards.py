"""Hugging Face benchmark leaderboards (the 2026 "eval results" feature).

GET /api/datasets/{benchmark_dataset}/leaderboard lists results keyed by Hugging
Face repo id, so they join to the catalogue with no name matching at all. The
feature is in beta, so this reader accepts a few response shapes and the build
carries on if it fails. List the benchmark datasets to read in data/config.yaml.
"""
from __future__ import annotations

import os
import urllib.parse

from ..http import get_json


def load(dataset_id: str) -> list[dict]:
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://huggingface.co/api/datasets/{urllib.parse.quote(dataset_id, safe='/')}/leaderboard"
    payload = get_json(url, headers=headers, timeout=60)
    rows = payload
    if isinstance(payload, dict):
        rows = payload.get("entries") or payload.get("results") or payload.get("leaderboard") or payload.get("rows") or []
    out = []
    for r in rows if isinstance(rows, list) else []:
        rid = r.get("model_id") or r.get("modelId") or r.get("model")
        val = r.get("value") if r.get("value") is not None else r.get("score")
        if isinstance(val, dict):
            val = val.get("value")
        if not isinstance(rid, str) or "/" not in rid or not isinstance(val, (int, float)):
            continue
        src = r.get("source")
        out.append(
            {
                "hf_id": rid,
                "score": float(val),
                "date": str(r.get("date") or "")[:10] or None,
                "verified": bool(r.get("verified")),
                "source": src.get("url") if isinstance(src, dict) else (src if isinstance(src, str) else None),
            }
        )
    if not out:
        raise RuntimeError(f"No usable rows in the leaderboard for {dataset_id}")
    return out
