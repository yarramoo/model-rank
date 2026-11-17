"""LMArena (formerly Chatbot Arena) - human-preference Elo ratings.

Published as the dataset lmarena-ai/leaderboard-dataset on Hugging Face under
CC-BY-4.0. One subset per arena (text, webdev, vision, ...), each with a
`latest` split. Rows: model_name, organization, license, rating, rank,
category, leaderboard_publish_date. Read through the dataset viewer's JSON API
so the build needs no parquet library.
"""
from __future__ import annotations

import os
import urllib.parse

from ..http import get_json

ROWS = "https://datasets-server.huggingface.co/rows"
DATASET = "lmarena-ai/leaderboard-dataset"


def load(config: str, category: str = "overall", max_rows: int = 3000) -> list[dict]:
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    out, offset, total = [], 0, None
    while total is None or offset < min(total, max_rows):
        q = urllib.parse.urlencode({"dataset": DATASET, "config": config, "split": "latest", "offset": offset, "length": 100})
        page = get_json(f"{ROWS}?{q}", headers=headers, timeout=60)
        rows = page.get("rows") or []
        total = page.get("num_rows_total", len(rows))
        if not rows:
            break
        for item in rows:
            r = item.get("row") or {}
            if category and (r.get("category") or "overall") != category:
                continue
            score = r.get("rating")
            if score is None:
                score = r.get("score")
            if not r.get("model_name") or not isinstance(score, (int, float)):
                continue
            out.append(
                {
                    "name": str(r["model_name"]),
                    "org": r.get("organization"),
                    "model_licence": r.get("license"),
                    "score": float(score),
                    "date": str(r.get("leaderboard_publish_date") or "")[:10] or None,
                }
            )
        offset += 100
    if not out:
        raise RuntimeError(f"LMArena '{config}' returned no rows for category '{category}'")
    # A dataset split can hold several publish dates; keep only the newest board.
    newest = max((r["date"] or "") for r in out)
    return [r for r in out if (r["date"] or "") == newest]
