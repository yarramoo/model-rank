"""Artificial Analysis - Intelligence Index and Coding Index.

Free API, key required (create one at artificialanalysis.ai, pass it as the
AA_API_KEY CI/CD variable). 100 requests per day on the free tier; this build
makes one. Their terms require visible attribution with a link, which the site
footer and the benchmark note provide. The free response has no open-weights
flag or parameter count, so rows are joined to the catalogue by name.
"""
from __future__ import annotations

import os

from ..http import get_json

# Overridable because the path changed between API versions.
URL = os.environ.get("AA_API_URL", "https://artificialanalysis.ai/api/v2/language/models/free")

INDICES = {
    "artificial_analysis_intelligence_index": "intelligence",
    "artificial_analysis_coding_index": "coding",
}


def load() -> list[dict]:
    key = os.environ.get("AA_API_KEY")
    if not key:
        raise LookupError("AA_API_KEY is not set")
    payload = get_json(URL, headers={"x-api-key": key}, timeout=60)
    rows = payload.get("data") if isinstance(payload, dict) else payload
    out = []
    for r in rows or []:
        ev = r.get("evaluations") or {}
        scores = {short: float(ev[k]) for k, short in INDICES.items() if isinstance(ev.get(k), (int, float))}
        if not r.get("name") or not scores:
            continue
        creator = r.get("model_creator") or {}
        out.append(
            {
                "name": str(r["name"]),
                "slug": r.get("slug"),
                "org": creator.get("name") if isinstance(creator, dict) else None,
                "scores": scores,
                "date": str(r.get("release_date") or "") or None,
            }
        )
    if not out:
        raise RuntimeError("Artificial Analysis returned no rows with an index score")
    return out
