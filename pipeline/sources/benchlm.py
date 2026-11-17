"""BenchLM - one composite score per model, plus category composites.

benchlm.ai publishes its leaderboards as static JSON under CC BY-NC 4.0
(non-commercial use, credit "Data from BenchLM.ai" with a link - the site footer
carries it). leaderboard.json holds:

    items:       the overall board; each row has model, creator, sourceType
                 ("Open Weight" / "Proprietary"), score, evidenceStatus
                 ("supported" / "estimated"), scoreInterval90 {lower, upper}
    categories:  {coding: [...], agentic: [...], ...} with the same row shape,
                 where `score` is the category score

The overall score is BenchLM's own weighted blend of published benchmark
results across its categories. Rows marked "estimated" rest on thinner evidence
and are shown with a tilde on the site.
"""
from __future__ import annotations

from ..http import get_json

URL = "https://benchlm.ai/data/leaderboard.json"


def _rows(raw, category: str | None) -> list[dict]:
    out = []
    for r in raw if isinstance(raw, list) else []:
        if not isinstance(r, dict):
            continue
        score = r.get("score")
        if not isinstance(score, (int, float)):
            score = r.get("displayScore")
        name = r.get("model")
        if not name or not isinstance(score, (int, float)) or r.get("rankingEligible") is False:
            continue
        interval = r.get("scoreInterval90") if isinstance(r.get("scoreInterval90"), dict) else {}
        out.append(
            {
                "name": str(name),
                "org": r.get("creator"),
                "model_licence": r.get("sourceType"),
                "score": float(score),
                "est": (r.get("evidenceStatus") or "").lower() == "estimated",
                "lo": interval.get("lower"),
                "hi": interval.get("upper"),
                "category": category,
                "label": r.get("categoryLabel"),
                "url": r.get("url"),
            }
        )
    return out


def parse(payload: dict) -> dict:
    """Return {'overall': [...], 'categories': {name: [...]}, 'updated': str | None}."""
    cats = payload.get("categories")
    categories = {}
    if isinstance(cats, dict):
        categories = {k: _rows(v, k) for k, v in cats.items()}
    elif isinstance(cats, list):  # tolerate [{category, items}] should the export change shape
        for c in cats:
            if isinstance(c, dict) and c.get("category"):
                categories[c["category"]] = _rows(c.get("items"), c["category"])
    return {
        "overall": _rows(payload.get("items"), None),
        "categories": {k: v for k, v in categories.items() if v},
        "updated": (payload.get("generatedAt") or "")[:10] or None,
    }


def load() -> dict:
    data = parse(get_json(URL, timeout=120))
    if len(data["overall"]) < 20:
        raise RuntimeError(f"BenchLM leaderboard looked wrong: {len(data['overall'])} overall rows")
    return data
