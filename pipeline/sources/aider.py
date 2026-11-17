"""Aider polyglot coding leaderboard (Apache-2.0 repo, YAML on GitHub).

Independent harness: 225 Exercism exercises across six languages, score is the
share solved within two attempts. Results stopped arriving in October 2025, so
this is a frozen source - kept because it covers a useful set of 2025 open
models, and labelled with its latest result date on the site.
"""
from __future__ import annotations

import yaml

from ..http import get_text

URL = "https://raw.githubusercontent.com/Aider-AI/aider/main/aider/website/_data/polyglot_leaderboard.yml"


def load() -> list[dict]:
    rows = yaml.safe_load(get_text(URL))
    out = []
    for r in rows or []:
        score = r.get("pass_rate_2")
        name = r.get("model")
        if not name or not isinstance(score, (int, float)):
            continue
        out.append({"name": str(name), "score": float(score), "date": str(r.get("date") or "") or None})
    if len(out) < 10:
        raise RuntimeError(f"Aider leaderboard looked wrong: {len(out)} rows")
    return out
