"""models.dev - the backbone catalogue.

An MIT-licensed, community-maintained catalogue (github.com/sst/models.dev) with
one TOML file per model under models/<lab>/<model>.toml. For each model it
records whether the weights are open, the release date, the licence, a link to
the weights, and benchmark scores with the URL each score came from.

We read the repository tarball rather than models.dev/api.json because the API
output is organised per hosting provider and drops the weights and benchmark
tables.
"""
from __future__ import annotations

import io
import pathlib
import re
import tarfile
import tomllib

from ..http import get_bytes

TARBALL = "https://github.com/sst/models.dev/archive/refs/heads/dev.tar.gz"
_HF = re.compile(r"https?://huggingface\.co/([^/\s]+/[^/?#\s]+)")
_MEMBER = re.compile(r"^[^/]+/(models|labs)/([^/]+)/([^/]+)\.toml$")


def _iter_files(local_dir: str | None):
    """Yield (kind, lab, stem, text) for every lab model or lab metadata file."""
    if local_dir:
        root = pathlib.Path(local_dir)
        for kind in ("models", "labs"):
            for p in sorted((root / kind).glob("*/*.toml")):
                yield kind, p.parent.name, p.stem, p.read_text(encoding="utf-8")
        return
    raw = get_bytes(TARBALL, timeout=120)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar:
            m = _MEMBER.match(member.name)
            if not m or not member.isfile():
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            yield m.group(1), m.group(2), m.group(3), f.read().decode("utf-8", "replace")


def load(local_dir: str | None = None) -> dict:
    """Return {'models': [...], 'labs': {lab: display name}} for open-weight text models."""
    models, labs = [], {}
    seen = 0
    for kind, lab, stem, text in _iter_files(local_dir):
        try:
            d = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            continue
        if kind == "labs":
            if stem == "lab" and d.get("name"):
                labs[lab] = d["name"]
            continue
        seen += 1
        if not d.get("open_weights"):
            continue
        if d.get("type") not in (None, "chat"):
            continue  # transcription, embedding, decision models etc.
        if "text" not in ((d.get("modalities") or {}).get("output") or []):
            continue
        hf_ids = []
        for w in d.get("weights") or []:
            m = _HF.match(w.get("url", ""))
            if m:
                hf_ids.append(m.group(1))
        benchmarks = []
        for b in d.get("benchmarks") or []:
            if isinstance(b.get("score"), (int, float)) and b.get("name"):
                benchmarks.append(
                    {
                        "name": str(b["name"]).strip(),
                        "score": float(b["score"]),
                        "metric": b.get("metric"),
                        "source": b.get("source"),
                        "date": str(b["date"]) if b.get("date") else None,
                    }
                )
        models.append(
            {
                "md_id": f"{lab}/{stem}",
                "lab": lab,
                "name": d.get("name") or stem,
                "description": d.get("description") or "",
                "released": str(d.get("release_date") or "") or None,
                "licence": d.get("license"),
                "status": d.get("status"),
                "hf_ids": hf_ids,
                "benchmarks": benchmarks,
            }
        )
    if seen < 50:
        raise RuntimeError(f"models.dev looked wrong: only {seen} model files found")
    return {"models": models, "labs": labs, "files_seen": seen}
