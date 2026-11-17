"""Build public/data/models.json: one file holding every model, its size and its scores.

    python -m pipeline.build_data --out public/data/models.json

What it does, in order:
  1. Catalogue   - open-weight text models from models.dev (fatal if this fails).
  2. Names       - extra spellings from OpenRouter, for joining leaderboards by name.
  3. Sizes       - parameter counts from Hugging Face weights, data/overrides.yaml,
                   the model's name, or its description; active counts for MoE models.
  4. Scores      - models.dev benchmark tables, LMArena, BenchLM, Artificial Analysis, Aider,
                   and any Hugging Face benchmark leaderboards listed in the config.
  5. Resilience  - a source that fails keeps its scores from the previous build and
                   is flagged as stale on the site, so one outage never empties a column.

Environment: HF_TOKEN (optional), AA_API_KEY (optional), PREVIOUS_SNAPSHOT (paths or URLs, comma-separated).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import sys
import urllib.parse
import zlib

import yaml

from . import params as P
from .http import FetchError, get_json
from .names import is_date_token, looks_quantised, name_keys, norm_key, repo_name, size_from_name
from .sources import aider, artificial_analysis, benchlm, hf_hub, hf_leaderboards, lmarena, modelsdev, openrouter

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOW = dt.datetime.now(dt.timezone.utc)
TODAY = NOW.date().isoformat()

_UNIT = {"M": 1e6, "B": 1e9, "T": 1e12}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_size(value) -> float | None:
    """'37B' -> 37e9, '560M' -> 5.6e8, 1.2e9 -> 1.2e9."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([MBT])\s*", str(value), re.I)
    if not m:
        raise ValueError(f"Cannot read size {value!r}; write it like 37B, 560M or 1T")
    return float(m.group(1)) * _UNIT[m.group(2).upper()]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-")


# --------------------------------------------------------------------------
# Source bookkeeping
# --------------------------------------------------------------------------

SOURCE_META = {
    "modelsdev": {
        "name": "models.dev",
        "url": "https://models.dev",
        "licence": "MIT",
        "role": "Model list, release dates, licences, and benchmark scores with a link to where each was published.",
    },
    "hub": {
        "name": "Hugging Face Hub",
        "url": "https://huggingface.co/models",
        "licence": "Public repository metadata",
        "role": "Parameter counts read from the published weights, and the settings used to spot mixture-of-experts models.",
    },
    "openrouter": {
        "name": "OpenRouter",
        "url": "https://openrouter.ai/models",
        "licence": "Public model list",
        "role": "Alternative spellings of model names, used only to match leaderboard rows to models.",
    },
    "lmarena": {
        "name": "LMArena",
        "url": "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset",
        "licence": "CC BY 4.0",
        "role": "Elo-style ratings from blind human votes.",
    },
    "benchlm": {
        "name": "BenchLM",
        "url": "https://benchlm.ai",
        "licence": "CC BY-NC 4.0 (non-commercial). Data from BenchLM.ai",
        "role": "One composite score per model, blended by BenchLM from other publishers' results, plus category composites.",
    },
    "aa": {
        "name": "Artificial Analysis",
        "url": "https://artificialanalysis.ai",
        "licence": "Free API, attribution required",
        "role": "Intelligence Index and Coding Index.",
    },
    "aider": {
        "name": "Aider polyglot leaderboard",
        "url": "https://aider.chat/docs/leaderboards/",
        "licence": "Apache-2.0 repository",
        "role": "Coding benchmark across six languages. No new results since October 2025.",
    },
    "hf_leaderboards": {
        "name": "Hugging Face benchmark leaderboards",
        "url": "https://huggingface.co/datasets?other=benchmark",
        "licence": "Per benchmark dataset",
        "role": "Community-submitted evaluation results keyed by repository.",
    },
}


class Sources:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    def set(self, sid: str, status: str, detail: str = "", fetched_at: str | None = None):
        meta = SOURCE_META[sid]
        self.rows[sid] = {
            "id": sid,
            **meta,
            "status": status,  # ok | stale | skipped | failed
            "detail": detail,
            "fetched_at": fetched_at or (NOW.isoformat(timespec="seconds") if status == "ok" else None),
        }
        log(f"[{sid}] {status}: {detail}")

    def status(self, sid: str) -> str:
        return self.rows.get(sid, {}).get("status", "skipped")


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------


def primary_hf(hf_ids: list[str]) -> str | None:
    plain = [h for h in hf_ids if not looks_quantised(h)]
    return (plain or hf_ids or [None])[0]


def build_catalogue(md: dict) -> dict[str, dict]:
    labs = md["labs"]
    models: dict[str, dict] = {}
    pending = []  # entries with no weights link
    for e in md["models"]:
        hf = primary_hf(e["hf_ids"])
        if not hf:
            pending.append(e)
            continue
        mid = hf.lower()
        m = models.get(mid)
        if m is None:
            m = models[mid] = _new_model(mid, e, labs, hf)
        _absorb(m, e)
    for e in pending:
        # Fold "Command A" (no link) into "Command A" (with link) from the same lab.
        key = norm_key(e["name"])
        target = next((m for m in models.values() if m["lab"] == e["lab"] and key in m["keys"]), None)
        if target is None:
            mid = f"md:{e['md_id']}".lower()
            target = models.get(mid)
            if target is None:
                target = models[mid] = _new_model(mid, e, labs, None)
        _absorb(target, e)
    for m in models.values():
        _pick_name(m)
    return models


def _new_model(mid: str, e: dict, labs: dict, hf: str | None) -> dict:
    return {
        "id": mid,
        "lab": e["lab"],
        "org": labs.get(e["lab"]) or e["lab"].replace("-", " ").title(),
        "hf_id": hf,
        "url": f"https://huggingface.co/{hf}" if hf else f"https://models.dev/?search={urllib.parse.quote(e['name'])}",
        "names": [],
        "keys": set(),
        "spellings": set(),
        "md_ids": [],
        "descriptions": [],
        "released": None,
        "licence": None,
        "md_benchmarks": [],
        "scores": {},
    }


def _absorb(m: dict, e: dict) -> None:
    m["names"].append(e["name"])
    m["md_ids"].append(e["md_id"])
    m["spellings"].update([e["name"], e["md_id"].split("/", 1)[1]] + [repo_name(h) for h in e["hf_ids"]])
    m["keys"] = {norm_key(sp) for sp in m["spellings"]} - {""}
    if e["description"]:
        m["descriptions"].append(e["description"])
    if e["released"] and (m["released"] is None or e["released"] < m["released"]):
        m["released"] = e["released"]
    m["licence"] = m["licence"] or e["licence"]
    m["md_benchmarks"].extend(e["benchmarks"])


def _pick_name(m: dict) -> None:
    """Several catalogue entries can point at one set of weights; show the plainest name."""
    if m["hf_id"]:
        want = norm_key(repo_name(m["hf_id"]))
        exact = [n for n in m["names"] if norm_key(n) == want]
        if exact:
            m["name"] = exact[0]
            return
        if len(set(m["names"])) > 1:
            m["name"] = repo_name(m["hf_id"]).replace("-", " ").replace("_", " ")
            return
    m["name"] = m["names"][0]


def add_discovered(models: dict, found: list[dict], author: str, max_age_days: int | None) -> int:
    """Add repos the catalogue does not have. A repo that is plainly the weights of a
    catalogue entry with no weights link is attached to that entry instead."""
    cutoff = (NOW - dt.timedelta(days=max_age_days)).date().isoformat() if max_age_days else ""
    linked = {m["hf_id"].lower() for m in models.values() if m["hf_id"]}
    added = 0
    for f in found:
        rid = f["hf_id"]
        if rid.lower() in models or rid.lower() in linked or looks_quantised(rid):
            continue
        k = name_keys(repo_name(rid))
        same = [m for m in models.values() if k.loose in m["keys"] or any((n.base, n.quals) == (k.base, k.quals) for n in map(name_keys, m["spellings"]))]
        if same:
            same_maker = same[0]["lab"].replace("-", "")[:4] == author.lower().replace("-", "")[:4]
            if len(same) == 1 and not same[0]["hf_id"] and same_maker:
                m = same[0]
                m.update(hf_id=rid, url=f"https://huggingface.co/{rid}", hf_from_discovery=True)
                m["spellings"].add(repo_name(rid))
                m["keys"].add(k.loose)
                linked.add(rid.lower())
            continue  # already listed under another spelling
        if f.get("created") and f["created"] < cutoff:
            continue
        models[rid.lower()] = {
            "id": rid.lower(),
            "lab": author.lower(),
            "org": author,
            "hf_id": rid,
            "url": f"https://huggingface.co/{rid}",
            "names": [repo_name(rid)],
            "name": repo_name(rid).replace("_", " "),
            "keys": {k.loose},
            "spellings": {repo_name(rid)},
            "md_ids": [],
            "descriptions": [],
            "released": f.get("created"),
            "licence": None,
            "md_benchmarks": [],
            "scores": {},
            "discovered": True,
        }
        linked.add(rid.lower())
        added += 1
    return added


_TRAILING_PAREN = re.compile(r"\s*\([^)]*\)\s*$")


def add_unlisted(board, row: dict, source_name: str):
    """A leaderboard lists an open-weight model the catalogue lacks. Add it when the
    name states its size, so it can be placed under a parameter limit."""
    total, _ = size_from_name(row["name"])
    if not total:
        return None
    k = name_keys(row["name"])
    mid = f"lb:{k.loose}"
    m = board.models.get(mid)
    if m is None:
        # drop a trailing "(BF16)" or "(Thinking)" from the display name, keep "(0325)"
        shorter = _TRAILING_PAREN.sub("", row["name"]).strip()
        clean = shorter if shorter and name_keys(shorter).loose == k.loose else row["name"]
        licence = row.get("model_licence")
        m = board.models[mid] = {
            "id": mid,
            "lab": (row.get("org") or "").lower(),
            "org": (row.get("org") or "Unknown"),
            "hf_id": None,
            "url": row.get("url") or "",
            "names": [clean],
            "name": clean,
            "keys": {k.loose},
            "spellings": {row["name"], clean},
            "md_ids": [],
            "descriptions": [],
            "released": None,
            "licence": licence if licence and licence.lower() not in ("open weight", "open") else None,
            "md_benchmarks": [],
            "scores": {},
            "unlisted": source_name,
        }
        for sp in m["spellings"]:
            board.add_spelling_for(sp, mid)
    return m


# --------------------------------------------------------------------------
# Sizes
# --------------------------------------------------------------------------


def load_overrides(path: pathlib.Path) -> dict[str, dict]:
    out = {}
    for row in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
        entry = {
            "arch": row.get("arch"),
            "total": parse_size(row.get("total")),
            "active": parse_size(row.get("active")),
            "basis": row.get("basis", "stated"),
            "source": row.get("source"),
        }
        if entry["basis"] not in ("stated", "third_party"):
            raise ValueError(f"overrides.yaml: basis must be stated or third_party ({row.get('id')})")
        for key in [row["id"], *(row.get("also") or [])]:
            out[str(key).lower()] = entry
    return out


_DESC_SIZE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?(B|T|billion|trillion)\b(?=(.{0,30}))")
_DESC_ACTIVE = re.compile(r"\s*(parameters?\s+)?(active|activated)", re.I)
_DESC_MOE = re.compile(r"\bMoE\b|mixture[- ]of[- ]experts", re.I)
_DESC_UNIT = {"B": 1e9, "T": 1e12, "billion": 1e9, "trillion": 1e12}


def size_from_description(text: str) -> tuple[float | None, float | None, bool]:
    """Read '(594B total, 17B active)' or '26B MoE model with 3B active parameters'."""
    total = active = None
    for num, unit, tail in _DESC_SIZE.findall(text or ""):
        val = float(num) * _DESC_UNIT[unit]
        if _DESC_ACTIVE.match(tail):
            active = active or val
        else:
            total = total or val
    if total and active and active >= total:
        total = active = None
    return total, active, bool(_DESC_MOE.search(text or ""))


# Above this size a repo's config.json is always read: large open models are mostly
# mixture-of-experts. Smaller ones are read only when the model type hints at MoE.
CONFIG_ALWAYS_ABOVE = 15e9
_MOE_TYPE = re.compile(r"moe|mixtral|expert|deepseek|sparse", re.I)


def hub_block(m: dict, prev_hub: dict | None, recheck_days: int, breaker: dict) -> dict | None:
    """Parameter count and MoE layout from the Hub, reusing the previous build's answer when fresh."""
    if not m["hf_id"]:
        return None
    if prev_hub and prev_hub.get("checked") and prev_hub.get("total"):
        age = (NOW.date() - dt.date.fromisoformat(prev_hub["checked"])).days
        # spread the re-checks out so they do not all land on the same day
        if age < recheck_days + zlib.crc32(m["id"].encode()) % max(1, recheck_days // 2):
            breaker["cached"] += 1
            return prev_hub
    if breaker["open"]:
        return prev_hub
    try:
        info = hf_hub.model_info(m["hf_id"])
        breaker["fails"] = 0
    except FetchError as e:
        breaker["fails"] += 1
        breaker["last_error"] = str(e)
        if breaker["fails"] >= 5:
            breaker["open"] = True
        return prev_hub
    hf_hub.pause()
    block = {
        "total": info["total"],
        "created": info["created"],
        "licence": info["licence"],
        "gated": info["gated"],
        "checked": TODAY,
        "moe": None,
        "active_est": None,
    }
    if not info["total"] or info["total"] >= CONFIG_ALWAYS_ABOVE or _MOE_TYPE.search(info.get("model_type") or ""):
        cfg = hf_hub.config(m["hf_id"])
        hf_hub.pause()
        if cfg is not None:
            shape = P.moe_shape(cfg)
            block["moe"] = bool(shape)
            if shape and info["total"]:
                block["active_est"] = P.estimate_active(info["total"], cfg)
    breaker["fetched"] += 1
    return block


def resolve_sizes(m: dict, overrides: dict, hub: dict | None) -> None:
    ov = overrides.get(m["id"]) or next((overrides[f"md:{i}".lower()] for i in m["md_ids"] if f"md:{i}".lower() in overrides), None)
    stated = ov if ov and ov["basis"] == "stated" else None
    third = ov if ov and ov["basis"] == "third_party" else None

    name_total = name_active = None
    candidates = ([repo_name(m["hf_id"])] if m["hf_id"] else []) + m["names"]
    for c in candidates:
        t, a = size_from_name(c)
        name_total = name_total or t
        name_active = name_active or a
    desc_total = desc_active = None
    desc_moe = False
    for d in m["descriptions"]:
        t, a, moe = size_from_description(d)
        desc_total, desc_active, desc_moe = desc_total or t, desc_active or a, desc_moe or moe

    hub = hub or {}
    hub_total = hub.get("total")
    # Packed or quantised weights can make the Hub's count far too small (a "35B" model
    # reported as 665 thousand). Ignore a count more than 2x away from the size in the name.
    if hub_total and name_total and not (0.5 <= hub_total / name_total <= 2.0):
        hub_total = None
    total, tsrc = None, None
    for val, src in (
        (stated and stated["total"], "stated"),
        (hub_total, "hub"),
        (third and third["total"], "third_party"),
        (name_total, "name"),
        (desc_total, "description"),
    ):
        if val:
            total, tsrc = float(val), src
            break

    if ov and ov["arch"]:
        arch, asrc = ov["arch"], "override"
    elif name_active or desc_active or desc_moe:
        arch, asrc = "moe", "name" if name_active else "description"
    elif hub.get("moe") is not None:
        arch, asrc = ("moe" if hub["moe"] else "dense"), "config"
    else:
        arch, asrc = "dense", "assumed"

    active, acsrc = None, None
    if arch == "dense":
        active, acsrc = total, tsrc
    else:
        for val, src in (
            (stated and stated["active"], "stated"),
            (name_active, "name"),
            (desc_active, "description"),
            (third and third["active"], "third_party"),
            (hub.get("active_est") if hub_total else None, "estimate"),
        ):
            if val:
                active, acsrc = float(val), src
                break
        if active and total and active >= total:
            active, acsrc = None, None

    m.update(
        arch=arch,
        arch_src=asrc,
        params_total=total,
        params_total_src=tsrc,
        params_active=active,
        params_active_src=acsrc,
        params_note=(ov or {}).get("source"),
    )
    if not m["released"] and hub.get("created"):
        m["released"] = hub["created"]
    if not m["licence"] and hub.get("licence"):
        m["licence"] = hub["licence"]


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------


class Board:
    """All benchmark columns and the name index used to attach rows to models."""

    AMBIGUOUS = -2  # several catalogue models share the key and none is clearly meant
    NO_MATCH = -1

    def __init__(self, models: dict, notes: dict):
        self.models = models
        self.notes = notes
        self.benchmarks: dict[str, dict] = {}
        self.unmatched: dict[str, list[str]] = {}
        self.by_loose: dict[str, set[str]] = {}
        self.by_base: dict[str, set[str]] = {}
        self.by_unsized: dict[str, set[str]] = {}
        self.spellings: dict[str, list] = {}   # model id -> [Keys, ...]
        self.alias: dict[str, str] = {}
        for m in list(models.values()):
            for sp in m["spellings"]:
                self.add_spelling_for(sp, m["id"])

    def add_spelling_for(self, name: str, mid: str) -> None:
        k = name_keys(name)
        if not k.loose:
            return
        self.spellings.setdefault(mid, []).append(k)
        self.by_loose.setdefault(k.loose, set()).add(mid)
        self.by_base.setdefault(k.base, set()).add(mid)
        if not k.quals:  # only a plain spelling may stand in for a size-less name
            self.by_unsized.setdefault(k.unsized, set()).add(mid)

    def add_alias(self, name: str, target: str) -> bool:
        mid = target.lower()
        if mid not in self.models:
            return False
        k = name_keys(name)
        self.alias[k.strict] = mid  # explicit aliases win over everything
        self.alias[k.loose] = mid
        return True

    def add_spelling(self, name: str, hf_id: str) -> None:
        mid = hf_id.lower()
        if mid in self.models:
            self.add_spelling_for(name, mid)

    def find(self, name: str) -> tuple[dict | None, int]:
        """Return (model, tier). Tier 0 is an alias, 1 the same name, 2 the same name apart from a
        snapshot date or maker prefix, 3 the same name without its size. Lower is a closer match.
        With no model, tier says why: NO_MATCH or AMBIGUOUS."""
        k = name_keys(name)
        for key in (k.strict, k.loose):
            if key in self.alias:
                return self.models[self.alias[key]], 0
        # 1. same name
        ids = self.by_loose.get(k.loose)
        if ids:
            return self._one(ids, k, 1)
        # 2. same name once qualifiers are set aside - and the qualifiers must not disagree
        ids = {i for i in self.by_base.get(k.base, ()) if self._quals_fit(i, k)}
        if ids:
            return self._one(ids, k, 2)
        # 3. the row gives no size and exactly one plain catalogue entry is that name plus a size
        if not k.sized and not k.quals:
            ids = self.by_unsized.get(k.unsized)
            if ids:
                return self._one(ids, k, 3)
        return None, self.NO_MATCH

    def _quals_fit(self, mid: str, k) -> bool:
        for sp in self.spellings[mid]:
            if sp.base != k.base:
                continue
            if sp.quals == k.quals:
                return True  # differs only by a maker prefix
            # The row carries a snapshot date and the catalogue entry is the undated one:
            # accept when the date is the entry's release date (MMDD or YYMM).
            if not sp.quals and k.quals and all(is_date_token(q) for q in k.quals):
                rel = (self.models[mid].get("released") or "").replace("-", "")
                if not rel:
                    continue
                stamps = {rel[4:8], rel[2:6], rel[:6], rel[:8], rel[2:8]}
                if all(q in stamps for q in k.quals):
                    return True
        return False

    def _one(self, ids: set[str], k, tier: int) -> tuple[dict | None, int]:
        if len(ids) == 1:
            return self.models[next(iter(ids))], tier
        # Several models share the key. Pick one only when it is clearly the one meant:
        # the one spelled exactly like the row, word for word ...
        exact = [i for i in ids if any(sp.strict == k.strict for sp in self.spellings[i])]
        if len(exact) == 1:
            return self.models[exact[0]], tier
        # ... or a chat-tuned model beside its base model: leaderboards test the chat one.
        tuned = [i for i in ids if any(sp.instruct for sp in self.spellings[i])]
        if len(tuned) == 1 and len(ids) == 2:
            return self.models[tuned[0]], tier
        return None, self.AMBIGUOUS  # refuse to guess

    def register(self, bid: str, name: str, source_id: str, unit: str = "%", higher_is_better: bool = True) -> dict:
        b = self.benchmarks.get(bid)
        note = self.notes.get(bid, {})
        if b is None:
            b = self.benchmarks[bid] = {
                "id": bid,
                "name": name,
                "source_id": source_id,
                "unit": note.get("unit", unit),
                "higher_is_better": note.get("higher_is_better", higher_is_better),
                "description": note.get("description", ""),
                "url": note.get("url"),
                # "broad" = compares whole models (composites, human preference); "task" = one task set
                "group": note.get("group") or ("broad" if source_id in ("benchlm", "lmarena", "aa") else "task"),
                "short": note.get("short"),
            }
        return b

    def put(self, m: dict, bid: str, value: float, *, src: str | None, self_reported: bool, date: str | None,
            via: str, override: bool = False, label: str | None = None, tier: int = 1, extra: dict | None = None) -> None:
        cur = m["scores"].get(bid)
        hib = self.benchmarks[bid]["higher_is_better"]
        better = cur is None or (value > cur["v"] if hib else value < cur["v"])
        # A direct source replaces a relayed copy. Within one source a closer name match
        # wins; between equally close matches the best run is kept.
        if cur is None:
            take = True
        elif cur["via"] != via:
            take = override
        else:
            take = tier < cur.get("tier", 1) or (tier == cur.get("tier", 1) and better)
        if take:
            entry = {"v": round(value, 2), "self": bool(self_reported), "via": via}
            if src:
                entry["src"] = src
            if date:
                entry["date"] = date
            if label:
                entry["label"] = label
            if tier != 1:
                entry["tier"] = tier
            if extra:
                entry.update({k: v for k, v in extra.items() if v is not None})
            m["scores"][bid] = entry

    def miss(self, source_id: str, name: str) -> None:
        self.unmatched.setdefault(source_id, [])
        if name not in self.unmatched[source_id]:
            self.unmatched[source_id].append(name)


def host_of(url: str | None) -> str:
    try:
        return (urllib.parse.urlparse(url or "").hostname or "").lower()
    except ValueError:
        return ""


def add_modelsdev_scores(board: Board, cfg: dict) -> None:
    indep = tuple(cfg.get("independent_hosts") or [])
    merge = cfg.get("merge_into") or {}
    for m in board.models.values():
        for b in m["md_benchmarks"]:
            bid = merge.get(b["name"]) or f"md:{slug(b['name'])}"
            unit = "Elo" if (b.get("metric") or "").lower() == "elo" else "%"
            board.register(bid, b["name"], "modelsdev", unit=unit)
            host = host_of(b.get("source"))
            independent = any(host == h or host.endswith("." + h) for h in indep)
            board.put(m, bid, b["score"], src=b.get("source"), self_reported=not independent, date=b.get("date"), via="modelsdev")


def add_named_rows(board: Board, source_id: str, bid: str, rows: list[dict], *, src: str, open_only: bool = False,
                   unlisted: str | None = None) -> int:
    """Attach leaderboard rows to models by name. With `unlisted` set (the source's
    display name), open-weight rows the catalogue lacks become models of their own
    when the name states a size."""
    hits = set()
    for r in rows:
        licence = (r.get("model_licence") or "").lower()
        is_open = bool(licence) and "proprietary" not in licence
        m, tier = board.find(r["name"])
        if m is None and unlisted and is_open and tier != Board.AMBIGUOUS:
            m, tier = add_unlisted(board, r, unlisted), 1
        if m is None:
            if not open_only or is_open:
                board.miss(source_id, r["name"])
            continue
        label = r["name"] if norm_key(r["name"]) not in {norm_key(n) for n in m["names"]} else None
        board.put(m, bid, r["score"], src=r.get("url") or src, self_reported=False, date=r.get("date"), via=source_id,
                  override=True, label=label, tier=tier, extra=r.get("extra"))
        hits.add(m["id"])
    return len(hits)


# --------------------------------------------------------------------------
# Previous snapshot (cache + fallback)
# --------------------------------------------------------------------------


def load_previous(refs: str | None) -> dict | None:
    """Load the last published snapshot. `refs` is a comma-separated list of paths or URLs; the first that loads wins."""
    for ref in [r.strip() for r in (refs or "").split(",") if r.strip()]:
        try:
            if re.match(r"^https?://", ref):
                data = get_json(ref, timeout=30, retries=0)
            else:
                data = json.loads(pathlib.Path(ref).read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("models"), list):
                log(f"[previous] loaded {len(data['models'])} models from {ref}")
                return data
        except Exception as e:  # first deploy, 404, private Pages, bad JSON: all fine
            log(f"[previous] {ref} not used ({type(e).__name__}: {e})")
    return None


def carry_forward(board: Board, sources: Sources, prev: dict | None, sid: str) -> None:
    """Keep a failed source's scores from the previous build and mark the source stale."""
    if not prev:
        return
    prev_src = next((s for s in prev.get("sources", []) if s["id"] == sid), None)
    wanted = {b["id"]: b for b in prev.get("benchmarks", []) if b.get("source_id") == sid}
    if not wanted or not prev_src or not prev_src.get("fetched_at"):
        return
    kept = 0
    for pm in prev["models"]:
        m = board.models.get(pm["id"])
        if not m:
            continue
        for bid, entry in (pm.get("scores") or {}).items():
            if bid in wanted and entry.get("via") == sid:
                pb = wanted[bid]
                board.register(bid, pb["name"], sid, unit=pb.get("unit", "%"), higher_is_better=pb.get("higher_is_better", True))
                board.benchmarks[bid].update(source_id=sid, name=pb["name"])
                cur = m["scores"].get(bid)
                if cur is None or cur.get("via") != sid:
                    m["scores"][bid] = entry
                    kept += 1
    if kept:
        was = sources.rows[sid]["detail"]
        sources.set(sid, "stale", f"Could not refresh ({was}); showing {kept} scores from the last good build.", fetched_at=prev_src["fetched_at"])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="public/data/models.json")
    ap.add_argument("--modelsdev-dir", default=os.environ.get("MODELSDEV_DIR"), help="Use a local checkout of sst/models.dev instead of downloading it")
    ap.add_argument("--previous", default=os.environ.get("PREVIOUS_SNAPSHOT"), help="Comma-separated paths or URLs of the last published models.json; the first that loads is used")
    ap.add_argument("--skip", default=os.environ.get("SKIP_SOURCES", ""), help="Comma-separated source ids to skip: hub,openrouter,lmarena,benchlm,aa,aider,hf_leaderboards")
    args = ap.parse_args(argv)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    cfg = yaml.safe_load((ROOT / "data" / "config.yaml").read_text(encoding="utf-8"))
    overrides = load_overrides(ROOT / "data" / "overrides.yaml")
    aliases = yaml.safe_load((ROOT / "data" / "aliases.yaml").read_text(encoding="utf-8")) or {}
    prev = load_previous(args.previous)
    prev_models = {pm["id"]: pm for pm in (prev or {}).get("models", [])}
    sources = Sources()

    # 1. Catalogue ------------------------------------------------------------
    try:
        md = modelsdev.load(args.modelsdev_dir)
    except Exception as e:
        log(f"FATAL: the model catalogue could not be read ({type(e).__name__}: {e}). Keeping the published site as it is.")
        return 2
    models = build_catalogue(md)
    sources.set("modelsdev", "ok", f"{len(models)} open-weight text models from {md['files_seen']} catalogue entries.")

    # 2. Hub discovery + sizes --------------------------------------------------
    hub_cfg = cfg.get("hub") or {}
    breaker = {"fails": 0, "open": "hub" in skip, "fetched": 0, "cached": 0, "last_error": ""}
    discovered = 0
    if not breaker["open"]:
        for org in hub_cfg.get("discover_orgs") or []:
            try:
                found = hf_hub.discover(org, int(hub_cfg.get("discover_limit", 15)))
                breaker["fails"] = 0
                try:  # the newest repos too, so a release shows up before it has downloads
                    hf_hub.pause()
                    found += hf_hub.discover(org, int(hub_cfg.get("discover_limit", 15)), sort="createdAt")
                except FetchError:
                    pass
            except FetchError as e:
                breaker["fails"] += 1
                breaker["last_error"] = str(e)
                if breaker["fails"] >= 3:
                    breaker["open"] = True
                    break
                continue
            discovered += add_discovered(models, found, org, int(hub_cfg.get("discover_max_age_days", 730)))
            hf_hub.pause()
    # Models found by an earlier build stay listed even if discovery fails today.
    for pid, pm in prev_models.items():
        if pm.get("discovered") and pid not in models and pm.get("hf_id"):
            add_discovered(models, [{"hf_id": pm["hf_id"], "created": pm.get("released")}], pm.get("org") or pm["hf_id"].split("/")[0], None)

    for m in models.values():
        prev_hub = (prev_models.get(m["id"]) or {}).get("hub")
        m["hub"] = hub_block(m, prev_hub, int(hub_cfg.get("recheck_days", 30)), breaker)
        resolve_sizes(m, overrides, m["hub"])
    if "hub" in skip:
        sources.set("hub", "skipped", f"Not re-read in this build; {breaker['cached']} sizes reused from the last build.")
    elif breaker["fetched"] or breaker["cached"]:
        note = f" Stopped early: {breaker['last_error']}" if breaker["open"] else ""
        sources.set("hub", "ok", f"{breaker['fetched']} repositories read, {breaker['cached']} reused from the last build, {discovered} added by discovery.{note}")
    else:
        sources.set("hub", "failed", f"Not reachable ({breaker['last_error'] or 'no repositories to read'}); sizes come from the override file, model names and descriptions.")

    # 3. Name index -------------------------------------------------------------
    board = Board(models, cfg.get("benchmark_notes") or {})
    if "openrouter" in skip:
        sources.set("openrouter", "skipped", "Skipped for this build.")
    else:
        try:
            rows = openrouter.load()
            for r in rows:
                board.add_spelling(r["name"], r["hf_id"])
                if r.get("slug"):
                    board.add_spelling(r["slug"].split("/", 1)[-1].split(":")[0], r["hf_id"])
            sources.set("openrouter", "ok", f"{len(rows)} name spellings read.")
        except Exception as e:
            sources.set("openrouter", "failed", f"{type(e).__name__}: {e}")
    bad_aliases = [name for name, target in aliases.items() if not board.add_alias(name, str(target))]
    if bad_aliases:
        log(f"[aliases] {len(bad_aliases)} aliases point at models not in the catalogue: {bad_aliases}")

    # 4. Scores -----------------------------------------------------------------
    add_modelsdev_scores(board, cfg.get("modelsdev") or {})

    def run(sid: str, fn) -> None:
        if sid in skip:
            sources.set(sid, "skipped", "Skipped for this build.")
        else:
            try:
                sources.set(sid, "ok", fn())
            except LookupError as e:  # missing key: configuration, not an outage
                sources.set(sid, "skipped", f"{e}. Add it as a CI/CD variable to switch this source on.")
            except Exception as e:
                sources.set(sid, "failed", f"{type(e).__name__}: {e}")
        if sources.status(sid) != "ok":
            carry_forward(board, sources, prev, sid)

    def do_aider() -> str:
        rows = aider.load()
        board.register("aider:polyglot", "Aider polyglot", "aider")
        board.benchmarks["aider:polyglot"].update(source_id="aider", name="Aider polyglot")
        hits = add_named_rows(board, "aider", "aider:polyglot", rows, src="https://aider.chat/docs/leaderboards/")
        return f"{len(rows)} rows read, {hits} matched to open-weight models."

    def do_lmarena() -> str:
        parts = []
        for b in (cfg.get("lmarena") or {}).get("boards") or []:
            rows = lmarena.load(b["config"], b.get("category", "overall"))
            board.register(b["id"], b["name"], "lmarena", unit="Elo")
            hits = add_named_rows(board, "lmarena", b["id"], rows, src="https://lmarena.ai/leaderboard", open_only=True,
                                  unlisted="LMArena" if (cfg.get("lmarena") or {}).get("add_unlisted", True) else None)
            parts.append(f"{b['name']}: {len(rows)} rows, {hits} matched")
        return "; ".join(parts) + "."

    def do_aa() -> str:
        rows = artificial_analysis.load()
        names = {"intelligence": "Artificial Analysis Intelligence Index", "coding": "Artificial Analysis Coding Index"}
        hits = 0
        for short, title in names.items():
            bid = f"aa:{short}"
            sub = [{"name": r["name"], "score": r["scores"][short], "date": None} for r in rows if short in r["scores"]]
            if not sub:
                continue
            board.register(bid, title, "aa", unit="index")
            board.benchmarks[bid].update(source_id="aa", name=title)
            hits += add_named_rows(board, "aa", bid, sub, src="https://artificialanalysis.ai/leaderboards/models")
        return f"{len(rows)} models read, {hits} scores matched to open-weight models."

    def do_benchlm() -> str:
        bl_cfg = cfg.get("benchlm") or {}
        data = benchlm.load()
        unlisted = "BenchLM" if bl_cfg.get("add_unlisted", True) else None
        boards = [("benchlm:overall", "BenchLM overall", data["overall"])]
        for cat in bl_cfg.get("categories") or []:
            if data["categories"].get(cat):
                label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cat).lower()
                boards.append((f"benchlm:{cat.lower()}", f"BenchLM {label}", data["categories"][cat]))
        parts = []
        for bid, title, rows in boards:
            for r in rows:
                r["extra"] = {"est": True if r.get("est") else None, "lo": r.get("lo"), "hi": r.get("hi")}
                r["date"] = data.get("updated")
            board.register(bid, title, "benchlm", unit="index")
            hits = add_named_rows(board, "benchlm", bid, rows, src="https://benchlm.ai", open_only=True,
                                  unlisted=unlisted if bid == "benchlm:overall" else None)
            open_rows = sum(1 for r in rows if "open" in (r.get("model_licence") or "").lower())
            parts.append(f"{title}: {len(rows)} rows ({open_rows} open-weight), {hits} matched")
        return "; ".join(parts) + "."

    def do_hf_leaderboards() -> str:
        parts = []
        for b in cfg.get("hf_leaderboards") or []:
            rows = hf_leaderboards.load(b["dataset"])
            board.register(b["id"], b["name"], "hf_leaderboards", unit=b.get("unit", "%"))
            hits = 0
            for r in rows:
                m = models.get(r["hf_id"].lower())
                if m:
                    board.put(m, b["id"], r["score"], src=r.get("source") or f"https://huggingface.co/datasets/{b['dataset']}",
                              self_reported=not r["verified"], date=r.get("date"), via="hf_leaderboards", override=True)
                    hits += 1
            parts.append(f"{b['name']}: {len(rows)} rows, {hits} matched")
        return "; ".join(parts) + "."

    run("aider", do_aider)
    run("lmarena", do_lmarena)
    run("benchlm", do_benchlm)
    run("aa", do_aa)
    for m in models.values():  # models added from a leaderboard get their size from the name
        if "arch" not in m:
            resolve_sizes(m, overrides, None)
    if cfg.get("hf_leaderboards"):
        run("hf_leaderboards", do_hf_leaderboards)

    # 5. Assemble -----------------------------------------------------------------
    min_models = int((cfg.get("modelsdev") or {}).get("min_models", 6))
    benchmarks = []
    for bid, b in board.benchmarks.items():
        entries = [m["scores"][bid] for m in models.values() if bid in m["scores"]]
        sized = [m for m in models.values() if bid in m["scores"] and m.get("params_total")]
        if not entries or (b["source_id"] == "modelsdev" and len(entries) < min_models):
            for m in models.values():
                m["scores"].pop(bid, None)
            continue
        dates = [e["date"] for e in entries if e.get("date")]
        b.update(count=len(entries), sized_count=len(sized), latest=max(dates) if dates else None,
                 self_count=sum(1 for e in entries if e.get("self")), est_count=sum(1 for e in entries if e.get("est")))
        benchmarks.append(b)
    # whole-model comparisons first, then by how many sized models each covers
    benchmarks.sort(key=lambda b: (b.get("group") != "broad", -b["sized_count"], b["name"].lower()))

    default = next((bid for bid in cfg.get("default_benchmark") or [] if any(b["id"] == bid for b in benchmarks)), None)
    if default is None and benchmarks:
        default = benchmarks[0]["id"]

    out_models = []
    for m in sorted(models.values(), key=lambda m: m["name"].lower()):
        row = {
            "id": m["id"],
            "name": m["name"],
            "org": m["org"],
            "hf_id": m["hf_id"],
            "url": m["url"],
            "released": m["released"],
            "licence": m["licence"],
            "arch": m["arch"],
            "arch_src": m["arch_src"],
            "params_total": m["params_total"],
            "params_total_src": m["params_total_src"],
            "params_active": m["params_active"],
            "params_active_src": m["params_active_src"],
            "scores": m["scores"],
        }
        if m.get("params_note"):
            row["params_note"] = m["params_note"]
        if m.get("hub"):
            row["hub"] = m["hub"]
        if m.get("discovered"):
            row["discovered"] = True
        if m.get("unlisted"):
            row["unlisted"] = m["unlisted"]
        out_models.append(row)

    sized = sum(1 for m in out_models if m["params_total"])
    snapshot = {
        "generated_at": NOW.isoformat(timespec="seconds"),
        "default_benchmark": default,
        "counts": {
            "models": len(out_models),
            "with_size": sized,
            "moe": sum(1 for m in out_models if m["arch"] == "moe"),
            "moe_without_active": sum(1 for m in out_models if m["arch"] == "moe" and not m["params_active"]),
            "with_any_score": sum(1 for m in out_models if m["scores"]),
            "from_leaderboards_only": sum(1 for m in out_models if m.get("unlisted")),
        },
        "sources": list(sources.rows.values()),
        "benchmarks": benchmarks,
        "models": out_models,
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (out.parent / "unmatched.json").write_text(
        json.dumps({"generated_at": snapshot["generated_at"], "unmatched": board.unmatched}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    log(
        f"Wrote {out}: {len(out_models)} models ({sized} with a size, {snapshot['counts']['with_any_score']} with a score), "
        f"{len(benchmarks)} benchmarks. Unmatched rows: "
        + (", ".join(f"{k} {len(v)}" for k, v in board.unmatched.items()) or "none")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
