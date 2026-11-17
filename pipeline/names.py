"""Model-name handling: join keys across sources and size tokens in names.

Sources name the same model differently ("Qwen3 30B A3B", "qwen/qwen3-30b-a3b",
"Qwen/Qwen3-30B-A3B-Instruct", "Qwen: Qwen3 30B A3B (free)"). Every name is
reduced to a few keys:

    strict    every word kept                        deepseekv4.1flash20260910max
    loose     run modes, "Instruct" and precision     deepseekv4.1flash0910
              labels dropped; dates folded to MMDD
    base      loose without its qualifiers            deepseekv4.1flash
    quals     the qualifiers that were set aside:     {"0910"}
              snapshot dates, "v0.2", "Preview"
    unsized   base without parameter sizes

A leaderboard row is matched at the tightest level that finds a catalogue model
(Board.find in build_data.py). Keys must be equal - there is no fuzzy matching -
and qualifiers may never disagree: "v0.2" is not "v0.3", a preview is not the
release, and a date stamp has to fit the model's release date. Anything else
goes to the unmatched report, to be fixed with one line in data/aliases.yaml.
"""
from __future__ import annotations

import re
from typing import NamedTuple

# Words that do not change which weights are meant.
_NOISE = {"instruct", "it", "chat", "hf", "free", "model", "the"}
# Chat-tuned variants: preferred over a base model when a row could mean either.
_INSTRUCT = {"instruct", "it", "chat"}
# Precision / packaging labels.
_QUANT_TOKENS = {"bf16", "fp16", "fp8", "fp4", "nvfp4", "mxfp8", "int4", "int8", "awq", "gguf", "gptq", "mlx"}
# Dropped at the "undated" tier.
_SOFT = {"preview", "latest", "exp", "experimental", "beta"}
_ORG_PREFIX = {"nvidia", "ibm", "meta", "google", "microsoft", "c4ai", "ai21"}

_PAREN = re.compile(r"\(([^)]*)\)")
_SPLIT = re.compile(r"[^a-z0-9.]+")
# Parenthetical run-mode labels describe how a model was run, not which weights.
_MODE = re.compile(
    r"^(non[- ]?)?(thinking|think|reasoning|no thinking|no think|max|xhigh|high|medium|low|minimal|default|extended|free|"
    r"exacto|nitro|standard|fast|instant|default fallback|adaptive|\d+k( thinking| think)?( tokens)?)$"
)
_DATE8 = re.compile(r"^20\d{2}(\d{4})$")
_DATEISH = re.compile(r"^(\d{4}|\d{6}|\d{8})$")
# "v0.2" style checkpoint versions. Not "V4.1" or "V3": those are the model's name.
_VERSION = re.compile(r"^v0(\.\d+)+$")
_SIZE_TOKEN = re.compile(r"^(a?\d+(\.\d+)?[bmt]|\d+x\d+(\.\d+)?b|e\d+b|\d+e)$")


class Keys(NamedTuple):
    strict: str
    loose: str
    base: str
    quals: frozenset
    unsized: str
    sized: bool
    instruct: bool


def _split(text: str) -> list[str]:
    return [t.strip(".") for t in _SPLIT.split(text) if t.strip(".")]


def is_date_token(tok: str) -> bool:
    return bool(_DATEISH.match(tok))


def name_keys(name: str) -> Keys:
    s = name.lower()
    if ": " in s:  # OpenRouter style "Maker: Model"
        s = s.split(": ", 1)[1]
    s = s.replace("+", " plus ").replace("/", " ")

    everything: list[str] = []  # strict
    kept: list[str] = []        # run-mode parentheticals removed
    pos = 0
    for m in _PAREN.finditer(s):
        outside = _split(s[pos:m.start()])
        everything += outside
        kept += outside
        for part in m.group(1).split(","):
            toks = _split(part)
            everything += toks
            if not _MODE.match(part.strip()):
                kept += toks
        pos = m.end()
    tail = _split(s[pos:])
    everything += tail
    kept += tail

    loose = []
    for t in kept:
        if t in _NOISE or t in _QUANT_TOKENS:
            continue
        d = _DATE8.match(t)
        loose.append(d.group(1) if d else t)
    quals = frozenset(t for t in loose if _DATEISH.match(t) or t in _SOFT or _VERSION.match(t))
    base = [t for t in loose if t not in quals]
    if len(base) > 1 and base[0] in _ORG_PREFIX:
        base = base[1:]
    unsized = [t for t in base if not _SIZE_TOKEN.match(t)]
    return Keys("".join(everything), "".join(loose), "".join(base), quals, "".join(unsized),
                len(unsized) != len(base), bool(_INSTRUCT & set(everything)))


def norm_key(name: str) -> str:
    """The loose key: 'Qwen: Qwen3 30B A3B (free)' -> 'qwen330ba3b'."""
    return name_keys(name).loose


def repo_name(hf_id: str) -> str:
    return hf_id.split("/", 1)[1] if "/" in hf_id else hf_id


_UNIT = {"m": 1e6, "b": 1e9, "t": 1e12}
# "30B", "2.4T", "560M" as a standalone token (not "8x7B", not "E2B", not "v2B")
_SIZE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)([bmt])(?![a-z0-9])", re.I)
# "A3B", "A95B": active parameters, the convention Qwen started
_ACTIVE = re.compile(r"(?<![a-z0-9.])a(\d+(?:\.\d+)?)([bm])(?![a-z0-9])", re.I)
# Names where a size token is per-expert or otherwise not the total
_UNSAFE = re.compile(r"\d+x\d|(?<![a-z0-9])\d+e(?![a-z0-9])", re.I)


def size_from_name(name: str) -> tuple[float | None, float | None]:
    """Return (total, active) parameter counts encoded in a name, when it is safe to read them.

    'Qwen3.8-2.4T-A95B' -> (2.4e12, 95e9); 'gemma-3-27b-it' -> (27e9, None);
    'Mixtral-8x7B' and 'Llama-4-Scout-17B-16E' -> (None, None): the token is per expert.
    """
    s = name.replace("_", "-")
    active = None
    m = _ACTIVE.search(s)
    if m:
        active = float(m.group(1)) * _UNIT[m.group(2).lower()]
    if _UNSAFE.search(s):
        return None, active
    total = None
    m = _SIZE.search(s)
    if m:
        total = float(m.group(1)) * _UNIT[m.group(2).lower()]
    if total is not None and active is not None and active >= total:
        return None, None
    return total, active


_QUANT = re.compile(r"(gguf|awq|gptq|mlx|fp8|nvfp4|mxfp8|int4|int8|bnb|exl2|[-_.]q\d|\dbit|quantized)", re.I)


def looks_quantised(repo: str) -> bool:
    return bool(_QUANT.search(repo_name(repo)))
