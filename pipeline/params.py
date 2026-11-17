"""Active-parameter estimate for mixture-of-experts models from config.json.

A routed MoE layer holds E expert feed-forward blocks but runs only k of them
per token. Everything else in the network (attention, embeddings, shared
experts, dense layers) runs for every token. So:

    active = total - moe_layers * (E - k) * params_per_expert
    params_per_expert = 3 * hidden_size * expert_intermediate_size   (gate, up, down)

Checked by hand against the makers' figures: DeepSeek-V3 (671B -> 37.5B, stated
37B), Qwen3-30B-A3B (30.5B -> 3.35B, stated 3.3B), Mixtral 8x7B (46.7B -> 12.9B,
stated 12.9B). It is still an estimate - new layouts can break the assumptions -
so the site shows it with a tilde and a stated figure in data/overrides.yaml
always wins.
"""
from __future__ import annotations

_EXPERTS = ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts", "num_routed_experts")
_TOPK = ("num_experts_per_tok", "num_experts_per_token", "moe_topk", "moe_top_k", "experts_per_token", "num_selected_experts")
_EXPERT_WIDTH = ("moe_intermediate_size", "expert_intermediate_size", "moe_hidden_size", "intermediate_size")
_HIDDEN = ("hidden_size", "d_model", "n_embd")
_LAYERS = ("num_hidden_layers", "n_layer", "num_layers")


def _first(cfg: dict, keys: tuple[str, ...]):
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return v
    return None


def text_config(cfg: dict) -> dict:
    """Multimodal repos nest the language model's settings."""
    for k in ("text_config", "llm_config", "language_config"):
        if isinstance(cfg.get(k), dict) and cfg[k]:
            merged = dict(cfg)
            merged.update(cfg[k])
            return merged
    return cfg


def moe_shape(cfg: dict) -> dict | None:
    """Return {'experts','top_k','width','hidden','moe_layers'} or None when the config is not a routed MoE."""
    cfg = text_config(cfg or {})
    experts = _first(cfg, _EXPERTS)
    top_k = _first(cfg, _TOPK)
    if not experts or not top_k or experts <= 1 or top_k >= experts:
        return None
    width = _first(cfg, _EXPERT_WIDTH)
    hidden = _first(cfg, _HIDDEN)
    layers = _first(cfg, _LAYERS)
    if not (width and hidden and layers):
        return {"experts": experts, "top_k": top_k}
    dense_first = cfg.get("first_k_dense_replace") or 0
    mlp_only = cfg.get("mlp_only_layers") or []
    moe_layers = layers - (dense_first if isinstance(dense_first, int) else 0) - (len(mlp_only) if isinstance(mlp_only, list) else 0)
    step = cfg.get("decoder_sparse_step")
    if isinstance(step, int) and step > 1:
        moe_layers = moe_layers // step
    return {"experts": int(experts), "top_k": int(top_k), "width": int(width), "hidden": int(hidden), "moe_layers": int(moe_layers)}


def estimate_active(total: float, cfg: dict) -> float | None:
    shape = moe_shape(cfg)
    if not shape or "width" not in shape or not total:
        return None
    per_expert = 3 * shape["hidden"] * shape["width"]
    inactive = shape["moe_layers"] * (shape["experts"] - shape["top_k"]) * per_expert
    active = total - inactive
    # Sanity: the answer must be a plausible share of the model.
    if active <= 0 or active >= total or active < 0.005 * total:
        return None
    return float(active)
