"""Tests for the data build. Run with:  python -m unittest discover -s tests -v

The end-to-end tests replace the network with canned responses shaped like the
documented APIs, so they check the joining, sizing and fallback logic - they do
not prove that the live endpoints still look like this.
"""
import io
import json
import os
import pathlib
import tempfile
import unittest
import urllib.error
from unittest import mock

from pipeline import build_data, params
from pipeline.names import looks_quantised, name_keys, norm_key, size_from_name

ROOT = pathlib.Path(__file__).resolve().parent.parent


class Names(unittest.TestCase):
    def test_same_model_same_key(self):
        spellings = ["Qwen3 30B A3B", "Qwen: Qwen3 30B A3B (free)", "qwen3-30b-a3b", "Qwen3-30B-A3B"]
        self.assertEqual({norm_key(s) for s in spellings}, {"qwen330ba3b"})
        self.assertEqual(norm_key("gemma-3-27b-it"), norm_key("Gemma 3 27B"))
        self.assertEqual(norm_key("Kimi K3 (Thinking)"), norm_key("Kimi K3"))

    def test_different_models_differ(self):
        self.assertNotEqual(norm_key("DeepSeek V3 (0324)"), norm_key("DeepSeek V3"))
        self.assertNotEqual(norm_key("Qwen3.5 27B"), norm_key("Qwen3 527B"))
        self.assertNotEqual(norm_key("GLM-5.3"), norm_key("GLM-5.3-Flash"))

    def test_size_from_name(self):
        self.assertEqual(size_from_name("Qwen3.8-2.4T-A95B"), (2.4e12, 95e9))
        self.assertEqual(size_from_name("gemma-3-27b-it"), (27e9, None))
        self.assertEqual(size_from_name("LFM2-350M"), (350e6, None))
        self.assertEqual(size_from_name("Qwen3-0.6B"), (0.6e9, None))

    def test_size_not_read_when_per_expert(self):
        for name in ["Mixtral-8x7B-v0.1", "Llama-4-Scout-17B-16E-Instruct", "Gemma 3n E2B IT", "Command R7B", "Kimi-K2.6", "phi-4"]:
            self.assertIsNone(size_from_name(name)[0], name)

    def test_quantised(self):
        self.assertTrue(looks_quantised("reflection/Beam-1.0-501B-A23B-FP8"))
        self.assertTrue(looks_quantised("someone/Model-GGUF"))
        self.assertFalse(looks_quantised("Qwen/Qwen3-30B-A3B"))


def board_of(*entries):
    """entries: (id, released, spelling, ...) -> a Board over just those models."""
    models = {}
    for mid, released, *spellings in entries:
        models[mid] = {"id": mid, "name": spellings[0], "names": [spellings[0]], "spellings": set(spellings),
                       "keys": set(), "hf_id": None, "released": released, "scores": {}}
    return build_data.Board(models, {})


class Matching(unittest.TestCase):
    """Cases from the first live build's unmatched list."""

    def found(self, board, name):
        m, tier = board.find(name)
        return (m["id"] if m else None), tier

    def test_run_modes_and_precision_labels_are_ignored(self):
        b = board_of(("k25", "2026-01-27", "Kimi K2.5"), ("nemo", None, "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"))
        self.assertEqual(self.found(b, "Kimi K2.5 (Instant)"), ("k25", 1))
        self.assertEqual(self.found(b, "NVIDIA-Nemotron-3-Ultra-550B-A55B (NVFP4)"), ("nemo", 1))

    def test_maker_prefix(self):
        b = board_of(("vl", None, "Nemotron 3.5 Super VL 120B A12B"))
        self.assertEqual(self.found(b, "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B (NVFP4)"), ("vl", 2))

    def test_instruct_and_base_no_longer_cancel_out(self):
        b = board_of(("inst", None, "Llama 3.2 1B Instruct", "Llama-3.2-1B-Instruct"), ("base", None, "Llama-3.2-1B"),
                     ("think", None, "Qwen3-Next 80B-A3B (Thinking)", "Qwen3-Next-80B-A3B-Thinking"),
                     ("qinst", None, "Qwen3-Next 80B-A3B Instruct"))
        self.assertEqual(self.found(b, "Llama 3.2 1B Instruct")[0], "inst")
        self.assertEqual(self.found(b, "Llama 3.2 1B")[0], "base")   # word-for-word match wins
        self.assertEqual(self.found(b, "Qwen3-Next-80B-A3B-Instruct")[0], "qinst")
        self.assertEqual(self.found(b, "Qwen3-Next-80B-A3B-Thinking")[0], "think")

    def test_chat_model_preferred_over_its_base(self):
        b = board_of(("it", None, "gemma-2-9b-it"), ("base", None, "gemma-2-9b-pt"))
        self.assertEqual(self.found(board_of(("it", None, "Gemma 2 9B IT"), ("base", None, "Gemma 2 9B Base")), "Gemma 2 9B IT")[0], "it")
        two = board_of(("inst", None, "Olmo 3 32B Instruct"), ("plain", None, "Olmo-3-32B-hf"))
        self.assertEqual(self.found(two, "Olmo 3 32B (Thinking)")[0], "inst")

    def test_plus_is_not_dropped(self):
        b = board_of(("r", None, "Command R", "c4ai-command-r-08-2024"), ("rplus", None, "Command R+", "c4ai-command-r-plus-08-2024"))
        self.assertEqual(self.found(b, "Command R+ (08-2024)")[0], "rplus")
        self.assertEqual(self.found(b, "Command R (08-2024)")[0], "r")

    def test_snapshot_date_must_fit_the_release_date(self):
        b = board_of(("flash", "2026-09-10", "DeepSeek V4.1 Flash"), ("flash1017", "2026-10-17", "DeepSeek V4.1 Flash 1017"),
                     ("small", "2025-06-20", "Mistral Small 3.2"))
        self.assertEqual(self.found(b, "DeepSeek-V4.1-Flash-20260910 (Max)"), ("flash", 2))
        self.assertEqual(self.found(b, "DeepSeek-V4.1-Flash-20261017 (Max)"), ("flash1017", 1))
        self.assertEqual(self.found(b, "Mistral Small 3.2 (2506)"), ("small", 2))
        self.assertEqual(self.found(b, "DeepSeek-V4.1-Flash-20260801")[0], None)  # some other snapshot

    def test_versions_and_previews_are_different_models(self):
        b = board_of(("m03", None, "Mistral 7B Instruct v0.3"), ("qwq", "2025-03-05", "QwQ 32B"), ("m21", None, "MiniMax-M2.1"))
        for other in ["Mistral 7B Instruct v0.2", "Mistral 7B Instruct", "QwQ-32B-Preview", "MiniMax-M2.1 Preview"]:
            self.assertEqual(self.found(b, other), (None, build_data.Board.NO_MATCH), other)

    def test_sizes_never_stand_in_for_each_other(self):
        b = board_of(("g8", None, "Granite 4.2 8B"), ("q7", None, "Qwen2.5 7B Instruct"), ("glimmer", None, "Muse Glimmer 30B"))
        for other in ["Granite 4.2 30B", "Granite 4.2 3B", "Qwen2.5-72B-Instruct"]:
            self.assertIsNone(self.found(b, other)[0], other)
        self.assertEqual(self.found(b, "Muse Glimmer"), ("glimmer", 3))  # no size given, one candidate

    def test_two_sizes_make_a_sizeless_name_ambiguous(self):
        b = board_of(("a", None, "Gemma 4 31B"), ("b", None, "Gemma 4 12B"))
        self.assertEqual(self.found(b, "Gemma 4"), (None, build_data.Board.AMBIGUOUS))

    def test_generation_names_are_not_versions(self):
        self.assertEqual(name_keys("DeepSeek V4.1 Flash").quals, frozenset())
        self.assertEqual(name_keys("Mixtral 8x7B Instruct v0.1").quals, frozenset({"v0.1"}))

    def test_alias_wins(self):
        b = board_of(("k2", None, "Kimi K2 Instruct"), ("k2b", None, "Kimi K2 Instruct 0905"))
        self.assertTrue(b.add_alias("Kimi K2 Preview (0711)", "k2"))
        self.assertEqual(self.found(b, "Kimi K2 Preview (0711)"), ("k2", 0))


class ActiveEstimate(unittest.TestCase):
    def test_against_maker_figures(self):
        cases = [
            # (total, config, maker's stated active count)
            (671e9, {"n_routed_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048, "hidden_size": 7168,
                     "num_hidden_layers": 61, "first_k_dense_replace": 3, "intermediate_size": 18432}, 37e9),
            (30.53e9, {"num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 768, "hidden_size": 2048,
                       "num_hidden_layers": 48, "intermediate_size": 6144}, 3.3e9),
            (46.7e9, {"num_local_experts": 8, "num_experts_per_tok": 2, "hidden_size": 4096, "num_hidden_layers": 32,
                      "intermediate_size": 14336}, 12.9e9),
        ]
        for total, cfg, stated in cases:
            est = params.estimate_active(total, cfg)
            self.assertAlmostEqual(est / stated, 1.0, delta=0.03)

    def test_dense_config_is_not_moe(self):
        self.assertIsNone(params.moe_shape({"hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336}))

    def test_nested_text_config(self):
        cfg = {"text_config": {"num_experts": 64, "num_experts_per_tok": 4, "moe_intermediate_size": 1024, "hidden_size": 2048, "num_hidden_layers": 24}}
        self.assertEqual(params.moe_shape(cfg)["experts"], 64)


class Descriptions(unittest.TestCase):
    def test_sizes(self):
        f = build_data.size_from_description
        self.assertEqual(f("Multimodal MoE reasoning model (594B total, 17B active) for agents"), (594e9, 17e9, True))
        self.assertEqual(f("Reasoning-tuned 26B MoE model with 3B active parameters"), (26e9, 3e9, True))
        self.assertEqual(f("model with 314 billion total parameters and 13.2 billion parameters activated per token"), (314e9, 13.2e9, False))
        self.assertEqual(f("Long-context model with a 1M-token window"), (None, None, False))


class DataFiles(unittest.TestCase):
    def test_overrides_load(self):
        ov = build_data.load_overrides(ROOT / "data" / "overrides.yaml")
        self.assertEqual(ov["deepseek-ai/deepseek-v3"]["active"], 37e9)
        for key, row in ov.items():
            if row["arch"] == "moe":
                self.assertTrue(row["total"] and row["active"] and row["active"] < row["total"], key)

    def test_parse_size(self):
        self.assertEqual(build_data.parse_size("5.1B"), 5.1e9)
        self.assertEqual(build_data.parse_size("560M"), 560e6)
        self.assertEqual(build_data.parse_size("1T"), 1e12)
        with self.assertRaises(ValueError):
            build_data.parse_size("lots")


# ---------------------------------------------------------------------------
# End to end, with the network replaced
# ---------------------------------------------------------------------------

def toml_model(name, hf=None, open_weights=True, released="2026-05-01", benchmarks=(), description="A test model"):
    out = [f'name = "{name}"', f'description = "{description}"', f"open_weights = {'true' if open_weights else 'false'}",
           f'release_date = "{released}"', "[modalities]", 'input = ["text"]', 'output = ["text"]']
    if hf:
        out += ["[[weights]]", 'label = "Hugging Face"', f'url = "https://huggingface.co/{hf}"']
    for bname, score, source in benchmarks:
        out += ["[[benchmarks]]", f'name = "{bname}"', f"score = {score}", f'source = "{source}"', 'date = "2026-10-01"']
    return "\n".join(out) + "\n"


def make_catalogue(root: pathlib.Path):
    files = {
        "acme/dense-7b": toml_model("Acme Dense 7B", "acme/Dense-7B-Instruct", benchmarks=[("Test Bench", 61.0, "https://huggingface.co/acme/Dense-7B-Instruct")]),
        "acme/sparse": toml_model("Acme Sparse", "acme/Sparse", benchmarks=[("Test Bench", 70.0, "https://artificialanalysis.ai/x")]),
        "acme/closed": toml_model("Acme Closed", open_weights=False),
        "beta/named-40b-a4b": toml_model("Beta 40B A4B", "beta/Beta-40B-A4B", benchmarks=[("Test Bench", 66.0, "https://beta.example/card")]),
        "beta/mystery": toml_model("Beta Mystery", "beta/Mystery"),
        "beta/packed-35b": toml_model("Beta Packed 35B", "beta/Packed-35B"),
        "acme/nolink-12b": toml_model("Acme Nolink 12B"),
    }
    # padding so the catalogue passes its sanity check (>= 50 files)
    for i in range(50):
        files[f"pad/closed-{i}"] = toml_model(f"Pad Closed {i}", open_weights=False)
    for key, text in files.items():
        p = root / "models" / f"{key}.toml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    (root / "labs" / "acme").mkdir(parents=True)
    (root / "labs" / "acme" / "lab.toml").write_text('name = "Acme Labs"\n', encoding="utf-8")


SPARSE_CFG = {"num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 768, "hidden_size": 2048, "num_hidden_layers": 48}


class FakeWeb:
    """Routes URLs to canned responses; anything listed in `down` raises like an outage."""

    def __init__(self):
        self.down: set[str] = set()
        self.calls: list[str] = []

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.calls.append(url)
        if any(d in url for d in self.down):
            raise urllib.error.URLError("simulated outage")
        body = self.route(url, dict(req.header_items()))
        if body is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        resp = mock.MagicMock()
        resp.read.return_value = data
        resp.__enter__.return_value = resp
        return resp

    def route(self, url, headers):
        if "huggingface.co/api/models?" in url:
            if "author=acme" not in url:
                return []
            if "sort=createdAt" in url:
                return [{"id": "acme/Nolink-12B-Instruct", "createdAt": "2026-10-01T00:00:00.000Z", "tags": []}]
            return [{"id": "acme/Tiny-1B", "createdAt": "2026-09-01T00:00:00.000Z", "tags": []},
                    {"id": "acme/Tiny-1B-GGUF", "createdAt": "2026-09-02T00:00:00.000Z", "tags": ["base_model:quantized:acme/Tiny-1B"]}]
        if url.endswith("/api/models/acme/Dense-7B-Instruct"):
            return {"id": "acme/Dense-7B-Instruct", "createdAt": "2026-05-01T00:00:00.000Z", "safetensors": {"parameters": {"BF16": 7_250_000_000}, "total": 7_250_000_000}, "cardData": {"license": "apache-2.0"}}
        if url.endswith("/api/models/acme/Sparse"):
            return {"id": "acme/Sparse", "createdAt": "2026-06-01T00:00:00.000Z", "safetensors": {"parameters": {"BF16": 30_530_000_000}, "total": 30_530_000_000}}
        if url.endswith("/api/models/beta/Packed-35B"):
            return {"id": "beta/Packed-35B", "safetensors": {"total": 664_944}}
        if url.endswith("/api/models/beta/Beta-40B-A4B"):
            return {"id": "beta/Beta-40B-A4B", "safetensors": {"total": 40_700_000_000}}
        if url.endswith("/api/models/acme/Nolink-12B-Instruct"):
            return {"id": "acme/Nolink-12B-Instruct", "safetensors": {"total": 12_300_000_000}}
        if url.endswith("/api/models/acme/Tiny-1B"):
            return {"id": "acme/Tiny-1B", "safetensors": {"total": 1_100_000_000}}
        if url.endswith("/acme/Sparse/resolve/main/config.json"):
            return SPARSE_CFG
        if url.endswith("/config.json"):
            return {"hidden_size": 1024, "num_hidden_layers": 12, "intermediate_size": 4096}
        if "openrouter.ai/api/v1/models" in url:
            return {"data": [{"id": "acme/sparse-pro", "name": "Acme: Sparse Pro", "hugging_face_id": "acme/Sparse"},
                             {"id": "x/closed", "name": "X: Closed", "hugging_face_id": None}]}
        if "datasets-server.huggingface.co/rows" in url:
            rows = [{"model_name": "Sparse Pro (Thinking)", "organization": "acme", "license": "Apache 2.0", "rating": 1401.5, "category": "overall", "leaderboard_publish_date": "2026-11-11"},
                    {"model_name": "Acme Dense 7B", "organization": "acme", "license": "MIT", "rating": 1250.0, "category": "overall", "leaderboard_publish_date": "2026-11-11"},
                    {"model_name": "Acme Dense 7B", "organization": "acme", "license": "MIT", "rating": 1300.0, "category": "coding", "leaderboard_publish_date": "2026-11-11"},
                    {"model_name": "Unknown Open 3B (BF16)", "organization": "who", "license": "MIT", "rating": 1100.0, "category": "overall", "leaderboard_publish_date": "2026-11-11"},
                    {"model_name": "Sizeless Open", "organization": "who", "license": "MIT", "rating": 1050.0, "category": "overall", "leaderboard_publish_date": "2026-11-11"},
                    {"model_name": "Big Closed", "organization": "corp", "license": "Proprietary", "rating": 1500.0, "category": "overall", "leaderboard_publish_date": "2026-11-11"}]
            return {"rows": [{"row_idx": i, "row": r} for i, r in enumerate(rows)], "num_rows_total": len(rows)}
        if "benchlm.ai/data/leaderboard.json" in url:
            def row(model, kind, score, status, lo, hi):
                return {"model": model, "creator": "Acme", "sourceType": kind, "score": score, "displayScore": score,
                        "evidenceStatus": status, "scoreInterval90": {"lower": lo, "upper": hi}, "rankingEligible": True,
                        "url": "https://benchlm.ai/models/" + model.lower().replace(" ", "-")}
            items = [row("Big Closed", "Proprietary", 90.0, "supported", 85, 95), row("Acme Sparse", "Open Weight", 61.5, "supported", 55, 68),
                     row("Acme Dense 7B", "Open Weight", 40.2, "estimated", 30, 50), row("Unknown Open 3B", "Open Weight", 22.0, "estimated", 10, 35)]
            items += [row(f"Filler Closed {i}", "Proprietary", 50.0 - i, "supported", 40, 60) for i in range(20)]
            return {"schemaVersion": "1.0", "generatedAt": "2026-11-16T20:19:49.100Z", "items": items,
                    "categories": {"coding": [row("Acme Sparse", "Open Weight", 70.0, "supported", 60, 80)]}}
        if "artificialanalysis.ai/api" in url:
            assert headers.get("X-api-key") == "test-key", headers
            return {"data": [{"name": "Beta 40B A4B", "slug": "beta-40b-a4b", "model_creator": {"name": "Beta"},
                              "evaluations": {"artificial_analysis_intelligence_index": 44.2, "artificial_analysis_coding_index": None}}]}
        if "raw.githubusercontent.com/Aider-AI" in url:
            rows = "".join(f"- model: filler-{i}\n  pass_rate_2: {i}.0\n  date: 2025-01-01\n" for i in range(12))
            return (rows + "- model: Acme Dense 7B\n  pass_rate_2: 33.3\n  date: 2025-09-01\n").encode()
        return None


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        make_catalogue(self.dir / "md")
        self.web = FakeWeb()
        self.cfg = {
            "modelsdev": {"min_models": 2, "independent_hosts": ["artificialanalysis.ai"], "merge_into": {}},
            "hub": {"recheck_days": 30, "discover_orgs": ["acme"], "discover_limit": 10, "discover_max_age_days": 730},
            "lmarena": {"add_unlisted": True, "boards": [{"config": "text", "category": "overall", "id": "lmarena:text", "name": "LMArena text"}]},
            "benchlm": {"add_unlisted": True, "categories": ["coding", "agentic"]},
            "hf_leaderboards": [],
            "default_benchmark": ["aa:intelligence"],
            "benchmark_notes": {},
        }
        real_load = build_data.yaml.safe_load

        def fake_yaml(text):
            return self.cfg if isinstance(text, str) and text.lstrip().startswith("# Build settings") else real_load(text)

        patches = [
            mock.patch("urllib.request.urlopen", self.web),
            mock.patch("time.sleep", lambda s: None),
            mock.patch.object(build_data.yaml, "safe_load", fake_yaml),
            mock.patch.dict(os.environ, {"AA_API_KEY": "test-key"}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def build(self, *extra):
        out = self.dir / "out" / "models.json"
        code = build_data.main(["--out", str(out), "--modelsdev-dir", str(self.dir / "md"), *extra])
        self.assertEqual(code, 0)
        data = json.loads(out.read_text())
        return data, {m["id"]: m for m in data["models"]}

    def test_full_build(self):
        data, m = self.build()
        self.assertEqual({s["id"]: s["status"] for s in data["sources"]},
                         {"modelsdev": "ok", "hub": "ok", "openrouter": "ok", "aider": "ok", "lmarena": "ok", "benchlm": "ok", "aa": "ok"})
        self.assertNotIn("md:acme/closed", m)  # closed weights are left out

        dense = m["acme/dense-7b-instruct"]
        self.assertEqual((dense["params_total"], dense["params_total_src"], dense["arch"]), (7.25e9, "hub", "dense"))
        self.assertEqual(dense["params_active"], dense["params_total"])
        self.assertEqual(dense["org"], "Acme Labs")
        self.assertEqual(dense["licence"], "apache-2.0")

        sparse = m["acme/sparse"]  # MoE spotted from config.json, active estimated from the weights
        self.assertEqual((sparse["arch"], sparse["arch_src"], sparse["params_active_src"]), ("moe", "config", "estimate"))
        self.assertAlmostEqual(sparse["params_active"] / 3.35e9, 1.0, delta=0.02)

        named = m["beta/beta-40b-a4b"]  # total from the weights, active from the name
        self.assertEqual((named["params_total"], named["params_total_src"]), (40.7e9, "hub"))
        self.assertEqual((named["params_active"], named["params_active_src"]), (4e9, "name"))

        packed = m["beta/packed-35b"]  # implausible Hub count: the name's size is used instead
        self.assertEqual((packed["params_total"], packed["params_total_src"]), (35e9, "name"))
        self.assertIsNone(m["beta/mystery"]["params_total"])  # the Hub had nothing: stays unknown, never guessed
        self.assertIn("acme/tiny-1b", m)                       # found by discovery
        self.assertNotIn("acme/tiny-1b-gguf", m)               # quantised copy skipped

        # scores: maker-reported vs independent, name joins, run-mode labels, OpenRouter spelling
        self.assertTrue(dense["scores"]["md:test-bench"]["self"])
        self.assertFalse(sparse["scores"]["md:test-bench"]["self"])
        self.assertEqual(sparse["scores"]["lmarena:text"]["v"], 1401.5)
        self.assertEqual(sparse["scores"]["lmarena:text"]["label"], "Sparse Pro (Thinking)")
        self.assertEqual(dense["scores"]["lmarena:text"]["v"], 1250.0)  # the 'coding' category row is ignored
        self.assertEqual(named["scores"]["aa:intelligence"]["v"], 44.2)
        self.assertEqual(dense["scores"]["aider:polyglot"]["v"], 33.3)
        self.assertEqual(data["default_benchmark"], "aa:intelligence")

        # BenchLM: one composite, estimated rows flagged, intervals kept, closed rows ignored
        bl = sparse["scores"]["benchlm:overall"]
        self.assertEqual((bl["v"], bl["lo"], bl["hi"], bl.get("est")), (61.5, 55, 68, None))
        self.assertTrue(dense["scores"]["benchlm:overall"]["est"])
        self.assertEqual(sparse["scores"]["benchlm:coding"]["v"], 70.0)
        overall = next(b for b in data["benchmarks"] if b["id"] == "benchlm:overall")
        self.assertEqual((overall["group"], overall["est_count"]), ("broad", 2))
        self.assertEqual(data["benchmarks"][0]["group"], "broad")  # whole-model comparisons are listed first

        # an open model only a leaderboard knows: added when its name states a size
        extra = m["lb:unknownopen3b"]
        self.assertEqual((extra["name"], extra["params_total"], extra["unlisted"]), ("Unknown Open 3B", 3e9, "LMArena"))
        self.assertEqual(extra["scores"]["lmarena:text"]["v"], 1100.0)
        self.assertEqual(extra["scores"]["benchlm:overall"]["v"], 22.0)  # the next source finds it by name

        # a catalogue entry with no weights link picks up the repo that discovery found
        nolink = m["md:acme/nolink-12b"]
        self.assertEqual((nolink["hf_id"], nolink["params_total"], nolink["params_total_src"]), ("acme/Nolink-12B-Instruct", 12.3e9, "hub"))
        self.assertNotIn("acme/nolink-12b-instruct", m)

        unmatched = json.loads((self.dir / "out" / "unmatched.json").read_text())["unmatched"]
        self.assertEqual(unmatched["lmarena"], ["Sizeless Open"])  # proprietary rows are not reported
        self.assertNotIn("benchlm", unmatched)

    def test_outage_keeps_last_good_scores(self):
        first, _ = self.build()
        prev = self.dir / "prev.json"
        prev.write_text(json.dumps(first))
        self.web.down = {"datasets-server.huggingface.co", "huggingface.co/api", "resolve/main"}
        self.web.calls.clear()
        data, m = self.build("--previous", str(prev))
        status = {s["id"]: s for s in data["sources"]}
        self.assertEqual(status["lmarena"]["status"], "stale")
        self.assertEqual(status["lmarena"]["fetched_at"], next(s for s in first["sources"] if s["id"] == "lmarena")["fetched_at"])
        self.assertEqual(m["acme/sparse"]["scores"]["lmarena:text"]["v"], 1401.5)
        self.assertIn("lmarena:text", {b["id"] for b in data["benchmarks"]})
        # sizes survive a Hub outage through the previous build's cache
        self.assertEqual(m["acme/dense-7b-instruct"]["params_total_src"], "hub")
        self.assertIn("acme/tiny-1b", m)
        self.assertEqual(m["acme/tiny-1b"]["params_total"], 1.1e9)

    def test_hub_answers_are_reused(self):
        first, _ = self.build()
        prev = self.dir / "prev.json"
        prev.write_text(json.dumps(first))
        self.web.calls.clear()
        self.build("--previous", str(prev))
        per_repo = [u for u in self.web.calls if "/api/models/" in u and "Mystery" not in u]
        self.assertEqual(per_repo, [])  # only the repo with no count yet is asked again

    def test_missing_key_is_a_skip_not_a_failure(self):
        with mock.patch.dict(os.environ, {"AA_API_KEY": ""}):
            data, _ = self.build()
        aa = next(s for s in data["sources"] if s["id"] == "aa")
        self.assertEqual(aa["status"], "skipped")

    def test_catalogue_failure_stops_the_build(self):
        code = build_data.main(["--out", str(self.dir / "x.json"), "--modelsdev-dir", str(self.dir / "nowhere")])
        self.assertEqual(code, 2)
        self.assertFalse((self.dir / "x.json").exists())


class OpenRouterShape(unittest.TestCase):
    def test_parse(self):
        from pipeline.sources import openrouter
        rows = openrouter.parse({"data": [{"id": "qwen/qwen3.6-27b", "name": "Qwen: Qwen3.6 27B", "hugging_face_id": "Qwen/Qwen3.6-27B"},
                                          {"id": "a/b", "name": "A: B", "hugging_face_id": None}]})
        self.assertEqual(rows, [{"hf_id": "Qwen/Qwen3.6-27B", "name": "Qwen: Qwen3.6 27B", "slug": "qwen/qwen3.6-27b"}])


if __name__ == "__main__":
    unittest.main()
