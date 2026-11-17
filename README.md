# Parameter budget

A static site that answers one question: **which open-weight language models fit under a
parameter limit, and which of them score best?**

Set a limit on **total** parameters (memory to load the model) or **active** parameters
(weights used per token, which is much lower for mixture-of-experts models), pick a
benchmark, and the site ranks everything that fits. A scatter plot of score against size
shows the limit as a line you can drag, and the best score available at each size.
Switch the table to **All benchmarks** to see every model that fits against every benchmark
it has a score for, and select any column heading to rank by it.

The page is plain HTML, CSS and JavaScript with no dependencies and no third-party
requests. All data comes from one file, `data/models.json`, which a scheduled GitLab CI job
rebuilds from public sources.


The workflow runs on every push, daily on a schedule, and on demand from the Actions tab.
GitHub pauses scheduled workflows in a repository with no activity for 60 days; re-enable
it from the Actions tab if that happens. The two CI files do not interfere, so delete
whichever you are not using.

## Run it locally

```bash
pip install -r requirements.txt
make test          # unit and end-to-end tests, no network needed
make data          # builds public/ with fresh data
make serve         # http://localhost:8000
```

The site must be served over HTTP (it fetches `data/models.json`), so opening
`index.html` from disk shows a message instead of the ranking.

## Where the data comes from

| Source | What it supplies | Access | Terms |
|---|---|---|---|
| [models.dev](https://github.com/sst/models.dev) | The list of open-weight models, release dates, licences, weights links, and benchmark scores with the URL each was published at | Repository tarball, no key | MIT |
| [Hugging Face Hub](https://huggingface.co/docs/hub/api) | Parameter counts from the weights (`safetensors.total`), `config.json` for spotting mixture-of-experts models, and newly released repositories from the organisations listed in `data/config.yaml` | Public API, token optional | Public metadata |
| [LMArena](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset) | Elo ratings from blind human votes (text and WebDev boards) | Hugging Face dataset viewer API, no key | CC BY 4.0 |
| [BenchLM](https://benchlm.ai/data) | One composite score per model (the default ranking), plus coding, agentic, reasoning and knowledge composites | Static JSON, no key | CC BY-NC 4.0: non-commercial use only, credit "Data from BenchLM.ai" |
| [Artificial Analysis](https://artificialanalysis.ai) | Intelligence Index and Coding Index | Free API, key needed, 100 requests a day | Attribution with a link (the footer provides it) |
| [Aider polyglot](https://aider.chat/docs/leaderboards/) | Coding benchmark, frozen since October 2025 | YAML file on GitHub | Apache-2.0 repository |
| [OpenRouter](https://openrouter.ai/api/v1/models) | Extra spellings of model names, for matching only | Public API, no key | No scores or prices are taken |

Sources that were looked at and left out: llm-stats.com (its free tier forbids
redistribution), the Open LLM Leaderboard (archived), LiveBench (no fresh machine-readable
release found), and Epoch AI's benchmark archive (CC BY 4.0 and worth adding, but its
columns could not be inspected while this was built). Hugging Face's new per-benchmark
leaderboards are supported: list the benchmark datasets under `hf_leaderboards` in
`data/config.yaml` and they are joined by repository id.

### Read this before trusting the BenchLM column

BenchLM's score is the one metric here that puts most recent models on a single scale, which
is why the site opens on it. It is a model, not a measurement. BenchLM starts from other
publishers' leaderboards (composite indices and human-preference boards), adjusts with direct
benchmark results, and for models with thin evidence leans on peers and the model's lineage.
It labels each row "supported" or "estimated"; on 16 November 2026 about 135 of its 215
ranked models were estimated. Estimated scores carry a tilde here and the tooltip gives
BenchLM's 90% range. The licence is non-commercial: fine for a personal site, not for a
commercial product without a licence from BenchLM. It is also a one-person project started
in August 2025, so treat its continued availability as uncertain; if it goes away the site
falls back to LMArena as the default.

### What has and has not been run against the live service

- **Run for real** (first by the author in a sandbox, then in the first live build on 17
  November 2026): models.dev, Aider, Hugging Face Hub, LMArena, Artificial Analysis and
  OpenRouter. In that build the Hub supplied 224 exact parameter counts, LMArena 421 text
  rows, and the free Artificial Analysis endpoint only 41 models, so that source adds
  little.
- **Written from the published file layout and tested only against canned responses:**
  BenchLM, the discovery of newest repositories, and the Hugging Face benchmark
  leaderboards. Check the job log after the next run: each source prints `ok`, `skipped`
  or `failed` with a reason.

The committed snapshot `site/data/models.json` is that first live build's data, re-joined
with the current matching rules.

## How sizes are decided

**Total parameters**, first match wins:

1. a maker-stated figure in `data/overrides.yaml` (`basis: stated`)
2. the count read from the weights on Hugging Face
3. a third-party figure in `data/overrides.yaml` (`basis: third_party`)
4. the size in the model's name (`Qwen3.5-27B`)
5. the size in the models.dev description

**Active parameters** for a mixture-of-experts model, first match wins:

1. a maker-stated figure in `data/overrides.yaml`
2. the `A3B`-style token in the name (`Qwen3-30B-A3B`)
3. the figure in the models.dev description
4. a third-party figure in `data/overrides.yaml`
5. an estimate from `config.json`: total minus the expert blocks that are not routed to
   (`moe_layers x (experts - top_k) x 3 x hidden x expert_width`). This reproduces the
   makers' figures for DeepSeek-V3, Qwen3-30B-A3B and Mixtral 8x7B to within 3%.

A dense model's active count is its total. A mixture-of-experts model with no active
figure is never given one by guesswork: it is listed as unknown and left out of rankings
by active parameters. Third-party figures and estimates are shown with a tilde, and
hovering any size says where it came from.

A model is treated as dense unless its name, description, configuration or an override
says otherwise. That errs on the safe side: a mixture-of-experts model nobody has flagged
is counted at its full size, so it can be wrongly excluded from a small active budget but
never wrongly included.

## How scores are joined

Leaderboards name models in their own way, and none of the ones used here carries a
Hugging Face repository id. Each name is reduced to keys of increasing looseness and a row
is attached at the tightest level that finds a catalogue model:

1. **Same name** once case, punctuation, the maker prefix, run-mode labels such as
   "(Thinking)" or "(Max)", precision labels such as "(NVFP4)" and words like "Instruct"
   are ignored.
2. **Same name apart from a snapshot date or a maker prefix.** A date on the row must be
   the catalogue entry's release date ("DeepSeek-V4.1-Flash-20260910" is the model
   released on 10 September 2026).
3. **Same name without its size**, only when the row gives no size and exactly one
   catalogue entry fits ("Muse Glimmer" is "Muse Glimmer 30B").

Keys must be equal; there is no fuzzy matching. Qualifiers may never disagree: "v0.2" is
not "v0.3", a preview is not the release, and "72B" is never "7B". When several catalogue
models share a key, the row is attached only if one is spelled exactly like it, or one is
the chat-tuned version of the other. When a name differs from the catalogue's, the table
shows "listed as ...".

**Models only a leaderboard knows.** An open-weight row on LMArena or BenchLM that matches
nothing is added as a model of its own when its name states a size ("Llama 3.1 405B
Instruct"), marked "listed by LMArena only". These have no release date, so they appear
only while Released is set to "Any time". Switch this off with `add_unlisted: false` in
`data/config.yaml`.

Rows that still fail are written to `data/unmatched.json` on the published site. To fix
one, add a line to `data/aliases.yaml`.

Scores recorded by models.dev keep the URL they were published at. A score whose source is
the maker's own page is marked "The maker" in the table; the hosts treated as independent
are listed in `data/config.yaml`. Many of the agentic-coding benchmarks are mostly
maker-reported, with different harnesses and settings per model, and the note under
"Rank by" says how many of a benchmark's scores that applies to.

## When a source is down

- Each source is fetched on its own. One failing does not stop the build.
- A failed source keeps its scores from the last good build (CI cache first, then the live
  site, then the committed snapshot) and the footer marks it stale with the date.
- Parameter counts are re-read only every 30 days, so a Hugging Face outage does not blank
  the size columns.
- If the model catalogue itself cannot be read, the job fails and the published site stays
  as it was.

## Files

```
site/                 the page: index.html, style.css, app.js, self-hosted font
site/data/models.json snapshot used until the first pipeline run, and as a last fallback
pipeline/             the data build (python -m pipeline.build_data --help)
pipeline/sources/     one small module per source
data/config.yaml      benchmark notes, discovery list, thresholds
data/overrides.yaml   hand-maintained parameter counts
data/aliases.yaml     extra name spellings for the join
tests/                python -m unittest discover -s tests
.gitlab-ci.yml        test, build and publish on GitLab
.github/workflows/    the same for GitHub
```

## Routine upkeep

- **A new mixture-of-experts model shows "active unknown" or a tilde:** add its stated
  figures to `data/overrides.yaml` with a link.
- **A leaderboard row is missing from the ranking:** look in `data/unmatched.json`, add an
  alias.
- **A benchmark column you want is missing:** models.dev benchmarks appear once
  `min_models` catalogue models have a score (default 6). Lower the threshold in
  `data/config.yaml`, or add a description for it under `benchmark_notes`.
- **The ten third-party entries in `data/overrides.yaml`** were relayed from Artificial
  Analysis model pages and a community wiki and have not been checked against the makers'
  pages. Totals are replaced automatically by the count from the weights once the Hugging
  Face step runs; the active figures stay until you replace them.

## Credits

Typeface: Archivo (SIL Open Font License, `site/fonts/OFL.txt`). Data belongs to the
sources above and is shown with attribution; LMArena data is CC BY 4.0 and Artificial
Analysis requires a visible link, both of which the site footer carries. BenchLM data is
CC BY-NC 4.0 and is credited there as "Data from BenchLM.ai".
