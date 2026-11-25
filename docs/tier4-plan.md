# Tier 4 plan: is there profile signal the methods leave unused?

Status: planned 2026-11-25, nothing run yet. Decision and go/no-go rule: [ADR 0001](adr/0001-diagnostics-before-llm-method.md).
Terms: [GLOSSARY.md](../GLOSSARY.md).

## Common setup

- **Reference model**: LightGCN+S2DR (`--variant s2dr-only`). Every gain below is measured against it, not against
  plain LightGCN.
- **Datasets**: amazon, yelp, steam. Yelp is the negative control: a semantic gain there is a warning sign.
- **Controls**: each condition runs with real, shuffled and random profiles, and seeds are paired. A gain counts as
  semantic only when real − best control > 0 across seeds.
- **Selection vs report**: every hyperparameter is picked on validation Recall@20 with seed 2025. The test results
  use the chosen setting only. The trainer does not store validation scores yet (see the code changes).
- **Where**: everything runs on nokton. The Mac runs only CPU smoke tests.
- **Order**: 1 → 2 → 3. Diagnostic 3 uses the fused scorer that diagnostic 2 picks.

## Diagnostic 1: is SDR switched off, and does a well-scaled SDR help?

**Question.** ProMax's SDR target is a softmax with temperature 0.2 over cosine similarities of whitened 32-d profiles.
If that target is near-uniform, then H_norm ≈ 1 and w sits at its 0.1 floor, so SDR barely acts. `nowhiten` changed
nothing, but it kept the same temperature and floor.

**Step 1a: logging** (3 runs, ProMax `full`, real profiles, seed 2025, one per dataset)
- Per epoch: mean of w, its quartiles, the share of users at the 0.1 floor, the mean of H_norm, and the SDR loss.
- Prediction: the share at the floor is close to 100%. If it is not, the scale hypothesis is wrong; stop step 1b and
  correct the `noentropy` caveat in FINDINGS.md.

**Step 1b: target sharpness** (57 runs, seed 2025, scored on validation)
- Base: `s2dr-only` + SDR (the `graph-train` variant), plus one `s2dr-only` run per dataset for a baseline
  validation score.
- Grid: target temperature τ_t ∈ {0.2, 0.05, 0.02} × profiles ∈ {whitened, raw} × `sdr_weight` ∈ {0.05, 0.5, 1.5},
  with the default w floor.
- `sdr_weight` is part of this grid (changed 2026-11-26, after the smoke test). A sharper target raises w from its
  floor toward 0.8 and also raises the SDR loss itself. At ProMax's default weight, the sharpest target collapses
  training. With the weight held fixed, 1b would always pick the weakest target.

**Step 1c: strength** (36 runs, seed 2025, scored on validation)
- Take the best (τ_t, profiles) setting from 1b.
- Grid: `sdr_weight` ∈ {0.01, 0.05, 0.1, 0.5, 1.5, 3.0} × weighting ∈ {entropy w with the floor, w = 1}.

**Step 1d: final** (45 runs)
- The best setting × {real, shuffled, random} × seeds 2021–2025 × 3 datasets.
- Report: semantic gain, and the gain over `s2dr-only`, at Recall/NDCG@20, overall and per activity quintile.
- **Go signal**: a semantic gain ≥ +0.002 Recall@20 (p < 0.05) on at least 2 datasets.

## Diagnostic 2: how much can the profiles add where CF is weak?

**Inputs.** LightGCN+S2DR CF embeddings from seeds 2021–2025. Rerun `s2dr-only --save_topk 50` if
`save/tier2/` does not hold them (15 runs). The raw profiles come from `data/<d>/`.

**Content-only scorers** (raw 1536-d profiles, frozen):
1. **User–item cosine**: the same as the `nowhiten` `llm` scorer (0.0378 on amazon). Recomputed here as a reference.
2. **Item kNN**: score(u, j) = the mean cosine between j's profile and the profiles of the user's training items,
   using each item's top 50 neighbours.
3. **Two-tower**: one MLP per side (1536 → 256 → 64), trained with BPR on the training graph with no ID
   embeddings. Early stopping on validation; seeds 2021–2025.

**Fusion.** Z-score each scorer per user, then score = (1 − α)·CF + α·content, with α ∈ {0, 0.05, …, 0.5}.
- **Global α**: one value, picked on validation.
- **Activity gate**: one α per activity quintile, picked on validation.
- **Oracle gate**: the best α per user, picked on test. This is a ceiling and is labelled as one.

**Report.** Recall/NDCG@20, overall, per activity quintile, and for head vs tail items. Each scorer and fusion is run
with real, shuffled and random profiles, so every row has a semantic part.
- **Go signal**: the activity gate adds ≥ +0.003 Recall@20 over LightGCN+S2DR for Q1 users, with real profiles above
  both controls.
- The gap between the activity gate and the oracle gate shows how much a better gate could still add.

## Diagnostic 3: does the profile gain grow as CF gets weaker?

**Split.** For each dataset, take test users with at least 4 training interactions and pick 50% of them at random,
using a fixed split seed that does not depend on the training seed. These are the **sparsified users**. Keep a
fraction f of their training interactions:
- f ∈ {1.0, 0.5, 0.25}, or exactly 1 interaction;
- at least 1 interaction is always kept;
- validation and test sets do not change;
- the other 50% of test users are left as they are and serve as the within-run control.

**Profile conditions.**
- **Leaky**: the real user profile (written from the full history). This is a ceiling.
- **Leak-free**: the user profile is replaced by the mean of the kept items' profiles. For ProEx, all 4 views use it.
- **Random**: the usual content-free control.
- Item profiles stay real in the leaky and leak-free conditions, since items lose no information.
- **LLM graph for ProMax**:
  - Leak-free condition: drop the LLM-added edges of sparsified users, because they were also produced from the full
    history.
  - Both conditions: drop any LLM-added edge that equals a removed interaction, so removed items cannot come back as
    LLM edges.

**Runs** (seeds 2021–2023)
- Per dataset, per f, per seed:
  - LightGCN+S2DR (×1)
  - ProEx, leaky / leak-free / random (×3)
  - ProMax, leaky / leak-free / random (×3)
- New f values: 3 values × 7 × 3 seeds × 3 datasets = 189 runs.
- Leak-free at f = 1.0: 2 × 3 × 3 = 18 runs. The other f = 1.0 cells reuse existing runs.
- The diagnostic 2 fused scorer (the activity gate) is computed offline from the LightGCN+S2DR embeddings of each f.

**Report.** For each f, Recall/NDCG@20 on sparsified vs untouched users; the leaky and leak-free semantic gain; and
the gain over LightGCN+S2DR.
- **Go signal**: the leak-free semantic gain on sparsified users grows as f shrinks.
- If the gain appears only with leaky profiles, it comes from profile leakage, not from the LLM.

## Code changes needed (in `encoder/`)

| Where | Change |
|---|---|
| `models/general_cf/lightgcn_promax.py` (CRLF) | Add the w / H_norm statistics to the `losses` dict, so the trainer logs their per-epoch mean. Add a `sdr_tau` config key (target temperature, default 0.2). |
| `config/configurator.py` | Add the flags `--sdr_tau`, `--sdr_weight`, `--entropy_weight`, `--whiten`, `--rag_graph`, so the 1b/1c grids need no new `VARIANTS` entries. Add a `--data_suffix` flag to load `data/<d><suffix>/`. |
| `trainer/trainer.py` | Store the best validation Recall/NDCG@20 in each `ablation.jsonl` record (needed for selection). |
| `make_sparse.py` (new) | Writes `data/<d>-sp<f>/`: a cut `trn_mat.pkl`, the same val/test, item profiles linked, leaky and leak-free user profiles, both versions of the LLM graph, and the list of sparsified users. |
| `diag_content.py` (new) | The 3 content-only scorers, fusion, activity and oracle gates, and per-quintile and head/tail reports for diagnostic 2 and the scorer part of diagnostic 3. |
| `run_tier4.sh`, `summarize_tier4.py` (new) | Grid runner with failed runs written to `save/result/tier4_failed.txt`, and the report written to `save/result/tier4_summary.md`. |

Each change gets a CPU smoke test on the Mac first. For example, `s2dr-only` with the new flags at their default
values must reproduce the current record exactly, before any server run.

## Budget

At the measured average of about 0.2 GPU-hours per run:

| diagnostic | runs | GPU-hours |
|---|---|---|
| 1 (a–d) | 141 | ~28 |
| 2 | 15 + MLP training | ~4 |
| 3 | 207 | ~43 |
| **total** | **~363** | **~75** |

Wall-clock time depends on how many runs share the GPUs. With 4 parallel runs, it is about 19 hours.

## Results go to

- `FINDINGS.md`, a new section "Tier 4".
- `RESEARCH_LOG.md`, with the code changes and run history.
- ADR 0001 gets an outcome line: go, or no-go.
