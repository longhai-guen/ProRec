# Research log: LLM profiles vs collaborative filtering (ProRec)

Last updated: 2026-11-26

**Research question.** Does the LLM latent space (LLM-generated user/item profiles) complement collaborative filtering (CF), or do LLM-enhanced recommenders gain for other reasons?

**Current answer.** See [FINDINGS.md](FINDINGS.md) for the full write-up of Tiers 1–3 and the LightGCN+S2DR baseline (2026-11-24). In short: most of the gain of ProEx and ProMax is content-free; a pure-CF LightGCN+S2DR baseline recovers 84–104% of ProMax's gain and beats ProEx on all three datasets; the semantic gain is at most about +0.004 Recall@20 and significant on only 2 of 6 method × dataset pairs.

**Next (2026-11-25).** Tier 4 diagnostics are planned in [docs/tier4-plan.md](docs/tier4-plan.md), with the go/no-go rule for designing an LLM method in [ADR 0001](docs/adr/0001-diagnostics-before-llm-method.md). Diagnostic 1 code is written and smoke-tested on CPU (2026-11-26); nothing has run on the server yet. Smoke-test notes: with every new flag at its default, `graph-train` and `s2dr-only` reproduce the Tier 3 code's test numbers exactly. With fake 64-d profiles, w already sits at the 0.1 floor for 100% of users at τ = 0.2 (whitened), and τ = 0.02 lifts it to about 0.8. The floor may therefore come from the target temperature itself, whatever the profiles contain.

---

## 1. Code changes (uncommitted, in `encoder/`)

| File | Change |
|---|---|
| `config/configurator.py` | New flags `--profile {real,none,shuffled,random}` and `--score {model,cf,llm,all}`; `perturb_profile()` (seeded shuffle / Gaussian noise with matching mean and std); guards so `none` and non-`model` scores are LightGCN-only |
| `models/general_cf/lightgcn_proex.py` | `profile=none` → plain LightGCN (BPR + reg); extra CoT profiles perturbed the same way; scorers `cf` (CF dot product), `llm` (MLP-projected, mean-pooled profiles), `model` (fused); `.cuda()` → `configs['device']` |
| `models/general_cf/lightgcn_promax.py` | `profile=none` → plain LightGCN; scorer `llm` = cosine of PCA-whitened profiles; `model` = `cf` by design |
| `models/base_model.py` | `self.score_mode = 'model'` |
| `trainer/trainer.py` | Final test runs once per scorer; tagged result line + JSON record appended to `save/result/ablation.jsonl` |
| `trainer/logger.py` | Logs go to `save/log/<model>[_profile-<p>]/<dataset>_seed<seed>_<time>.log` |
| `config/modelconf/lightgcn_proex.yml` | `k: [5, 10, 20]` (otherwise identical to ProMax's config, so one LightGCN baseline serves both) |
| `run_tier1.sh` (new) | Grid runner, all lists overridable by env (`CUDA`, `DATASETS`, `SEEDS`, `MODELS`, `PROFILES`); failures → `save/result/tier1_failed.txt` |
| `summarize_ablation.py` (new) | Mean ± std over seeds per dataset × model × profile × score; `--k 5 10 20` by default; a rerun of the same seed replaces the earlier record |
| Tier 2 (2026-11-12): `configurator.py` | `--profile_side {both,user,item}` (the side not kept gets random noise); `--save_topk K` |
| Tier 2: `trainer.py`, `metrics.py` | With `--save_topk`, per-user top-K lists per scorer and the final CF/profile embeddings → `save/tier2/<dataset>/<model>_<tag>_seed<s>_{model,cf,llm,emb}.npz`; records carry `side` |
| Tier 2: `lightgcn_proex.py`, `lightgcn_promax.py` | `export_embeds()`; ProEx perturbs each side separately |
| `run_tier2.sh` (new) | GPU runner: none + {real, shuffled, random, real-user, real-item} × {ProEx, ProMax} × seeds, all with `--save_topk 50 --score all` |
| `analyze_tier2.py` (new) | Report per dataset → `save/result/tier2_<dataset>.md`: A overall, B user-activity quintiles, C head/tail items, D complementarity/serendipity metrics, E CKA + kNN overlap |
| Tier 3 (2026-11-14): `configurator.py` | `--variant` (component ablations, table `VARIANTS`); score `orig` for ProEx (fused with the original profile only, no test-time mean pooling), included in `--score all` |
| Tier 3: `lightgcn_promax.py` | `rag_graph {llm,train,shuffled,random}` (S2DR second view: LLM-added edges kept / removed / moved to other users / same count random), `entropy_weight`, `whiten {pca_white,raw}` |
| `run_tier3.sh`, `summarize_tier3.py` (new) | Each variant × {real, random} × 3 seeds; report → `save/result/tier3_summary.md` (content-free gain, semantic gain, change vs full) |
| Baseline (2026-11-19): `configurator.py`, `summarize_tier3.py` | Variant `s2dr-only` = pure-CF LightGCN+S2DR (`rag_graph=train`, `sdr_weight=0`, ProMax hyperparameters). No SDR, so profiles have no effect (smoke test: real == random to the last digit); run once per seed with `real`, 5 seeds. With `keep_rate` 1 both S2DR views are the same graph, so S2DR reduces to a uniformity loss. The summary reuses the real run as random for it, so "real vs full" = ProMax(full) gap |
| Tier 4 diag 1 (2026-11-26): `lightgcn_promax.py` | `sdr_tau` config key (SDR target temperature, default 0.2); per-batch SDR weight statistics `w_mean`, `w_q25/50/75`, `w_floor` (share of users at the 0.1 floor) and `h_norm` returned in the losses dict |
| Tier 4: `configurator.py` | Flags `--sdr_tau`, `--sdr_weight`, `--sdr_w {entropy,one}`, `--whiten {pca_white,raw}` (ProMax only, applied after `--variant`). A run that sets any of them gets a variant label such as `graph-train+tau0.05+sw0.5+w1+raw` and an `hp` field in its record, so it never merges with the plain Tier 3 variant. `--log_sdr` writes per-epoch loss and w statistics to the log file |
| Tier 4: `trainer.py` | Every record now carries the validation metrics at the best epoch (`val_recall`, `val_ndcg`), used to select settings |
| `run_tier4.sh`, `summarize_tier4.py` (new) | Stages `STAGE=1a/1b/1c/1d` of diagnostic 1 (see [docs/tier4-plan.md](docs/tier4-plan.md)). The summary selects by validation Recall@20 at seed 2025 and writes `save/result/tier4_best.json`, which the next stage reads; report → `save/result/tier4_summary.md` |

Line endings: `lightgcn_proex.py`, `lightgcn_promax.py`, `lightgcn_proex.yml` are CRLF; keep them that way.

### How to run (server `nokton`, from `encoder/`)

```bash
CUDA=4 DATASETS=yelp nohup bash run_tier1.sh > tier1_yelp.out 2>&1 &
python summarize_ablation.py            # or --k 20 --metric recall
CUDA=4 DATASETS=amazon nohup bash run_tier2.sh > tier2_amazon.out 2>&1 &
python analyze_tier2.py --dataset amazon   # → save/result/tier2_amazon.md
```

Tier 2 code was smoke-tested on the Mac (CPU, 2 epochs, fake profiles): all 11 conditions write their files, and the analysis numbers agree with `ablation.jsonl`.

### Early stopping (unchanged from the authors)

Validate every 3 epochs on validation Recall@20 (last `k`); strict improvement resets patience; stop after 20 evaluations without improvement (60 epochs); reload the best checkpoint; test once per scorer. `cf`/`llm` are scored at the checkpoint chosen by `model`. Kept as is: a stricter rule would undertrain the LightGCN baseline and inflate the apparent LLM gain.

---

## 2. Tier 1 ablation design

Same model, losses, hyperparameters, early stopping and seeds (2021–2025); only the profile input changes.

| Condition | Profiles given to the model | A gain here means |
|---|---|---|
| `none` | No profiles: plain LightGCN | Baseline |
| `real` | Original LLM profile embeddings | Full effect of the method |
| `shuffled` | Real embeddings permuted across users/items | Gain without matching content (keeps the embedding geometry) |
| `random` | Gaussian noise, same per-dimension mean/std | Gain from extra losses and parameters alone |

Scorers from one trained model: `cf` (CF vectors only), `llm` (profile vectors only), `model` (the method's fused score).

**Content-free** = the profile carries no correct information about that user/item (shuffled or random). Gain decomposition, Recall@20, `model` score:

- total gain = real − none
- content-free gain = max(shuffled, random) − none
- semantic gain = real − max(shuffled, random)
- content-free share = content-free gain ÷ total gain

---

## 3. Results (preliminary)

Source: `encoder/save/result/tier1_summary_no_real.txt` (5 seeds; steam 4 at copy time) and `tier1_summary_with_real.md`. `real` rows come from earlier old-code runs: 1–3 runs each, seed 2025, `model` score only.

### Recall@20 (change vs LightGCN)

| Method | Profile | Amazon | Yelp | Steam |
|---|---|---|---|---|
| LightGCN | none | 0.1471 | 0.1165 | 0.1446 |
| ProEx | real | 0.1529 (+4.0%) | 0.1311 (+12.5%) | 0.1474 (+1.9%) |
| ProEx | shuffled | 0.1488 (+1.2%) | 0.1270 (+9.0%) | 0.1434 (−0.8%) |
| ProEx | random | 0.1513 (+2.9%) | 0.1289 (+10.6%) | 0.1466 (+1.4%) |
| ProMax | real | 0.1623 (+10.4%) | 0.1331 (+14.2%) | 0.1511 (+4.5%) |
| ProMax | shuffled | 0.1579 (+7.3%) | 0.1320 (+13.3%) | 0.1494 (+3.3%) |
| ProMax | random | 0.1585 (+7.7%) | 0.1324 (+13.6%) | 0.1491 (+3.1%) |

Seed std of Recall@20: 0.0002–0.0015. The `llm`-only score with perturbed profiles is 0.002–0.006 (chance level, no leakage).

### Gain decomposition

| Dataset · Method | Total gain | Semantic gain | Content-free share |
|---|---|---|---|
| Amazon · ProEx | 0.0058 | 0.0016 | 72% |
| Amazon · ProMax | 0.0152 | 0.0038 | 75% |
| Yelp · ProEx | 0.0146 | 0.0022 | 85% |
| Yelp · ProMax | 0.0166 | 0.0007 | 96% |
| Steam · ProEx | 0.0028 | 0.0008 | 71% |
| Steam · ProMax | 0.0065 | 0.0017 | 74% |

### Findings

1. Both methods beat LightGCN everywhere, but 71–96% of the gain needs no meaningful profile.
2. The semantic part is small (≤ +0.0038) and dataset-dependent: clearest for ProMax on Amazon/Steam (~3.5σ); within noise for Yelp·ProMax and Steam·ProEx.
3. ProMax with shuffled profiles beats ProEx with real profiles on all datasets: the method's objective matters more than profile use.
4. For ProEx, `random` > `shuffled`: realistic but wrong profiles mislead more than noise.
5. For ProEx with perturbed profiles, `cf` ≥ `model`: the content-free gain lives in the CF embeddings; adding profiles at test time only adds noise.
6. A comparison against plain LightGCN alone overstates the LLM contribution by 3.5×–24× (total ÷ semantic gain).

### Caveats

- `real` rows are single runs compared with 5-seed means; no paired tests yet.
- The complementarity test (fused vs CF-only vs profile-only on `real`) is not done: old runs recorded only `model`.
- ProMax's shuffled/random runs still use the LLM-reranked graph (`new_trn_rag_mat.pkl`) in S2DR, so its content-free gain may contain LLM signal.
- One backbone, one hyperparameter setting.

---

## 4. Professor's feedback (received ~2026-10-28)

1. **Serendipity literature → concrete complementarity metrics.** Serendipity = relevant + unexpected, with unexpectedness measured against a baseline recommender. Using CF as that baseline makes complementarity measurable. Proposed metrics (per user; C = CF-only top-K, L = profile-only top-K, F = fused top-K, T = test items):
   - Disagreement: list overlap |C∩L| / |C∪L|
   - Complementary potential: HitJaccard; complementary hit ratio |hits(L) \ hits(C)| / |hits(C) ∪ hits(L)|; oracle-union recall
   - Realized utility: serendipity w.r.t. CF |(F \ C) ∩ T| / K; unexpectedness |F \ C| / K; Recall(F) − Recall(C)
   - Novelty of complementary hits (popularity / long-tail share)
   - Every metric reported relative to the content-free (shuffled/random) value
   - Papers to read: Kotkov et al. 2016 (survey, KBS); Kaminskas & Bridge 2016 (TiiS); Ge et al. 2010 (RecSys); Murakami et al. 2008; Vargas & Castells 2011 (RecSys); Ziarani & Ravanmehr 2021 (JCST); recent LLM-serendipity work. Verify citations.
2. **Dataset types and characteristics.** Current datasets (RLMRec preprocessed, random 60/20/20 split):

   | | Amazon (Books) | Yelp | Steam |
   |---|---|---|---|
   | Users × items | 11,000 × 9,332 | 11,091 × 11,010 | 23,310 × 5,237 |
   | Interactions / density | 200,860 / 0.20% | 277,535 / 0.23% | 525,922 / 0.43% |
   | Per-user median / p10 | 11 / 7 | 16 / 10 | 15 / 10 |
   | Users ≤ 10 interactions | 44% | 14% | 14% |
   | Top-20% items' share | 45% | 47% | 73% |

   Few truly cold users and no temporal split, so the sparse regime can't be tested yet. Proposal: controlled sparsity on existing data first (keep 20/40/60/80% of each user's training interactions), then datasets chosen on two axes, CF density × semantic information / LLM knowledge (e.g. MovieLens-1M as a dense high-knowledge control; Amazon Reviews 2023, Goodreads or MIND for the sparse text-rich regime).

---

## 5. Next steps (in order)

0. Done 2026-11-24: ran the LightGCN+S2DR baseline (`s2dr-only`, 3 datasets × 5 seeds). If it matches ProMax, ProMax's gain needs no LLM signal.
   Result (Recall@20 / NDCG@20, paired over 5 seeds, t-test p<0.05 marked *):
   - LightGCN+S2DR beats LightGCN on every dataset (amazon +0.0122*, yelp +0.0167*, steam +0.0056*); it covers 84–104% of ProMax's Recall@20 gain.
   - ProMax − LightGCN+S2DR: amazon R +0.0012 (n.s.) / N +0.0018*; yelp −0.0006 / −0.0001 (n.s.); steam +0.0011* / +0.0009*. All the LLM signal (profiles + LLM graph) adds is at most about 0.001–0.002.
   - LightGCN+S2DR beats ProEx on all three datasets (R@20 −0.0055*, −0.0032*, −0.0025* for ProEx − baseline).
1. Rerun `real` for ProEx and ProMax with 5 seeds and `--score all` (30 runs, code ready) → complementarity test and paired tests.
2. Save per-user metrics and top-K lists per scorer → serendipity-style complementarity metrics and user-activity / item-popularity strata (RQ1).
3. Controlled sparsity option in the data loader.
4. ProMax with the reranked graph shuffled or removed.
5. Add DMRec (Mult-VAE: `none` = plain Mult-VAE; `cf`/`llm` = decode `mu_src` / `mu_llm`) and more LLM-enhanced methods (e.g. RLMRec).
6. Small fixes: de-duplicate `ablation.jsonl` in `summarize_ablation.py`; `BASELINE=0` toggle in `run_tier1.sh`; commit the code.

### Publication plan

- Near term: evaluation / reproducibility paper ("content-free controls for LLM-enhanced CF") for SIGIR 2027 short or reproducibility track (deadline usually late January; verify), fallback RecSys 2027 reproducibility track. Needs 4–6 methods, 3–5 datasets incl. controlled sparsity, paired tests, complementarity metrics.
- Later: full paper with the reliability–complementarity selective transfer method (RQ2/RQ3).

---

## 6. Outputs produced

| What | Where |
|---|---|
| Results without `real` | `encoder/save/result/tier1_summary_no_real.txt` |
| Merged table with `real` | `encoder/save/result/tier1_summary_with_real.md` |
| Written findings | `encoder/save/result/tier1_findings.md` |
| Results slides (3, deck style, with notes) | `encoder/save/result/tier1_results_slides.pptx` → imported as slides 8–10 of the proposal deck |
| Findings doc (chart + tables) | https://claude.ai/artifact/KjBBjbwrChqRsYrEhQUevJ |
| Lab report talk script + Q&A | https://claude.ai/artifact/4P8YJukBHh8nWAkCePADBW |
| Proposal deck | https://docs.google.com/presentation/d/10iQdrCOJseJlJvDX-TioYj-ZRk_RjTdUUQ-Im3zMNYI/edit |
