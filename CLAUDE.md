# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Start every task with grill-with-docs

Before starting any new task (a feature, fix, refactor, plan, design, prompt or anything related), run grill-with-docs
first, before reading code in depth or writing any. It is user-only (`disable-model-invocation`), so do what it does:
call the Skill tool for `grilling`, then `domain-modeling`. Interview the user until the task is sharp, and record terms
in `GLOSSARY.md` and decisions as ADRs in `docs/adr/` along the way. Skip it only when the user says so for that task.

## What this project is

ProRec is the authors' library of LLM-profile recommenders: DMRec (SIGIR'25), ProEx (KDD'26) and ProMax (SIGIR'26), built
on the RLMRec training framework. This fork is used for research on how LLM latent spaces (profile embeddings) work with
CF recommenders. It asks whether the LLM profiles really improve CF, or whether the reported gains come from other
parts of the methods. The work so far is ablations of ProEx and ProMax on LightGCN. Their status, design and findings are
in `RESEARCH_LOG.md`, which is the running record; the consolidated results are in `FINDINGS.md`. Read it before planning new experiments, and update it when code or
results change.

## Running

There is no build, lint or test suite. All code runs from `encoder/`, because the paths `../data/<dataset>/` and
`../encoder/config/modelconf/` are relative to it.

```bash
cd encoder
python train_encoder.py --model lightgcn_proex --dataset amazon --cuda 0      # one training + final test
python train_encoder.py --model lightgcn_promax --dataset yelp --device cpu   # CPU smoke test
```

Ablation flags (defined in `config/configurator.py`):
- `--profile {real,none,shuffled,random}`: `none` = plain LightGCN with the method's losses off. `shuffled` permutes
  profile rows across entities. `random` = Gaussian noise with the real per-dimension mean and std.
- `--profile_side {both,user,item}`: keep real profiles on one side only; the other side gets `random`.
- `--score {model,cf,llm,orig,all}`: which scorer the final test uses. `all` evaluates every scorer after one
  training run. `orig` is ProEx only (original profile, no mean pooling).
- `--variant <name>`: Tier 3 component ablation from the `VARIANTS` dict (e.g. `nos2dr`, `graph-train`, `p1`).
- `--save_topk K`: also save each user's top-K lists and the embeddings to `save/tier2/<dataset>/`.
- `--seed`: the experiments use seeds 2021–2025 (Tier 3 uses 2021–2023).

Batch runners and summaries (environment variables like `CUDA`, `DATASETS`, `SEEDS`, `MODELS` override the defaults):
- `run_tier1.sh` → `summarize_ablation.py` (profile × scorer table)
- `run_tier2.sh` → `analyze_tier2.py --dataset <d>` (writes `save/result/tier2_<d>.md`)
- `run_tier3.sh` → `summarize_tier3.py` (writes `save/result/tier3_summary.md`)

Every test run appends one JSON record to `save/result/ablation.jsonl`. The summary scripts dedupe it by key, and the
latest record wins. A failed run's arguments go to `save/result/tier*_failed.txt`.

## Where experiments run

Real runs happen only on the GPU server `nokton` (`~/projects/ProRec`, conda env `prorec`). Do not run real experiments
locally. Use CPU runs only as smoke tests. Datasets and outputs are not in git: `/data`, `encoder/save/log`,
`encoder/save/result` and `encoder/save/tier2` are ignored. On the server, `data/` holds links to a shared folder of
profile embeddings, which `git reset --hard` or `git clean` can break. Each dataset needs `trn/val/tst_mat.pkl`,
`usr_emb_np.pkl`, `itm_emb_np.pkl`, the ProEx extra views `usr/itm_emb_np_{1,2,3}.pkl`, and for ProMax
`new_trn_rag_mat.pkl` (the training graph plus edges added by the LLM). If these are missing, every profile run fails at
config load with `FileNotFoundError`, and only the `none` runs succeed.

## Architecture

`train_encoder.py` runs five steps: seed → data handler → model → logger → trainer.
- **Config** (`config/configurator.py`): `configs` is a global dict, built at import time from argparse plus
  `config/modelconf/<model>.yml`. Each yml has a per-dataset block under `model:` (e.g. `model.amazon.sdr_weight`); the
  model reads it as `self.hyper_config`. The configurator also loads the profile embeddings, applies `perturb_profile`,
  and applies `VARIANTS` overrides: a key goes into the dataset block if it exists there, otherwise into `configs['model']`.
- **Model loading** (`models/bulid_model.py`, note the spelling): imports `models.<data.type>.<model.name>` and finds the
  class whose name matches case-insensitively. A new model therefore needs a module file, a class with the same name and
  a yml with the same name.
- **Models** (`models/general_cf/`):
  - `lightgcn_proex` adds MLP-projected profiles to the CF embeddings, with 4 profile views (original + 3 CoT
    extrapolations), environment losses, and mean pooling at test time.
  - `lightgcn_promax` uses PCA-whitened profiles in training only: SDR is entropy-weighted distillation toward profile
    similarities, and S2DR is a contrastive loss against a second view built on `new_trn_rag_mat.pkl`. At test time the
    score is CF only, so its `model` and `cf` scorers are identical.
  - `mult_vae_*` / `cvga_*` are the DMRec variants (generative); the ablations do not use them.
  - Both LightGCN models expose `score_mode` (scorer selection in `full_predict`) and `export_embeds()` for Tier 2.
- **Trainer** (`trainer/trainer.py`): validates every `test_step` epochs and stops early after `patience` checks without
  improvement. The final test loops over the requested scorers, logs each, and appends it to `ablation.jsonl`.
  `trainer/logger.py` names log directories by `profile_tag()` (e.g. `real-user`, `random@graph-train`).

## Conventions

- `models/general_cf/lightgcn_proex.py` and `lightgcn_promax.py` have CRLF line endings. Keep CRLF when editing them.
- Every new ablation must be measured against content-free controls (`shuffled` / `random`) with seeds paired. A gain
  counts as semantic only when real − best control > 0 across seeds.
- For architecture work use /improve-codebase-architecture (vocabulary: codebase-design); for a strict quality review use /thermo-nuclear-code-quality-review.
