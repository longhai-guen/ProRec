# Findings: do LLM profiles really improve CF?

Last updated: 2026-11-24. All Tier 1, 2 and 3 experiments and the LightGCN+S2DR baseline are complete. Terms
are defined in [GLOSSARY.md](GLOSSARY.md). The code and run history are in [RESEARCH_LOG.md](RESEARCH_LOG.md).

## Answer in brief

For ProEx (KDD'26) and ProMax (SIGIR'26) on LightGCN, most of the reported gain does not need the LLM profiles.

1. **Most of the gain is content-free.** If real profiles are replaced by shuffled ones or by noise with the same
   statistics, ProEx keeps 42–93% of its Recall@20 gain over LightGCN and ProMax keeps 83–99%. The **semantic gain**
   (real − best control) is significant only for ProEx on amazon (+0.0038) and ProMax on steam (+0.0011). It is zero
   on yelp, the dataset with the largest total gains.
2. **ProMax's gain is explained by S2DR, which works as a pure-CF regularizer.** LightGCN+S2DR (training graph only,
   no SDR, no profiles) recovers 84–104% of ProMax's Recall@20 gain. Everything ProMax adds on top of it, profiles and
   LLM edges together, is at most +0.0012 Recall@20 / +0.0018 NDCG@20.
3. **That pure-CF baseline beats ProEx** on all three datasets in every seed (Recall@20 −0.0025 to −0.0055 for ProEx).
4. **Profiles alone are almost useless for retrieval.** Scoring by profile similarity alone gives Recall@20 of
   0.002–0.012, against about 0.15 for CF. The two candidate lists barely overlap, but giving the extra slots to CF
   beats giving them to the profiles.
5. **Where the semantic gain exists, it sits with low-activity users and tail items**, and it needs both user and
   item profiles to be real (for ProEx).

## Setup

- **Data**: amazon (book), yelp and steam in the authors' splits. Test users: 9,860 / 10,593 / 22,345. Training edges:
  120,464 / 166,620 / 316,190.
- **Methods**: LightGCN (backbone, plus the `none` setting), ProEx and ProMax with the authors' hyperparameters, and the
  pure-CF baseline LightGCN+S2DR (`--variant s2dr-only`, ProMax hyperparameters).
- **Profile settings** (`--profile`, `--profile_side`):
  - `real`
  - `shuffled`: rows permuted, so each entity gets another entity's profile
  - `random`: Gaussian noise with the real per-dimension mean and std
  - `real-user` / `real-item`: real on one side, random on the other
- **Scorers** (`--score`):
  - `model`: what the method uses
  - `cf`: CF embeddings only
  - `llm`: profiles only
  - `orig` (ProEx only): the original profile, with no mean pooling over the 4 views
- **Protocol**:
  - Seeds 2021–2025 (Tier 3 variants: 2021–2023).
  - Validation every 3 epochs, patience 20, at most 3000 epochs. These are the authors' settings, kept so the
    LightGCN baseline is not undertrained.
  - Metrics: Recall@20 and NDCG@20 over all items, with training items masked.
- **Statistics**:
  - Means ± sample std over seeds.
  - Differences are paired by seed.
  - `*` = two-sided paired t-test, p < 0.05.
  - `(w/n)` = the number of seeds where the difference is positive.

## Tier 1: overall results and gain decomposition

### Recall@20 / NDCG@20

| model | profile | amazon R | amazon N | yelp R | yelp N | steam R | steam N |
|---|---|---|---|---|---|---|---|
| LightGCN | – | 0.1471 ± 0.0016 | 0.0892 ± 0.0009 | 0.1165 ± 0.0009 | 0.0739 ± 0.0006 | 0.1440 ± 0.0014 | 0.0913 ± 0.0007 |
| **LightGCN+S2DR** | – | 0.1593 ± 0.0005 | 0.0986 ± 0.0003 | **0.1332 ± 0.0007** | **0.0852 ± 0.0006** | 0.1495 ± 0.0007 | 0.0958 ± 0.0002 |
| ProEx | real | 0.1538 ± 0.0014 | 0.0938 ± 0.0003 | 0.1300 ± 0.0010 | 0.0822 ± 0.0007 | 0.1470 ± 0.0008 | 0.0937 ± 0.0004 |
| ProEx | shuffled | 0.1483 ± 0.0009 | 0.0912 ± 0.0009 | 0.1275 ± 0.0011 | 0.0810 ± 0.0007 | 0.1437 ± 0.0004 | 0.0912 ± 0.0002 |
| ProEx | random | 0.1500 ± 0.0020 | 0.0920 ± 0.0006 | 0.1292 ± 0.0005 | 0.0825 ± 0.0005 | 0.1463 ± 0.0009 | 0.0933 ± 0.0004 |
| ProMax | real | **0.1605 ± 0.0013** | **0.1004 ± 0.0006** | 0.1326 ± 0.0004 | 0.0851 ± 0.0005 | **0.1506 ± 0.0005** | **0.0967 ± 0.0003** |
| ProMax | shuffled | 0.1579 ± 0.0012 | 0.0989 ± 0.0006 | 0.1320 ± 0.0007 | 0.0848 ± 0.0004 | 0.1495 ± 0.0005 | 0.0959 ± 0.0004 |
| ProMax | random | 0.1585 ± 0.0013 | 0.0988 ± 0.0005 | 0.1324 ± 0.0006 | 0.0850 ± 0.0006 | 0.1491 ± 0.0005 | 0.0958 ± 0.0003 |

### Decomposition (Recall@20 / NDCG@20, paired, 5 seeds)

Total = real − LightGCN. Content-free = best control − LightGCN. Semantic = real − best control. The best control is
the one with the higher Recall@20: random everywhere except ProMax on steam, where it is shuffled.

| method | dataset | total gain | content-free gain | semantic gain | semantic share of R gain |
|---|---|---|---|---|---|
| ProEx | amazon | +0.0066* / +0.0046* | +0.0028* / +0.0028* | **+0.0038* (5/5) / +0.0018* (5/5)** | 58% |
| ProEx | yelp | +0.0136* / +0.0084* | +0.0127* / +0.0086* | +0.0008 (3/5) / −0.0003 (2/5) | 6% |
| ProEx | steam | +0.0031* / +0.0024* | +0.0023* / +0.0020* | +0.0008 (4/5) / +0.0004 (4/5) | 26% |
| ProMax | amazon | +0.0134* / +0.0112* | +0.0114* / +0.0096* | +0.0020 (4/5) / +0.0016* (5/5) | 15% |
| ProMax | yelp | +0.0161* / +0.0112* | +0.0159* / +0.0111* | +0.0002 (2/5) / +0.0001 (1/5) | 1% |
| ProMax | steam | +0.0066* / +0.0054* | +0.0055* / +0.0046* | **+0.0011* (5/5) / +0.0008* (5/5)** | 17% |

Every total gain is significant. The content-free gain is significant everywhere. The semantic gain is significant
in Recall only for ProEx on amazon and ProMax on steam, and in NDCG also for ProMax on amazon.

### Scorers (Recall@20)

| dataset | method | profile | model | cf | llm | orig |
|---|---|---|---|---|---|---|
| amazon | ProEx | real | 0.1538 | 0.1496 | 0.0116 | 0.1528 |
| amazon | ProEx | random | 0.1500 | 0.1504 | 0.0025 | 0.1485 |
| yelp | ProEx | real | 0.1300 | 0.1273 | 0.0043 | 0.1292 |
| yelp | ProEx | random | 0.1292 | 0.1290 | 0.0022 | 0.1279 |
| steam | ProEx | real | 0.1470 | 0.1472 | 0.0097 | 0.1458 |
| steam | ProEx | random | 0.1463 | 0.1473 | 0.0058 | 0.1427 |
| amazon | ProMax | real / random | 0.1605 / 0.1585 | = model | 0.0041 / 0.0019 | – |
| yelp | ProMax | real / random | 0.1326 / 0.1324 | = model | 0.0019 / 0.0019 | – |
| steam | ProMax | real / random | 0.1506 / 0.1491 | = model | 0.0038 / 0.0039 | – |

- Real profiles in ProEx carry some retrieval signal on their own (`llm` is 1.7–4.6× random), but at 3–8% of CF's
  recall.
- ProEx's fused score helps over its own CF part only with real profiles (amazon +0.0042, yelp +0.0027); on steam it
  does not.
- ProMax scores with CF only at test time, so `model` = `cf`. Its PCA-whitened profiles have no retrieval signal
  (`llm` ≈ random on yelp and steam). Without whitening (Tier 3 `nowhiten`), the same profiles reach 0.0378 on amazon,
  so whitening removes the similarity structure.

## Tier 2: where the gain comes from

Full reports: `encoder/save/result/tier2_{amazon,yelp,steam}.md` (not in git). All numbers are Recall@20 over 5 seeds.

### One side vs both sides (gain over random, paired)

| dataset | method | real-user | real-item | real (both) |
|---|---|---|---|---|
| amazon | ProEx | −0.0000 (2/5) | +0.0008 (4/5) | **+0.0038* (5/5)** |
| amazon | ProMax | −0.0009 (1/5) | **+0.0019* (5/5)** | +0.0020 (4/5) |
| yelp | ProEx | +0.0006 | +0.0007 | +0.0008 |
| yelp | ProMax | +0.0002 | +0.0005* | +0.0002 |
| steam | ProEx | +0.0003 | +0.0009 | +0.0008 |
| steam | ProMax | **+0.0012* (5/5)** | +0.0003 | **+0.0015* (5/5)** |

- In ProEx, the amazon semantic gain appears only when **both** sides are real. It comes from user and item profiles
  matching each other in the LLM space, not from either side alone.
- In ProMax, one side carries the gain: the item side on amazon, the user side on steam.

### User activity (training-interaction quintiles)

The content-free share of the total gain grows with user activity. The profile content matters most for sparse users:

| method / dataset | Q1 (least active) | Q2 | Q3 | Q4 | Q5 (most active) |
|---|---|---|---|---|---|
| ProEx amazon | 1% | 29% | 44% | 55% | 64% |
| ProMax amazon | 68% | 48% | 95% | 96% | 104% |
| ProMax steam | 44% | 83% | 86% | 88% | 99% |
| ProEx yelp | 81% | 99% | 91% | 94% | 100% |
| ProMax yelp | 99% | 99% | 97% | 98% | 101% |

- On amazon, the semantic gain for ProEx is +0.0043 in Q1 and +0.0069 in Q3, against +0.0024 in Q5.
- On steam, ProMax's semantic gain is +0.0019 in Q1 and +0.0000 in Q5.
- On steam, ProEx gives no gain at all to the least active users (Q1 total gain −0.0006).
- On yelp, no quintile shows a semantic gain.
- The per-quintile numbers are noisy (std 0.001–0.005).

### Item popularity (head = top 20% of items by training interactions)

| dataset | method | tail recall: LightGCN → real (random) | head recall: LightGCN → real | semantic tail gain |
|---|---|---|---|---|
| amazon | ProEx | 0.0278 → 0.0472 (0.0436) | 0.2699 → 0.2571 | +0.0037 |
| amazon | ProMax | 0.0278 → 0.0453 (0.0466) | 0.2699 → 0.2740 | −0.0014 |
| yelp | ProEx | 0.0057 → 0.0177 (0.0150) | 0.2271 → 0.2384 | +0.0027 |
| yelp | ProMax | 0.0057 → 0.0229 (0.0230) | 0.2271 → 0.2382 | −0.0001 |
| steam | ProEx | 0.0064 → 0.0099 (0.0096) | 0.1780 → 0.1800 | +0.0002 |
| steam | ProMax | 0.0064 → 0.0097 (0.0096) | 0.1780 → 0.1824 | +0.0007 |

- Both methods raise tail recall a lot: 1.6–1.7× on amazon and 3–4× on yelp.
- Content-free controls get almost the same tail boost. The tail lift is mostly a regularization effect.
- ProEx's semantic part favors the tail (amazon +0.0037, yelp +0.0027), but on amazon it costs head recall
  (0.2699 → 0.2571).

### Complementarity: are profile-only recommendations a useful second source?

ProEx, amazon, top-20 (C = CF list, L = profile list):

- **The lists barely overlap.** The overlap is 0.3% (0.1% with random profiles).
- **L adds few new hits.** Among all hits from either list, 4.9% are hits that only L finds (random profiles: 1.4%).
  When C finds nothing for a user, L rescues 2.6% of those users (random: 0.7%).
- **CF uses the extra slots better.** The union of C and L reaches Recall 0.158, while CF alone with the same 40 slots
  reaches 0.220.
- **Profile-only hits are mostly tail.** 83% of L's hits that C misses are tail items (random: 60%).

The other datasets and ProMax show the same pattern, with a smaller real − random gap. The profile tower is not a
complementary retriever. It helps, where it helps, by shaping the CF space in training.

### Geometry (linear CKA and kNN@10 overlap, 3,000 sampled entities)

- **Real profiles leave a trace in the CF space; controls do not.** CKA between CF and profile spaces with real
  profiles: items 0.10–0.13 (ProEx) and 0.11–0.19 (ProMax). With shuffled or random profiles: about 0.01.
- **That trace alone does not mean better recommendations.** On yelp, ProMax's CF space aligns with the real profiles
  (item CKA 0.114) without any accuracy gain.
- **ProMax reshapes the CF space much more than ProEx**, compared with plain LightGCN from the same seed: item CKA
  0.66–0.87 against 0.91–0.97. Random profiles reshape it just as much, so the reshaping comes from the losses, not
  the profile content.

## Tier 3: which component carries the gain (Recall@20, 3 seeds; `full` uses 5)

### ProEx

| variant | amazon real vs full | yelp | steam | what it shows |
|---|---|---|---|---|
| `p1` (original profile only) | −0.0118* | −0.0077* | −0.0055* | The 4 profile views are essential. With one view ProEx falls below LightGCN on amazon and steam, and real < random (amazon semantic −0.0034*). |
| `noinv` (no variance loss) | +0.0004 | −0.0004 | −0.0002 | The invariance loss does nothing. |
| `noex` (no ex/uniformity loss) | −0.0053* | −0.0069* | −0.0072* | The ex loss is needed. Without it, the semantic gain turns negative on yelp (−0.0036) and steam (−0.0023*), and the profile scorer gets stronger while the fused model gets worse. |
| `orig − model` (test with the original profile only) | −0.0009 | −0.0008 | −0.0013 | Mean pooling over the 4 views at test time helps a little. What matters is training with 4 views. |

### ProMax

| variant | amazon real vs full | yelp | steam | what it shows |
|---|---|---|---|---|
| `nosdr` (no SDR) | −0.0002 | +0.0000 | −0.0008* | SDR adds almost nothing. Without SDR, real = random exactly: SDR is the only path for profile content into ProMax. |
| `nos2dr` (no S2DR) | −0.0200* | −0.0132* | −0.0050* | S2DR carries the gain. SDR alone is below LightGCN on amazon (content-free −0.0071*). |
| `graph-train` (S2DR on the training graph) | −0.0015* | +0.0004 | −0.0005 | The LLM-added edges add at most about 0.0015 (amazon). |
| `graph-shuffled` / `graph-random` | −0.0017 / −0.0007 | +0.0007 / +0.0009 | −0.0010 / −0.0005* | Moving the added edges to other users or to random items barely matters. |
| `nowhiten` (raw profiles in SDR) | −0.0003 | +0.0005 | −0.0003 | Whitening does not matter for accuracy. |
| `noentropy` (SDR weight w = 1) | −0.0968* | −0.0435* | −0.0567* | Training collapses (see the caveat below). |
| **`s2dr-only` = LightGCN+S2DR** | −0.0012 (5 seeds) | +0.0006 | −0.0011* | **The pure-CF baseline matches ProMax** (next section). |

**Mechanism.** The model config sets `keep_rate: 1.0` and `forward()` applies no edge dropout. So with the training
graph as the second view, both S2DR views are identical, and the InfoNCE alignment term is constant. S2DR then reduces
to a **uniformity loss** on the CF embeddings, in the spirit of DirectAU. This plain regularizer produces 84–104% of
ProMax's gain.

## The pure-CF baseline: LightGCN+S2DR (5 seeds)

| dataset | baseline − LightGCN (R / N) | ProMax − baseline (R / N) | ProEx − baseline (R / N) | baseline share of ProMax's R gain |
|---|---|---|---|---|
| amazon | +0.0122* (5/5) / +0.0094* (5/5) | +0.0012 (4/5) / **+0.0018* (5/5)** | −0.0055* (0/5) / −0.0048* (0/5) | 91% |
| yelp | +0.0167* (5/5) / +0.0113* (5/5) | −0.0006 (2/5) / −0.0001 (2/5) | −0.0032* (0/5) / −0.0029* (0/5) | 104% |
| steam | +0.0056* (5/5) / +0.0045* (5/5) | **+0.0011* (5/5) / +0.0009* (5/5)** | −0.0025* (0/5) / −0.0021* (0/5) | 84% |

- **What all LLM signals together add to ProMax** (profiles through SDR, plus LLM edges) is +0.0011 to +0.0018 on
  amazon and steam, and nothing on yelp.
- **ProEx loses to the pure-CF baseline in every seed on every dataset.** Its semantic gain on amazon (+0.0038) is real
  but smaller than what a plain uniformity regularizer gives (+0.0122).

## Synthesis

- **The answer to the research question is "barely".** In these two methods, the LLM latent space contributes a
  small, dataset-dependent gain: about 0–0.004 Recall@20, significant on 2–3 of 6 method × dataset pairs. Most of each
  method's improvement comes from its training objective acting as a regularizer: S2DR's uniformity in ProMax, and the
  multi-view noise plus ex loss in ProEx. Those regularizers work just as well with content-free profiles, or with no
  profiles at all.
- **Where real profiles help:**
  - low-activity users and tail items (amazon, steam)
  - for ProEx, only when user and item profiles are both real

  This is the cold-start niche the methods motivate, but the effect is small.
- **Fair comparisons need two baselines.** Profile methods should be compared with (a) content-free controls under
  the same objective, and (b) a CF baseline with the same regularizer. Against plain LightGCN alone, the whole gain gets attributed to
  the LLM, though the semantic part is only 1–58% of it.

## Caveats

- **Seed counts.** Tier 3 variants have 3 seeds; differences under about 0.001 there are not reliable. Tier 1/2, the
  `full` rows and the baseline have 5 seeds.
- **Hyperparameters.** All methods use the authors' hyperparameters. The baseline reuses ProMax's `s2dr_weight` and
  temperature without tuning, which, if anything, favors ProMax.
- **The `noentropy` collapse is probably a scale effect, not a property of entropy weighting** (hypothesis, not
  measured). The weight is `w = clamp(1 − H_norm, 0.1, 1)`. Whitened profiles carry almost no similarity signal, so
  the target distribution is near-uniform, H_norm ≈ 1, and w most likely sits at the 0.1 floor. Setting w = 1 then
  makes SDR about 10× stronger and pulls the CF space toward a near-uniform target. Confirming this needs a log of the
  w distribution on the server.
- **Scope.**
  - Only LightGCN-based ProEx and ProMax were studied.
  - DMRec (Mult-VAE) and other LLM-enhanced methods, such as RLMRec and KAR, are not covered yet.
  - Only three datasets, with the authors' splits; there is no controlled-sparsity study.
- **The LLM graph** adds about 20% edges over the training graph (median 2 per user, about 72% of users). Only about
  0.5% of the added edges are test items, so it cannot leak much test signal.

## Where the results are

On the Mac in `encoder/save/result/`; these files are not in git:
- `ablation.jsonl`: one record per run and scorer. All tables above are computed from it.
- `tier1_final.txt`: the full table at k = 5/10/20.
- `tier2_{amazon,yelp,steam}.md`: the full Tier 2 reports.
- `tier3_summary.md`: Tier 3 and the baseline.

To recompute: `python summarize_ablation.py`, `python analyze_tier2.py --dataset <d>` (needs `save/tier2/` from the
server), `python summarize_tier3.py`. Run them from `encoder/`.
