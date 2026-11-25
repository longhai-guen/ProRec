# LLM profiles vs collaborative filtering

This project tests whether LLM-derived user/item profiles really improve CF recommenders, or whether the gains of
published profile methods come from parts that need no profile content.

## Profiles and controls

**Profile**:
An LLM-generated text description of a user or item, used as a fixed embedding vector.
_Avoid_: LLM embedding, semantic embedding

**Real profile**:
The profile the LLM produced for that exact user or item.

**Content-free control**:
A stand-in for real profiles that keeps their statistics but carries no information about the entity: shuffled (another
entity's profile) or random (noise with the same per-dimension mean and spread).
_Avoid_: fake profile, noise baseline

## Gains

**Total gain**:
A method's metric minus plain LightGCN's, paired by seed.

**Content-free gain**:
The part of the total gain a method still gets with content-free controls instead of real profiles.
_Avoid_: structural gain, regularization gain

**Semantic gain**:
Real profiles minus the best content-free control, paired by seed; the only part attributable to profile content.
_Avoid_: LLM gain, content gain

## ProMax components

**SDR** (supervised distribution reshaping):
ProMax's distillation term that pulls CF user–item similarities toward profile similarities; the only path by which
profile content reaches ProMax.

**S2DR** (self-supervised distribution reshaping; S²DR in the paper):
ProMax's contrastive term between CF embeddings on the training graph and on a second graph.

**LLM graph**:
The training graph plus the extra user–item edges an LLM added by re-ranking.
_Avoid_: RAG graph, augmented graph

**LightGCN+S2DR**:
The pure-CF baseline: LightGCN with S2DR on the training graph, no SDR and no profiles. ProMax minus this baseline is
everything ProMax gains from LLM signals.
_Avoid_: s2dr-only (that is only the run's variant name), ProMax without LLM

## Diagnostics (Tier 4)

**Content-only scorer**:
A scorer that ranks items from the profiles alone, with no CF embeddings: user–item profile cosine, item-to-item
profile kNN over the user's history, or a two-tower model trained on frozen profiles.
_Avoid_: LLM retriever, profile model

**Activity gate**:
A fusion of CF and a content-only scorer whose mixing weight depends only on the user's activity quintile, chosen on
validation data.
_Avoid_: adaptive fusion, smart gate

**Oracle gate**:
The same fusion with the best mixing weight picked per user on the test data; a ceiling, never a result.

**Sparsified user**:
A test user whose training history was cut to a fraction f of its interactions for the controlled-sparsity study.

**Profile leakage**:
A profile still encoding interactions that were removed from training, because the LLM wrote it from the full history.

**Leak-free user profile**:
A sparsified user's stand-in profile, built only from the profiles of the items the user kept.
_Avoid_: rebuilt profile, synthetic profile
