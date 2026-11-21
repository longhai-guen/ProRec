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

**SDR**:
ProMax's distillation term that pulls CF user–item similarities toward profile similarities; the only path by which
profile content reaches ProMax.

**S2DR**:
ProMax's contrastive term between CF embeddings on the training graph and on a second graph.

**LLM graph**:
The training graph plus the extra user–item edges an LLM added by re-ranking.
_Avoid_: RAG graph, augmented graph

**LightGCN+S2DR**:
The pure-CF baseline: LightGCN with S2DR on the training graph, no SDR and no profiles. ProMax minus this baseline is
everything ProMax gains from LLM signals.
_Avoid_: s2dr-only (that is only the run's variant name), ProMax without LLM
