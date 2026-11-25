# Diagnose the profile signal before designing an LLM method

Status: accepted (2026-11-25)

Tiers 1–3 showed where ProEx's and ProMax's gains come from (content-free regularization), but not whether the
profiles carry more signal than these methods extract. Before designing a new method, we run three Tier 4
diagnostics against the LightGCN+S2DR baseline, on all three datasets with yelp as a negative control: (1) SDR with a
sharper target and weight, (2) an upper bound on the content-only signal with an activity gate vs an oracle gate, and
(3) a controlled-sparsity study. A gated LLM method is designed only if one of these holds: well-scaled SDR gives a
semantic gain ≥ +0.002 Recall@20 (p < 0.05) on at least 2 datasets; the activity gate adds ≥ +0.003 Recall@20 for
Q1 users; or the leak-free semantic gain grows as f shrinks. Otherwise the evaluation study is the main contribution.

## Considered options

- **Go straight to method design.** Rejected: the largest semantic gain so far (+0.0038) is too small to motivate a
  method without evidence that more signal is available.
- **Regenerate user profiles with an LLM for the sparsity study.** Rejected for now (cost and time). Instead each
  condition runs twice: with the real (leaky) user profile as a ceiling, and with a leak-free user profile. A gain
  that appears only with leaky profiles is attributed to profile leakage, not to the LLM.

## Consequences

- The thresholds are fixed before the runs, so they cannot be tuned to the outcome.
- All SDR settings are selected on validation data with one seed; only the chosen setting gets 5 seeds on test.
