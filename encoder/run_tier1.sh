#!/usr/bin/env bash
# Tier 1 ablation: is the LLM profile signal complementary to CF?
#   profile: none (pure LightGCN) | real | shuffled | random
#   score:   model | cf | llm  (all three evaluated after a single training run via --score all)
# Results are appended to save/result/ablation.jsonl; summarize with: python summarize_ablation.py
#
# Run from encoder/. Every list can be overridden through the environment, e.g.
#   CUDA=1 DATASETS="amazon" SEEDS="2025" bash run_tier1.sh

CUDA=${CUDA:-5}
DEVICE=${DEVICE:-cuda}
DATASETS=${DATASETS:-"amazon yelp steam"}
SEEDS=${SEEDS:-"2021 2022 2023 2024 2025"}
MODELS=${MODELS:-"lightgcn_proex lightgcn_promax"}

#exclude real mode because it has been ran
PROFILES=${PROFILES:-"shuffled random"}

cd "$(dirname "$0")"
FAILED=save/result/tier1_failed.txt

run() {
    echo ">>> $*"
    python train_encoder.py --device "$DEVICE" --cuda "$CUDA" "$@" || echo "$*" >> "$FAILED"
}

for dataset in $DATASETS; do
    for seed in $SEEDS; do
        # Pure LightGCN baseline: both LightGCN configs share every hyperparameter, so run it once
        run --model lightgcn_proex --dataset "$dataset" --seed "$seed" --profile none

        for model in $MODELS; do
            for profile in $PROFILES; do
                run --model "$model" --dataset "$dataset" --seed "$seed" --profile "$profile" --score all
            done
        done
    done
done

[ -f "$FAILED" ] && echo "Some runs failed, see $FAILED"
python summarize_ablation.py
