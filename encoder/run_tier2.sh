#!/usr/bin/env bash
# Tier 2: where does the complementarity come from?
# Trains every condition on the GPU with --save_topk, so each run also writes per-user top-K lists (model/cf/llm)
# and the final embeddings to save/tier2/<dataset>/. Metrics still go to save/result/ablation.jsonl, so these runs
# also give the Tier 1 'real' rows with all three scorers (summarize_ablation.py keeps the latest record per seed).
#
#   profile conditions: none, real, shuffled, random, plus real-user / real-item (--profile_side: the other side's
#   profiles are replaced by noise), which split the semantic gain into its user and item parts.
#
# Run from encoder/ in the conda env with CUDA torch. Every list can be overridden through the environment, e.g.
#   CUDA=4 DATASETS=amazon nohup bash run_tier2.sh > tier2_amazon.out 2>&1 &
#   CUDA=5 DATASETS=yelp SEEDS="2021 2022" SIDES="" nohup bash run_tier2.sh > tier2_yelp_a.out 2>&1 &
# Then: python analyze_tier2.py --dataset amazon   (writes save/result/tier2_amazon.md)

CUDA=${CUDA:-0}
DATASETS=${DATASETS:-"amazon yelp steam"}
SEEDS=${SEEDS:-"2021 2022 2023 2024 2025"}
MODELS=${MODELS:-"lightgcn_proex lightgcn_promax"}
PROFILES=${PROFILES:-"real shuffled random"}
SIDES=${SIDES-"user item"}          # set SIDES="" to skip the user-only / item-only runs
BASELINE=${BASELINE:-1}             # 0 skips the plain LightGCN run (only if its Tier 2 files already exist)
TOPK=${TOPK:-50}

cd "$(dirname "$0")"
FAILED=save/result/tier2_failed.txt

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "torch cannot see a GPU: activate the CUDA conda env first"; exit 1; }

run() {
    echo ">>> [$(date '+%F %T')] $*"
    python train_encoder.py --device cuda --cuda "$CUDA" --save_topk "$TOPK" "$@" || echo "$*" >> "$FAILED"
}

for dataset in $DATASETS; do
    for seed in $SEEDS; do
        if [ "$BASELINE" = 1 ]; then
            run --model lightgcn_proex --dataset "$dataset" --seed "$seed" --profile none
        fi
        for model in $MODELS; do
            for profile in $PROFILES; do
                run --model "$model" --dataset "$dataset" --seed "$seed" --profile "$profile" --score all
            done
            for side in $SIDES; do
                run --model "$model" --dataset "$dataset" --seed "$seed" --profile real --profile_side "$side" --score all
            done
        done
    done
    python analyze_tier2.py --dataset "$dataset" > /dev/null && echo "Tier 2 report: save/result/tier2_${dataset}.md"
done

[ -f "$FAILED" ] && echo "Some runs failed, see $FAILED"
python summarize_ablation.py
