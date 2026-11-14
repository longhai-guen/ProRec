#!/usr/bin/env bash
# Tier 3: which component of each method carries the gain?
# Every component variant is trained with real AND random profiles, so each one splits into a content-free part
# (random - LightGCN) and a semantic part (real - random). The 'full' rows come from the Tier 1/2 runs.
#   ProEx:  p1 (original profile only), noinv (no variance loss), noex (no uniformity loss);
#           every ProEx run also scores 'orig' (original profile only at test time, no mean pooling)
#   ProMax: nosdr (S2DR only), nos2dr (SDR only), noentropy (w = 1), nowhiten (raw profiles for SDR),
#           graph-train / graph-shuffled / graph-random (LLM-added edges removed / moved to other users / random).
#           ProMax random@graph-shuffled is the fully content-free ProMax (no LLM signal in profiles or graph).
#
# Run from encoder/ in the conda env with CUDA torch. Lists can be overridden through the environment, e.g.
#   CUDA=4 DATASETS=amazon nohup bash run_tier3.sh > tier3_amazon.out 2>&1 &
#   CUDA=5 DATASETS=yelp MODELS=lightgcn_promax nohup bash run_tier3.sh > tier3_yelp_promax.out 2>&1 &
# Then: python summarize_tier3.py   (writes save/result/tier3_summary.md)

CUDA=${CUDA:-0}
DATASETS=${DATASETS:-"amazon yelp steam"}
SEEDS=${SEEDS:-"2021 2022 2023"}
MODELS=${MODELS:-"lightgcn_proex lightgcn_promax"}
PROFILES=${PROFILES:-"real random"}
PROEX_VARIANTS=${PROEX_VARIANTS:-"p1 noinv noex"}
PROMAX_VARIANTS=${PROMAX_VARIANTS:-"graph-shuffled graph-random graph-train nosdr nos2dr noentropy nowhiten"}
TOPK=${TOPK:-0}                     # e.g. TOPK=50 to also save Tier 2 style top-K lists

cd "$(dirname "$0")"
FAILED=save/result/tier3_failed.txt

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "torch cannot see a GPU: activate the CUDA conda env first"; exit 1; }

run() {
    echo ">>> [$(date '+%F %T')] $*"
    python train_encoder.py --device cuda --cuda "$CUDA" --save_topk "$TOPK" "$@" || echo "$*" >> "$FAILED"
}

for dataset in $DATASETS; do
    for seed in $SEEDS; do
        for model in $MODELS; do
            if [ "$model" = lightgcn_proex ]; then variants=$PROEX_VARIANTS; else variants=$PROMAX_VARIANTS; fi
            for variant in $variants; do
                for profile in $PROFILES; do
                    run --model "$model" --dataset "$dataset" --seed "$seed" --profile "$profile" --variant "$variant" --score all
                done
            done
        done
    done
done

[ -f "$FAILED" ] && echo "Some runs failed, see $FAILED"
python summarize_tier3.py
