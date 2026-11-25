#!/usr/bin/env bash
# Tier 4, diagnostic 1: is ProMax's SDR switched off, and does a well-scaled SDR help on top of LightGCN+S2DR?
# Plan: docs/tier4-plan.md. Run one stage at a time; after each, `python summarize_tier4.py` reports it and writes the
# selected settings to save/result/tier4_best.json, which the next stage reads.
#   STAGE=1a  ProMax full, real profiles, seed 2025, with per-epoch SDR weight statistics in the log   (3 runs)
#   STAGE=1b  graph-train (= LightGCN+S2DR + SDR) x sdr_tau {0.2,0.05,0.02} x whiten {pca_white,raw}
#             x sdr_weight {0.05,0.5,1.5}, seed 2025, plus s2dr-only at seed 2025 for a baseline validation score
#             (57 runs). sdr_weight is in this grid because a sharper target raises both w and the SDR loss, so
#             at a fixed weight 1b would always pick the weakest target.
#   STAGE=1c  best (tau, whiten) from 1b x sdr_weight {0.01,0.05,0.1,0.5,1.5,3.0} x sdr_w {entropy,one},
#             seed 2025                                                                               (36 runs)
#   STAGE=1d  best setting from 1c x profile {real,shuffled,random} x seeds 2021-2025, top-50 lists saved (45 runs)
# Selection uses validation Recall@20 only (seed 2025); test numbers of 1b/1c are never used to choose.
#
# Run from encoder/ in the conda env with CUDA torch, e.g.
#   STAGE=1a CUDA=4 nohup bash run_tier4.sh > tier4_1a.out 2>&1 &
#   STAGE=1b CUDA=4 DATASETS=amazon nohup bash run_tier4.sh > tier4_1b_amazon.out 2>&1 &

STAGE=${STAGE:?set STAGE to 1a, 1b, 1c or 1d}
CUDA=${CUDA:-0}
DATASETS=${DATASETS:-"amazon yelp steam"}
SEEDS=${SEEDS:-"2021 2022 2023 2024 2025"}      # 1d only
TAUS=${TAUS:-"0.2 0.05 0.02"}
WHITENS=${WHITENS:-"pca_white raw"}
SDR_WEIGHTS_1B=${SDR_WEIGHTS_1B:-"0.05 0.5 1.5"}
SDR_WEIGHTS=${SDR_WEIGHTS:-"0.01 0.05 0.1 0.5 1.5 3.0"}   # 1c
SDR_WS=${SDR_WS:-"entropy one"}
TOPK=${TOPK:-50}                                 # 1d only

cd "$(dirname "$0")"
FAILED=save/result/tier4_failed.txt
BEST=save/result/tier4_best.json

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "torch cannot see a GPU: activate the CUDA conda env first"; exit 1; }

run() {
    echo ">>> [$(date '+%F %T')] $*"
    python train_encoder.py --device cuda --cuda "$CUDA" "$@" || echo "$*" >> "$FAILED"
}

# best flags selected by summarize_tier4.py for a stage and dataset, e.g. "--sdr_tau 0.05 --whiten raw"
best() {
    python - "$BEST" "$1" "$2" <<'EOF'
import json, sys
path, stage, dataset = sys.argv[1:]
try:
    flags = json.load(open(path))[stage][dataset]
except (OSError, KeyError):
    sys.exit('no selection for stage {} / {} in {}: run summarize_tier4.py first'.format(stage, dataset, path))
print(' '.join('--{} {}'.format(k, v) for k, v in flags.items()))
EOF
}

P="--model lightgcn_promax --log_sdr"
for dataset in $DATASETS; do
    case $STAGE in
    1a)
        run $P --dataset "$dataset" --seed 2025 --profile real ;;
    1b)
        run $P --dataset "$dataset" --seed 2025 --profile real --variant s2dr-only
        for tau in $TAUS; do for whiten in $WHITENS; do for sw in $SDR_WEIGHTS_1B; do
            run $P --dataset "$dataset" --seed 2025 --profile real --variant graph-train --sdr_tau "$tau" --whiten "$whiten" --sdr_weight "$sw"
        done; done; done ;;
    1c)
        flags=$(best 1b "$dataset") || exit 1
        for sw in $SDR_WEIGHTS; do for w in $SDR_WS; do
            run $P --dataset "$dataset" --seed 2025 --profile real --variant graph-train $flags --sdr_weight "$sw" --sdr_w "$w"
        done; done ;;
    1d)
        flags=$(best 1c "$dataset") || exit 1
        for seed in $SEEDS; do for profile in real shuffled random; do
            run $P --dataset "$dataset" --seed "$seed" --profile "$profile" --variant graph-train $flags --save_topk "$TOPK" --score all
        done; done ;;
    *)
        echo "unknown STAGE $STAGE"; exit 1 ;;
    esac
done

[ -f "$FAILED" ] && echo "Some runs failed, see $FAILED"
python summarize_tier4.py
