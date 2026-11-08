"""Summarize save/result/ablation.jsonl as mean ± std over seeds.

Usage: python summarize_ablation.py [--file save/result/ablation.jsonl] [--metric recall ndcg] [--k 20]
"""
import json
import argparse
from collections import defaultdict

import numpy as np

PROFILE_ORDER = ['none', 'real', 'shuffled', 'random']
SCORE_ORDER = ['model', 'cf', 'llm']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='save/result/ablation.jsonl')
    parser.add_argument('--metric', nargs='+', default=['recall', 'ndcg'])
    parser.add_argument('--k', type=int, default=20)
    args = parser.parse_args()

    groups = defaultdict(lambda: defaultdict(list))
    with open(args.file) as f:
        for line in f:
            r = json.loads(line)
            # the pure LightGCN baseline is shared by every LightGCN method
            model = 'lightgcn' if r['profile'] == 'none' else r['model']
            key = (r['dataset'], model, r['profile'], r['score'])
            k_idx = r['k'].index(args.k)
            for m in args.metric:
                groups[key][m].append(r[m][k_idx])
            groups[key]['seeds'].append(r['seed'])

    def order(key):
        dataset, model, profile, score = key
        return (dataset, model != 'lightgcn', model, PROFILE_ORDER.index(profile), SCORE_ORDER.index(score))

    header = ['dataset', 'model', 'profile', 'score', 'n'] + ['{}@{}'.format(m, args.k) for m in args.metric]
    rows = []
    for key in sorted(groups, key=order):
        vals = groups[key]
        rows.append(list(key) + [str(len(vals['seeds']))] +
                    ['{:.4f} ± {:.4f}'.format(np.mean(vals[m]), np.std(vals[m])) for m in args.metric])

    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for row in [header] + rows:
        print('  '.join(str(x).ljust(w) for x, w in zip(row, widths)))


if __name__ == '__main__':
    main()
