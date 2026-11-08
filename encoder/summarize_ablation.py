"""Summarize save/result/ablation.jsonl as mean ± std over seeds.

Usage: python summarize_ablation.py [--file save/result/ablation.jsonl] [--metric recall ndcg] [--k 5 10 20]
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
    parser.add_argument('--k', type=int, nargs='+', default=[5, 10, 20])
    args = parser.parse_args()

    groups = defaultdict(lambda: defaultdict(list))
    with open(args.file) as f:
        for line in f:
            r = json.loads(line)
            # the pure LightGCN baseline is shared by every LightGCN method
            model = 'lightgcn' if r['profile'] == 'none' else r['model']
            key = (r['dataset'], model, r['profile'], r['score'])
            for k in args.k:
                if k in r['k']:
                    for m in args.metric:
                        groups[key]['{}@{}'.format(m, k)].append(r[m][r['k'].index(k)])
            groups[key]['seeds'].append(r['seed'])

    def order(key):
        dataset, model, profile, score = key
        return (dataset, model != 'lightgcn', model, PROFILE_ORDER.index(profile), SCORE_ORDER.index(score))

    cols = ['{}@{}'.format(m, k) for k in args.k for m in args.metric]
    header = ['dataset', 'model', 'profile', 'score', 'n'] + cols
    rows = []
    for key in sorted(groups, key=order):
        vals = groups[key]
        rows.append(list(key) + [str(len(vals['seeds']))] +
                    ['{:.4f} ± {:.4f}'.format(np.mean(vals[c]), np.std(vals[c])) if vals[c] else '-' for c in cols])

    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for row in [header] + rows:
        print('  '.join(str(x).ljust(w) for x, w in zip(row, widths)))


if __name__ == '__main__':
    main()
