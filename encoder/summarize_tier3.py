"""Tier 3: which component carries the gain? Reads save/result/ablation.jsonl (latest record per seed wins).

For every method and component variant (full = the original method, from the Tier 1/2 runs), with the `model` scorer:
  real, random        metric with real / random (content-free) profiles
  content-free gain   random - LightGCN        (what the component gives without profile content)
  semantic gain       real - random            (what profile content adds on top)
  real vs full        real(variant) - real(full)
ProEx also gets `orig - model` on real profiles: fused score with the original profile only minus mean pooling.
Differences are paired by seed (mean ± std over the shared seeds).

Usage: python summarize_tier3.py [--metric recall] [--k 20] [--out save/result/tier3_summary.md]
"""
import json
import argparse
from collections import defaultdict

import numpy as np

MODEL_LABEL = {'lightgcn_proex': 'ProEx', 'lightgcn_promax': 'ProMax'}
VARIANT_ORDER = ['full', 'p1', 'noinv', 'noex', 'nosdr', 'nos2dr', 'noentropy', 'nowhiten',
                 'graph-train', 'graph-shuffled', 'graph-random']


def ms(vals, signed=True):
    if not vals:
        return '-'
    fmt = '{:+.4f} ± {:.4f}' if signed else '{:.4f} ± {:.4f}'
    return fmt.format(np.mean(vals), np.std(vals)) + ' (n={})'.format(len(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='save/result/ablation.jsonl')
    parser.add_argument('--metric', default='recall')
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--out', default='save/result/tier3_summary.md')
    args = parser.parse_args()

    # value[(dataset, model, profile, variant, score)][seed]
    value = defaultdict(dict)
    with open(args.file) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('side', 'both') != 'both' or args.k not in r['k']:
                continue
            model = 'lightgcn' if r['profile'] == 'none' else r['model']
            key = (r['dataset'], model, r['profile'], r.get('variant', 'full'), r['score'])
            value[key][r['seed']] = r[args.metric][r['k'].index(args.k)]

    def get(*key):
        return value.get(key, {})

    def diff(a, b):
        return [a[s] - b[s] for s in sorted(set(a) & set(b))]

    datasets = sorted({k[0] for k in value})
    md = ['# Tier 3: component ablations ({}@{}, `model` scorer)'.format(args.metric, args.k), '',
          'content-free gain = random - LightGCN; semantic gain = real - random; real vs full = real(variant) - '
          'real(full). All paired by seed.', '']
    for ds in datasets:
        none = get(ds, 'lightgcn', 'none', 'full', 'model')
        md += ['## {}'.format(ds), '', 'LightGCN: {}'.format(ms(list(none.values()), False)), '']
        for model in ['lightgcn_proex', 'lightgcn_promax']:
            variants = [v for v in VARIANT_ORDER if get(ds, model, 'real', v, 'model') or get(ds, model, 'random', v, 'model')]
            if not variants:
                continue
            full_real = get(ds, model, 'real', 'full', 'model')
            header = ['variant', 'real', 'random', 'content-free gain', 'semantic gain', 'real vs full']
            if model == 'lightgcn_proex':
                header.append('orig - model (real)')
            rows = []
            for v in variants:
                real, rand = get(ds, model, 'real', v, 'model'), get(ds, model, 'random', v, 'model')
                row = [v, ms(list(real.values()), False), ms(list(rand.values()), False), ms(diff(rand, none)), ms(diff(real, rand)),
                       '-' if v == 'full' else ms(diff(real, full_real))]
                if model == 'lightgcn_proex':
                    row.append(ms(diff(get(ds, model, 'real', v, 'orig'), real)))
                rows.append(row)
            md += ['### {}'.format(MODEL_LABEL[model]), '',
                   '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
            md += ['| ' + ' | '.join(r) + ' |' for r in rows]
            md += ['']
    text = '\n'.join(md)
    with open(args.out, 'w') as f:
        f.write(text + '\n')
    print(text)
    print('\nSaved to', args.out)


if __name__ == '__main__':
    main()
