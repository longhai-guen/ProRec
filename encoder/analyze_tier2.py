"""Tier 2: where does the complementarity come from?

Reads the per-user top-K lists and embeddings written by train_encoder.py --save_topk K
(save/tier2/<dataset>/<model>_<tag>_seed<seed>_{model,cf,llm,emb}.npz) and reports, mean ± std over seeds:

  A. Overall Recall/NDCG@K per condition and scorer (sanity check against ablation.jsonl)
  B. User-activity strata (quintiles of training interactions): where the total / semantic gain sits
  C. Item popularity (head = top 20% items by training interactions): head/tail recall, tail share of the lists
  D. Complementarity metrics, CF-only (C) vs profile-only (L) vs fused (F) lists, and F vs plain LightGCN (N)
  E. Geometry: linear CKA and kNN overlap between the CF space and the profile space

Every condition is shown next to the content-free controls (shuffled / random); "semantic" = real - best control.
Usage: python analyze_tier2.py --dataset amazon [--k 20] [--dir save/tier2] [--out save/result/tier2_amazon.md]
"""
import os
import re
import glob
import pickle
import argparse
from collections import defaultdict

import numpy as np

NAME_RE = re.compile(r'^(?P<model>.+?)_(?P<tag>(?:none|real|shuffled|random)(?:-user|-item)?)'
                     r'_seed(?P<seed>\d+)_(?P<kind>model|cf|llm|emb)\.npz$')
TAG_ORDER = ['none', 'real', 'real-user', 'real-item', 'shuffled', 'shuffled-user', 'shuffled-item', 'random']
CONTROLS = ['shuffled', 'random']
MODEL_LABEL = {'lightgcn_proex': 'ProEx', 'lightgcn_promax': 'ProMax'}


# ----------------------------------------------------------------------------- data

class Data:
    def __init__(self, dataset, data_root):
        def load(name):
            with open(os.path.join(data_root, dataset, name), 'rb') as f:
                return (pickle.load(f) != 0).tocsr()
        trn, tst = load('trn_mat.pkl'), load('tst_mat.pkl')
        self.n_users, self.n_items = trn.shape
        self.user_train = np.asarray(trn.sum(1)).ravel()
        self.item_pop = np.asarray(trn.sum(0)).ravel()
        self.test_len = np.asarray(tst.sum(1)).ravel()
        tst = tst.tocoo()
        self.test_keys = np.sort(tst.row.astype(np.int64) * self.n_items + tst.col)
        # head = top 20% items by training popularity
        order = np.argsort(-self.item_pop, kind='stable')
        self.is_tail = np.ones(self.n_items, dtype=bool)
        self.is_tail[order[:int(round(0.2 * self.n_items))]] = False
        self.tail_test_len = np.bincount(tst.row[self.is_tail[tst.col]], minlength=self.n_users)

    def hits(self, users, lists):
        keys = users[:, None].astype(np.int64) * self.n_items + lists
        pos = np.searchsorted(self.test_keys, keys)
        pos[pos == len(self.test_keys)] = 0
        return self.test_keys[pos] == keys


def activity_strata(data, users, n=5):
    """Equal-size groups of test users by training-interaction count (rank-based, so ties never empty a group)."""
    order = np.argsort(data.user_train[users], kind='stable')
    group = np.empty(len(users), dtype=np.int64)
    for g, idx in enumerate(np.array_split(order, n)):
        group[idx] = g
    labels = []
    for g in range(n):
        c = data.user_train[users[group == g]]
        labels.append('Q{} ({}-{})'.format(g + 1, int(c.min()), int(c.max())))
    return group, labels


# ----------------------------------------------------------------------------- per-user metrics

def recall_u(data, users, h, k):
    return h[:, :k].sum(1) / data.test_len[users]


def ndcg_u(data, users, h, k):
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (h[:, :k] * disc).sum(1)
    idcg = np.cumsum(disc)[np.minimum(data.test_len[users], k).astype(int) - 1]
    return dcg / idcg


def member(a, b):
    """member[u, j] = a[u, j] appears in b[u]."""
    return (a[:, :, None] == b[:, None, :]).any(2)


# ----------------------------------------------------------------------------- formatting

def ms(vals, fmt='{:.4f}', pct=False):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return '-'
    m, s = np.mean(vals), np.std(vals)
    if pct:
        return '{:.1f}% ± {:.1f}'.format(100 * m, 100 * s)
    return (fmt + ' ± ' + fmt).format(m, s) + ('' if len(vals) > 1 else ' (n=1)')


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    return '\n'.join(out)


def paired(a, b):
    """Per-seed differences a - b over shared seeds (dicts seed -> value)."""
    return [a[s] - b[s] for s in sorted(set(a) & set(b))]


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--dir', default='save/tier2')
    parser.add_argument('--data_root', default='../data')
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--knn', type=int, default=10)
    parser.add_argument('--sample', type=int, default=3000, help='entities sampled for CKA / kNN overlap')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    K = args.k
    out_path = args.out or 'save/result/tier2_{}.md'.format(args.dataset)

    data = Data(args.dataset, args.data_root)
    files = defaultdict(dict)   # (model, tag, seed) -> kind -> path
    for path in glob.glob(os.path.join(args.dir, args.dataset, '*.npz')):
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        model = 'lightgcn' if m['tag'] == 'none' else m['model']
        files[(model, m['tag'], int(m['seed']))][m['kind']] = path
    if not files:
        raise SystemExit('No Tier 2 files in {}/{}. Train with --save_topk 50 first.'.format(args.dir, args.dataset))

    # load lists, sorted by user id so every run is aligned
    users = None
    lists, hits = {}, {}
    for key, kinds in files.items():
        for kind, path in kinds.items():
            if kind == 'emb':
                continue
            z = np.load(path)
            order = np.argsort(z['users'])
            u, l = z['users'][order], z['topk'][order]
            if l.shape[1] < 2 * K:
                raise SystemExit('{} has top-{}; need at least {} (2K for the oracle comparison).'.format(path, l.shape[1], 2 * K))
            if users is None:
                users = u
            assert np.array_equal(users, u), 'test users differ in ' + path
            lists[key + (kind,)] = l
            hits[key + (kind,)] = data.hits(users, l)

    models = sorted({k[0] for k in files if k[0] != 'lightgcn'})
    tags_of = {m: [t for t in TAG_ORDER if any(k[0] == m and k[1] == t for k in files)] for m in models}
    seeds_of = lambda m, t: sorted(s for (mm, tt, s) in files if mm == m and tt == t)
    none_seeds = seeds_of('lightgcn', 'none')

    def per_seed(model, tag, kind, fn):
        """seed -> fn(list, hits) for one condition."""
        out = {}
        for s in (none_seeds if tag == 'none' else seeds_of(model, tag)):
            key = ('lightgcn' if tag == 'none' else model, tag, s, kind)
            if key in lists:
                out[s] = fn(lists[key], hits[key])
        return out

    rec = lambda l, h: recall_u(data, users, h, K).mean()
    best_ctrl = {}
    for m in models:
        cands = [t for t in CONTROLS if t in tags_of[m]]
        best_ctrl[m] = max(cands, key=lambda t: np.mean(list(per_seed(m, t, 'model', rec).values()))) if cands else None

    md = ['# Tier 2 analysis: {} (K = {})'.format(args.dataset, K), '',
          'Test users: {}. Seeds per condition are in section A. "semantic" = real - best content-free control '
          '(per seed, paired); best control per method: {}.'.format(
              len(users), ', '.join('{} {}'.format(MODEL_LABEL.get(m, m), best_ctrl[m]) for m in models)), '']

    # ------------------------------------------------------------------ A. overall
    md += ['## A. Overall Recall@{0} / NDCG@{0}'.format(K), '']
    rows = []
    nd = lambda l, h: ndcg_u(data, users, h, K).mean()
    rows.append(['LightGCN', 'none', 'model', len(none_seeds), ms(per_seed(None, 'none', 'model', rec).values()),
                 ms(per_seed(None, 'none', 'model', nd).values())])
    for m in models:
        for t in tags_of[m]:
            for kind in ['model', 'cf', 'llm']:
                r = per_seed(m, t, kind, rec)
                if r:
                    rows.append([MODEL_LABEL.get(m, m), t, kind, len(r), ms(r.values()), ms(per_seed(m, t, kind, nd).values())])
    md += [table(['method', 'profile', 'score', 'seeds', 'Recall@{}'.format(K), 'NDCG@{}'.format(K)], rows), '']

    # ------------------------------------------------------------------ B. user activity
    group, glabels = activity_strata(data, users)
    md += ['## B. User-activity strata (quintiles of training interactions), Recall@{}'.format(K), '',
           'Total = real - none; semantic = real - best control (5th column); user-/item-side = real-user / real-item - random '
           '(the other side gets noise). CF share = content-free gain / total gain.', '']

    def strat_rec(model, tag):
        def fn(l, h):
            r = recall_u(data, users, h, K)
            return np.array([r[group == g].mean() for g in range(len(glabels))])
        return per_seed(model, tag, 'model', fn)

    base = strat_rec(None, 'none')
    for m in models:
        real, ctrl = strat_rec(m, 'real'), strat_rec(m, best_ctrl[m]) if best_ctrl[m] else {}
        rand = strat_rec(m, 'random')
        sides = {t: strat_rec(m, t) for t in ['real-user', 'real-item'] if t in tags_of[m]}
        header = ['stratum', 'users', 'none', 'real', best_ctrl[m] or 'control', 'total gain', 'semantic gain', 'CF share']
        header += ['{} gain'.format(t.split('-')[1] + '-side') for t in sides]
        rows = []
        for g, lab in enumerate(glabels):
            col = lambda d: {s: v[g] for s, v in d.items()}
            b, r, c = col(base), col(real), col(ctrl)
            total = paired(r, b)
            cfree = paired(c, b)
            share = np.mean(cfree) / np.mean(total) if total and cfree and np.mean(total) > 0 else np.nan
            row = [lab, int((group == g).sum()), ms(b.values()), ms(r.values()), ms(c.values()),
                   ms(total), ms(paired(r, c)), '-' if np.isnan(share) else '{:.0f}%'.format(100 * share)]
            row += [ms(paired(col(d), col(rand))) for d in sides.values()]
            rows.append(row)
        md += ['### {}'.format(MODEL_LABEL.get(m, m)), '', table(header, rows), '']

    # ------------------------------------------------------------------ C. item popularity
    md += ['## C. Item popularity (head = top 20% items by training interactions)', '',
           'Tail/head recall = hits on tail/head test items / all tail/head test items (micro). '
           'Tail share = fraction of the top-{} slots filled with tail items.'.format(K), '']
    tail_test = data.tail_test_len[users].sum()
    head_test = data.test_len[users].sum() - tail_test

    def pop_stats(l, h):
        tail = data.is_tail[l[:, :K]]
        hk = h[:, :K]
        return np.array([(hk & tail).sum() / tail_test, (hk & ~tail).sum() / head_test, tail.mean()])

    rows = [['LightGCN', 'none'] + [ms([v[i] for v in per_seed(None, 'none', 'model', pop_stats).values()],
                                       pct=(i == 2)) for i in range(3)]]
    for m in models:
        for t in tags_of[m]:
            st = per_seed(m, t, 'model', pop_stats)
            rows.append([MODEL_LABEL.get(m, m), t] + [ms([v[i] for v in st.values()], pct=(i == 2)) for i in range(3)])
        if 'real' in tags_of[m] and best_ctrl[m]:
            r, c = per_seed(m, 'real', 'model', pop_stats), per_seed(m, best_ctrl[m], 'model', pop_stats)
            rows.append([MODEL_LABEL.get(m, m), '**semantic (real - {})**'.format(best_ctrl[m])] +
                        [ms(paired({s: v[i] for s, v in r.items()}, {s: v[i] for s, v in c.items()}), pct=(i == 2))
                         for i in range(3)])
    md += [table(['method', 'profile', 'tail recall@{}'.format(K), 'head recall@{}'.format(K), 'tail share of list'], rows), '']

    # ------------------------------------------------------------------ D. complementarity
    md += ['## D. Complementarity (C = CF-only, L = profile-only, F = fused, N = plain LightGCN, T = test items; top-{})'.format(K), '',
           'Disagreement and potential compare C with L; realized utility compares F with C (ProMax: F = C by design, so '
           'these are 0) and F with N from the same seed. Hit ratios are micro-averaged over users. Read every '
           'value against the shuffled/random columns: only the difference is due to profile content.', '']
    metric_names = [
        ('list overlap #(C∩L) / #(C∪L)', False),
        ('HitJaccard #(hC∩hL) / #(hC∪hL)', False),
        ('complementary hit ratio #(hL−hC) / #(hC∪hL)', False),
        ('CF-miss rescue: P(L hits ≥1, given C hits 0)', True),
        ('Recall C', False),
        ('Recall L', False),
        ('oracle-union recall #(hC∪hL) / #T (2K slots)', False),
        ('CF Recall@2K (same 2K budget)', False),
        ('oracle best-of-two recall (per user)', False),
        ('tail share of hL−hC', True),
        ('unexpectedness vs CF #(F−C) / K', False),
        ('serendipity vs CF #((F−C)∩T) / K', False),
        ('Recall F - Recall C', False),
        ('unexpectedness vs LightGCN #(F−N) / K', False),
        ('serendipity vs LightGCN #((F−N)∩T) / K', False),
        ('lost vs LightGCN #((N−F)∩T) / K', False),
        ('tail share of (F−N)∩T', True),
    ]

    def comp_stats(m, t, s):
        g = lambda kind: (lists.get((m, t, s, kind)), hits.get((m, t, s, kind)))
        (C2, hC2), (L, hL), (F, hF) = g('cf'), g('llm'), g('model')
        Nl, hN = lists.get(('lightgcn', 'none', s, 'model')), hits.get(('lightgcn', 'none', s, 'model'))
        v = [np.nan] * len(metric_names)
        T = data.test_len[users]
        if C2 is not None and L is not None:
            C, hC, L, hL = C2[:, :K], hC2[:, :K], L[:, :K], hL[:, :K]
            cInL, lInC = member(C, L), member(L, C)
            inter = cInL.sum(1)
            v[0] = (inter / (2 * K - inter)).mean()
            hit_inter = (hC & cInL).sum(1)
            nC, nL = hC.sum(1), hL.sum(1)
            hit_union = nC + nL - hit_inter
            l_only = hL & ~lInC
            v[1] = hit_inter.sum() / max(hit_union.sum(), 1)
            v[2] = l_only.sum() / max(hit_union.sum(), 1)
            miss = nC == 0
            v[3] = (nL[miss] > 0).mean() if miss.any() else np.nan
            v[4], v[5] = (nC / T).mean(), (nL / T).mean()
            v[6] = (hit_union / T).mean()
            v[7] = (hC2[:, :2 * K].sum(1) / T).mean()
            v[8] = np.maximum(nC / T, nL / T).mean()
            v[9] = data.is_tail[L[l_only]].mean() if l_only.any() else np.nan
        if C2 is not None and F is not None:
            C, hC, Fk, hFk = C2[:, :K], hC2[:, :K], F[:, :K], hF[:, :K]
            fNotC = ~member(Fk, C)
            v[10] = fNotC.mean()
            v[11] = (hFk & fNotC).sum(1).mean() / K
            v[12] = (hFk.sum(1) / T).mean() - (hC.sum(1) / T).mean()
        if F is not None and Nl is not None:
            Fk, hFk, Nk, hNk = F[:, :K], hF[:, :K], Nl[:, :K], hN[:, :K]
            fNotN, nNotF = ~member(Fk, Nk), ~member(Nk, Fk)
            v[13] = fNotN.mean()
            ser = hFk & fNotN
            v[14] = ser.sum(1).mean() / K
            v[15] = (hNk & nNotF).sum(1).mean() / K
            v[16] = data.is_tail[Fk[ser]].mean() if ser.any() else np.nan
        return v

    for m in models:
        tags = [t for t in tags_of[m] if t != 'none']
        stats = {t: {s: comp_stats(m, t, s) for s in seeds_of(m, t)} for t in tags}
        header = ['metric'] + tags + (['real - {}'.format(best_ctrl[m])] if 'real' in tags and best_ctrl[m] else [])
        rows = []
        for i, (name, pct) in enumerate(metric_names):
            row = [name] + [ms([v[i] for v in stats[t].values()], pct=pct) for t in tags]
            if len(header) > len(tags) + 1:
                r = {s: v[i] for s, v in stats['real'].items()}
                c = {s: v[i] for s, v in stats[best_ctrl[m]].items()}
                row.append(ms(paired(r, c), pct=pct))
            rows.append(row)
        md += ['### {}'.format(MODEL_LABEL.get(m, m)), '', table(header, rows), '']

    # ------------------------------------------------------------------ E. geometry
    md += ['## E. Geometry of the trained spaces', '',
           'Linear CKA (1 = same geometry up to rotation/scale) and kNN@{} overlap (share of an item\'s {} nearest '
           'neighbours, cosine, shared by both spaces) on {} sampled entities. "CF vs LightGCN" compares the method\'s '
           'CF item space with plain LightGCN\'s from the same seed: how much training with profiles reshaped it.'.format(
               args.knn, args.knn, args.sample), '']
    rng = np.random.RandomState(0)
    u_idx = rng.choice(data.n_users, min(args.sample, data.n_users), replace=False)
    i_idx = rng.choice(data.n_items, min(args.sample, data.n_items), replace=False)

    def cka(x, y):
        x = x.astype(np.float64) - x.mean(0)
        y = y.astype(np.float64) - y.mean(0)
        return np.linalg.norm(y.T @ x) ** 2 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y))

    def knn_overlap(a, b, q):
        def nn(e):
            e = e.astype(np.float64)
            e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            sims = e[q] @ e.T
            sims[np.arange(len(q)), q] = -np.inf
            return np.argpartition(-sims, args.knn, axis=1)[:, :args.knn]
        na, nb = nn(a), nn(b)
        return np.mean([len(np.intersect1d(x, y)) / args.knn for x, y in zip(na, nb)])

    rows = []
    for m in models:
        for t in [t for t in tags_of[m] if t != 'none']:
            vals = defaultdict(list)
            for s in seeds_of(m, t):
                p = files[(m, t, s)].get('emb')
                if not p:
                    continue
                e = np.load(p)
                if 'llm_item' in e:
                    vals['cka_u'].append(cka(e['cf_user'][u_idx], e['llm_user'][u_idx]))
                    vals['cka_i'].append(cka(e['cf_item'][i_idx], e['llm_item'][i_idx]))
                    vals['knn'].append(knn_overlap(e['cf_item'], e['llm_item'], i_idx))
                pn = files.get(('lightgcn', 'none', s), {}).get('emb')
                if pn:
                    en = np.load(pn)
                    vals['cka_n'].append(cka(e['cf_item'][i_idx], en['cf_item'][i_idx]))
                    vals['knn_n'].append(knn_overlap(e['cf_item'], en['cf_item'], i_idx))
            if vals:
                rows.append([MODEL_LABEL.get(m, m), t] + [ms(vals[c], '{:.3f}') for c in ['cka_u', 'cka_i', 'knn', 'cka_n', 'knn_n']])
    if rows:
        md += [table(['method', 'profile', 'CKA users CF-vs-profile', 'CKA items CF-vs-profile',
                      'kNN@{} items CF-vs-profile'.format(args.knn), 'CKA items CF vs LightGCN',
                      'kNN@{} items CF vs LightGCN'.format(args.knn)], rows), '']
    else:
        md += ['No *_emb.npz files found.', '']

    text = '\n'.join(md)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(text + '\n')
    print(text)
    print('\nSaved to', out_path)


if __name__ == '__main__':
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        main()
