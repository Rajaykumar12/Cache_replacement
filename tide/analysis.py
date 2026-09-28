"""Statistics over bench_results_{synth,real}.csv, shared by bench.py and report/make_report.py.

A cell is (trace, instance, cache size). Baselines run once per cell; TIDE runs over several network seeds.
Aggregates are means over (trace, size) of the mean over instances. Confidence intervals are 95%:
  * per (trace, size): Student t over instances x seeds (TIDE) or instances (baselines);
  * aggregates and paired differences: a hierarchical bootstrap that resamples generator instances within each
    (trace, size) and TIDE seeds within each instance. They measure run-to-run noise for this workload mix,
    not uncertainty about which workloads matter.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
LEARNED = ['LIRS', 'LeCaR', 'Cacheus', 'LRB']
HEAD = ['LRU', 'LFU', 'ARC', 'S3-FIFO', 'SIEVE', *LEARNED, 'TIDE']
MIN_GAP = 0.5  # normalise only where Belady beats LRU by more than this many points


def load(which):
    return pd.read_csv(os.path.join(ROOT, f'bench_results_{which}.csv'))


def cells(df):
    """One row per (trace, inst, c): hit rate of every policy (TIDE = seed mean) plus TIDE's seed list."""
    df = df.drop_duplicates(['trace', 'inst', 'c', 'policy', 'seed'])  # tiny traces round sizes to the same c
    key = ['trace', 'inst', 'c']
    h = df.groupby(key + ['policy'])['hit'].mean().unstack('policy')
    h['frac'] = df.groupby(key)['frac'].min()
    h['tide_seeds'] = df[df.policy == 'TIDE'].groupby(key)['hit'].apply(list)
    return h.reset_index()


def tci(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.allclose(x, x[0]): return 0.0
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def per_size(h):
    """(trace, frac) table: mean and t-CI half-width per policy."""
    rows = []
    for (tr, f), g in h.groupby(['trace', 'frac'], sort=False):
        r = dict(trace=tr, frac=f, c=int(g.c.iloc[0]), n_inst=len(g))
        for p in HEAD + ['TIDE-oracle', 'Belady']:
            vals = sum(g.tide_seeds, []) if p == 'TIDE' else g[p].tolist()
            r[p], r[p + '_ci'] = float(np.mean(vals)), tci(vals)
        d = [np.mean(s) - a for s, a in zip(g.tide_seeds, g.ARC)]
        r['dARC'], r['dARC_ci'] = float(np.mean(d)), tci(d)
        rows.append(r)
    return pd.DataFrame(rows)


def _agg(h, tide_col='TIDE'):
    """Mean normalised score per policy, and mean raw hit rate, over (trace, frac) groups."""
    ok = h[h.Belady - h.LRU > MIN_GAP]
    norm = {p: ((ok[p if p != 'TIDE' else tide_col] - ok.LRU) / (ok.Belady - ok.LRU)).groupby([ok.trace, ok.frac]).mean().mean()
            for p in HEAD}
    raw = {p: h[p if p != 'TIDE' else tide_col].groupby([h.trace, h.frac]).mean().mean() for p in HEAD}
    return norm, raw


def bootstrap(h, B=4000, seed=0):
    """Resample instances within (trace, frac) and TIDE seeds within instance; returns draws of (norm, raw)."""
    rng = np.random.default_rng(seed)
    groups = [g.reset_index(drop=True) for _, g in h.groupby(['trace', 'frac'], sort=False)]
    draws = []
    for _ in range(B):
        parts = []
        for g in groups:
            s = g.iloc[rng.integers(0, len(g), len(g))].copy()
            s['TIDE_b'] = [np.mean(rng.choice(v, len(v))) for v in s.tide_seeds]
            parts.append(s)
        draws.append(_agg(pd.concat(parts), 'TIDE_b'))
    return draws


def ranking(h, B=4000):
    """Policy table with normalised score, mean hit rate, wins, and bootstrap CIs; plus TIDE - X paired diffs."""
    norm, raw = _agg(h)
    draws = bootstrap(h, B)
    q = lambda xs: (np.percentile(xs, 2.5), np.percentile(xs, 97.5))
    wins = (h[HEAD].values >= h[HEAD].max(axis=1).values[:, None] - 1e-9).sum(0)  # ties: each tied policy wins
    wins = dict(zip(HEAD, wins))
    rows = []
    for p in HEAD:
        nl, nh = q([d[0][p] for d in draws])
        dl, dh = q([d[1]['TIDE'] - d[1][p] for d in draws])
        rows.append(dict(policy=p, norm=norm[p], norm_lo=nl, norm_hi=nh, hit=raw[p], wins=int(wins[p]),
                         tide_minus=raw['TIDE'] - raw[p], tide_minus_lo=dl, tide_minus_hi=dh))
    return pd.DataFrame(rows).sort_values('norm', ascending=False).reset_index(drop=True)


def summarize(which, B=4000):
    h = cells(load(which))
    s = per_size(h)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
    cols = ['trace', 'frac', 'c', 'n_inst'] + HEAD + ['TIDE-oracle', 'Belady', 'dARC', 'dARC_ci']
    print(f'\n[{which}] hit rate (%) per (trace, size); TIDE = mean over instances x seeds')
    print(s[cols].round(2).to_string(index=False))
    r = ranking(h, B)
    print(f'\n[{which}] ranking over {len(s)} (trace, size) groups, {len(h)} cells; 95% bootstrap CIs')
    print(r.round(3).to_string(index=False))
    w = s.loc[s.dARC.idxmin()]
    print(f'worst TIDE - ARC: {w.dARC:+.2f} +/- {w.dARC_ci:.2f} ({w.trace}, c={w.c}); '
          f'TIDE >= ARC in {(s.dARC >= 0).sum()}/{len(s)} groups')
    return h, s, r


if __name__ == '__main__':
    import sys
    for w in sys.argv[1:] or ['synth', 'real']:
        if os.path.exists(os.path.join(ROOT, f'bench_results_{w}.csv')): summarize(w)
