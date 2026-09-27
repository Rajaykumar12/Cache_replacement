"""Trace loading and trend-workload generators. Every trace is a list of int keys."""
import csv
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_csv(path):
    """Reads item_id only: the CSVs also carry time_to_next_request etc., which would leak the future."""
    with open(path) as f:
        return [int(r['item_id']) for r in csv.DictReader(f)]


def _zipf(rng, n, n_items, alpha):
    w = 1.0 / np.arange(1, n_items + 1) ** alpha
    return rng.choice(n_items, size=n, p=w / w.sum())


def lifecycle(n=1_000_000, seed=0):
    """Items are born at random times; each one's requests rise and decay (gamma(2, tau) after birth)."""
    rng = np.random.default_rng(seed)
    cnt = np.zeros(0, np.int64)
    while cnt.sum() < 1.2 * n:
        cnt = np.concatenate([cnt, np.minimum(3 * (rng.pareto(1.2, n // 20) + 1), 5000).astype(np.int64)])
    m = len(cnt)
    birth = rng.uniform(0, n, m)
    tau = np.exp(rng.uniform(np.log(300), np.log(30_000), m))
    times = np.repeat(birth, cnt) + rng.gamma(2.0, np.repeat(tau, cnt))
    items = np.repeat(np.arange(m), cnt)
    return items[np.argsort(times, kind='stable')][:n].tolist()


def cycle(n=1_000_000, seed=0, groups=40, size=100, noise=0.3):
    """Groups of keys recur in a fixed cyclic order (one shuffled pass per visit) over a zipf background."""
    rng = np.random.default_rng(seed)
    bg = rng.random(n) < noise
    nl = n - int(bg.sum())
    visits = -(-nl // size)
    loop = ((np.arange(visits) % groups)[:, None] * size + np.argsort(rng.random((visits, size)), axis=1)).ravel()[:nl]
    out = np.empty(n, np.int64)
    out[~bg] = loop
    out[bg] = groups * size + _zipf(rng, int(bg.sum()), 100_000, 1.2)
    return out.tolist()


def phases(n=1_000_000, seed=0, rounds=2):
    """Each round: zipf A -> one-time scan -> loop -> zipf B (permuted ranks). Rounds repeat the same A/B/loop keys."""
    rng = np.random.default_rng(seed)
    u = n // (16 * rounds)  # 16 units per round
    perm = rng.permutation(100_000)
    parts, scan_id = [], 1_000_000
    for _ in range(rounds):
        parts.append(('zA', _zipf(rng, 6 * u, 100_000, 1.0)))
        parts.append(('scan', np.arange(scan_id, scan_id + 2 * u))); scan_id += 2 * u
        parts.append(('loop', 4 * u))
        parts.append(('zB', perm[_zipf(rng, 4 * u, 100_000, 1.0)]))
    # Loop is sized to ~1.3x the 1%-of-footprint cache, so it thrashes LRU/ARC there but is learnable.
    f0 = len(set(np.concatenate([p for k, p in parts if k != 'loop']).tolist()))
    L = max(16, round(0.013 * f0 / 0.987))
    loop = 2_000_000 + np.arange(L)
    out = [np.resize(loop, p) if k == 'loop' else p for k, p in parts]
    return np.concatenate(out)[:n].tolist()


_CSV = [os.path.join(ROOT, 'data', f) for f in ('zipf_100k.csv', 'web_traffic_100k.csv')]
_CSV += sorted(glob.glob(os.path.join(ROOT, 'archive', 'data_gen', '*.csv')))
CSV = {os.path.basename(p)[:-4]: (lambda p=p: load_csv(p)) for p in _CSV}
TREND = {'lifecycle': lifecycle, 'cycle': cycle, 'phases': phases}
SUITE = {**CSV, **TREND}
