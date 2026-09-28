"""Trace generators and loaders. Every trace is a list of int keys.

Synthetic generators take a seed, so each one yields independent instances (bench.py --instances). They are
seeded ports of the project's original generators (../workload_generator.py, ../data_gen/*.py) with the same
parameters. Real traces come from realtraces.py.
"""
import csv

import numpy as np

import realtraces

N_SMALL, ITEMS = 100_000, 10_000  # original workload_generator.py settings


def load_csv(path):
    """Reads item_id only: the CSVs also carry time_to_next_request etc., which would leak the future."""
    with open(path) as f:
        return [int(r['item_id']) for r in csv.DictReader(f)]


def _zipf(rng, n, n_items, alpha):
    w = 1.0 / np.arange(1, n_items + 1) ** alpha
    return rng.choice(n_items, size=n, p=w / w.sum())


# ---------- small synthetic workloads (100k requests) ----------
def zipf_alpha(alpha, seed=0, n=N_SMALL, n_items=ITEMS):
    rng = np.random.default_rng(seed)
    if alpha > 1: return ((rng.zipf(alpha, n) - 1) % n_items).tolist()
    return _zipf(rng, n, n_items, alpha).tolist()


def zipf_unbounded(seed=0, n=N_SMALL, alpha=1.2):
    """data_gen/generate_zipf_data.py: raw numpy zipf ids, long tail left unbounded."""
    return np.random.default_rng(seed).zipf(alpha, n).tolist()


def web_traffic(seed=0, n=N_SMALL, n_items=5000):
    """data_gen/generate_web_traffic.py: 70% of requests to a 5% hot set; 10% of it replaced every 2000 requests."""
    rng = np.random.default_rng(seed)
    hot = rng.choice(np.arange(1, n_items + 1), n_items // 20, replace=False)
    k = len(hot) // 10
    out = np.empty(n, np.int64)
    for s in range(0, n, 2000):
        if s: hot[rng.choice(len(hot), k, replace=False)] = rng.choice(np.arange(1, n_items + 1), k, replace=False)
        m = min(2000, n - s)
        is_hot = rng.random(m) < 0.7
        out[s:s + m] = np.where(is_hot, rng.choice(hot, m), rng.integers(1, n_items + 1, m))
    return out.tolist()


def uniform(seed=0, n=N_SMALL, n_items=ITEMS):
    return np.random.default_rng(seed).integers(0, n_items, n).tolist()


def gaussian(seed=0, n=N_SMALL, n_items=ITEMS):
    s = np.random.default_rng(seed).normal(n_items / 2, n_items / 6, n)
    return np.clip(s, 0, n_items - 1).astype(int).tolist()


def bursty(seed=0, n=N_SMALL, n_items=ITEMS):
    """Uniform background; with 10% chance per step, a burst of 10-99 repeats of one key."""
    rng, s = np.random.default_rng(seed), []
    while len(s) < n:
        if rng.random() < 0.1: s.extend([int(rng.integers(n_items))] * int(rng.integers(10, 100)))
        else: s.append(int(rng.integers(n_items)))
    return s[:n]


def periodic(seed=0, n=N_SMALL, n_items=ITEMS, period=5000):
    """80% of requests go to a 100-key hot window that shifts by 100 keys every `period` requests."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    shift = (t // period) * 100 % n_items
    hot = (shift + rng.integers(0, 100, n)) % n_items
    return np.where(rng.random(n) < 0.8, hot, rng.integers(0, n_items, n)).tolist()


def adversarial(seed=0, n=N_SMALL, loop=31):
    """Cyclic scan over 31 keys (one more than the original 30-slot target cache). Deterministic up to a key
    relabelling, so instances differ only in labels; kept for completeness."""
    perm = np.random.default_rng(seed).permutation(loop) if seed else np.arange(loop)
    return np.resize(perm, n).tolist()


# ---------- trend workloads (1M requests) ----------
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


SMALL = {
    'zipf_100k': zipf_unbounded, 'web_traffic_100k': web_traffic,
    **{f'zipf_alpha_{a}': (lambda seed=0, a=a: zipf_alpha(a, seed)) for a in (0.5, 0.8, 1.0, 1.2, 1.5)},
    'uniform': uniform, 'gaussian': gaussian, 'bursty': bursty, 'periodic': periodic, 'adversarial': adversarial,
}
TREND = {'lifecycle': lifecycle, 'cycle': cycle, 'phases': phases}
REAL = realtraces.REAL
SUITE = [*TREND, *SMALL, *REAL]


def make(name, inst=0, n_trend=1_000_000, n_real=realtraces.N):
    """Trace `name`, instance `inst` (the generator seed). Real traces have a single instance (first n_real requests)."""
    if name in TREND: return TREND[name](n_trend, seed=inst)
    if name in SMALL: return SMALL[name](seed=inst)
    if name in REAL:
        assert inst == 0, 'real traces have one instance'
        return realtraces.load(name, n_real)
    raise KeyError(name)
