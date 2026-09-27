"""Assert-based checks for TIDE. Run: python3 test_tide.py"""
import random

import numpy as np

from policies import ARC, belady, next_use
from tide import Net, Tide, TideOracle
from traces import SUITE


def mixed_trace(n=30_000, seed=1):
    r = random.Random(seed); out = []
    while len(out) < n:
        mode = r.random()
        if mode < 0.5: out += [int(r.paretovariate(1.1)) for _ in range(500)]         # skewed
        elif mode < 0.75: out += list(range(10**6 + len(out), 10**6 + len(out) + 200))  # scan
        else: out += list(range(5000, 5000 + r.randint(20, 80))) * 5                  # loop
    return out[:n]


def hits(policy, trace):
    return bytes(policy.access(x) for x in trace)


def test_invariants():
    tr = mixed_trace()
    for c in (16, 50):
        t = Tide(c, force=True, W=4 * c)  # random-ish untrained scores exercise override, MRU and bypass paths
        for x in tr:
            t.access(x)
            n1, n2, b1, b2 = len(t.T1), len(t.T2), len(t.B1), len(t.B2)
            assert n1 + n2 <= c and n1 + b1 <= c and n1 + n2 + b1 + b2 <= 2 * c and 0 <= t.p <= c
            assert n1 + n2 + b1 + b2 < c or n1 + n2 == c  # ARC lemma: directory >= c implies a full cache
        assert t.n_byp and t.n_over, "bypass and override paths not exercised"
        assert all(k in t.H for k in list(t.T1) + list(t.T2))


def test_inactive_is_arc():
    tr = SUITE['zipf_100k']()[:30_000]
    for c in (30, 300):
        t = Tide(c, W=10**9)  # never activates
        assert hits(t, tr) == hits(ARC(c), tr)
        t2 = Tide(c, W=10**9); sh = ARC(c)
        for x in tr: t2.access(x); sh.access(x)
        assert list(t2.shadow.T1) == list(sh.T1) and list(t2.shadow.T2) == list(sh.T2)


def test_labels():
    from bisect import bisect_left
    tr = mixed_trace(8000)
    pos = {}
    for i, x in enumerate(tr): pos.setdefault(x, []).append(i + 1)  # 1-based request times
    W = 200
    t = Tide(20, W=W, log=True)
    for x in tr: t.access(x)
    assert len(t.log) > 1000
    for k, t0, d in t.log:                # candidate k snapshotted at time t0
        p = pos[k]; j = bisect_left(p, t0 + 1)
        true_d = p[j] - t0 if j < len(p) else 10**12
        if d is None: assert true_d >= W, (k, t0, true_d)
        else: assert d == true_d and d < W, (k, t0, d, true_d)


def test_grad():
    rng = np.random.default_rng(0)
    net = Net(12, n_hid=8, dtype=np.float64)
    X, y = rng.standard_normal((20, 12)), rng.standard_normal(20)
    _, G = net.grads(X, y)
    for p, g in zip(net.P, G):
        for idx in list(np.ndindex(p.shape))[:10]:
            old = p[idx]
            p[idx] = old + 1e-6; lp = net.grads(X, y)[0]
            p[idx] = old - 1e-6; lm = net.grads(X, y)[0]
            p[idx] = old
            num = (lp - lm) / 2e-6
            assert abs(num - g[idx]) <= 1e-4 * max(1.0, abs(num)), (num, g[idx])


def test_learns_loop():
    tr = list(range(17)) * 1500  # loop of c+1 keys: LRU and ARC get 0%
    c, half = 16, len(tr) // 2
    ht, ha = hits(Tide(c), tr), hits(ARC(c), tr)
    gain = 100 * (sum(ht[half:]) - sum(ha[half:])) / half
    assert gain > 40, gain


def test_oracle_ceiling():
    tr = SUITE['zipf_100k']()
    c = 185
    o, a = sum(hits(TideOracle(c, next_use(tr)), tr)), sum(hits(ARC(c), tr))
    assert o >= a, (o, a)
    assert sum(belady(tr, c, next_use(tr))) >= a


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
