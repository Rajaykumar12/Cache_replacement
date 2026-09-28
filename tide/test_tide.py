"""Assert-based checks for TIDE. Run: python3 test_tide.py"""
import random

import numpy as np

import os
import tempfile

import traces
from policies import ARC, LRU, LibCS, belady, next_use
from tide import Net, Tide, TideOracle
from tide_fast import TideFast, feat_score


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
    tr = traces.make('zipf_100k')[:30_000]
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
    tr = traces.make('zipf_100k')
    c = 185
    o, a = sum(hits(TideOracle(c, next_use(tr)), tr)), sum(hits(ARC(c), tr))
    assert o >= a, (o, a)
    assert sum(belady(tr, c, next_use(tr))) >= a


def test_fast_matches_reference():
    tr = mixed_trace()
    for c in (16, 50):
        assert hits(TideFast(c, W=10**9), tr) == hits(ARC(c), tr)  # inactive TideFast is exactly ARC
        ref, fast = sum(hits(Tide(c), tr)), sum(hits(TideFast(c), tr))
        assert abs(ref - fast) <= 0.003 * len(tr), (c, ref, fast)  # float32 op order may flip rare near-ties


def test_fast_features():
    tr, c = mixed_trace(10_000), 40
    a, b = Tide(c, W=10**9), TideFast(c, W=10**9)  # inactive: identical ARC state and metadata
    for x in tr: a.access(x); b.access(x)
    keys = list(a.T1)[:4] + [tr[-1]]
    ref = np.array([a._feat(k, 0.0, 0.0) for k in keys[:4]] + [a._feat(keys[4], 0.0, 1.0)], np.float32)
    sl = np.array([b.H[k] for k in keys], np.int64)
    X, out = np.zeros((6, 12), np.float32), np.zeros(6, np.float32)
    kd = b.kd
    feat_score(b.M, sl, 5, 4, 0.0, float(b.t), b.inv_c, b.lg_miss, kd[0], kd[1], kd[2], b.net.flat, 12, 64, X, out, True)
    assert np.allclose(X[:5], ref, rtol=1e-5, atol=1e-6), np.abs(X[:5] - ref).max()
    assert np.allclose(out[:5], a.net.predict(ref), rtol=1e-4, atol=1e-5)


def test_async_trainer():
    tr, c = mixed_trace(), 16
    t = TideFast(c, train='async')
    h = sum(hits(t, tr))
    t.close()
    assert not t.th.is_alive() and t.n_steps > 0 and h > 0


def test_libcs_baselines():
    tr = mixed_trace(20_000)
    assert hits(LibCS('LRU', 50), tr) == hits(LRU(50), tr)  # adapter reproduces our LRU exactly
    loop = list(range(17)) * 500
    assert sum(hits(LRU(16), loop)) == 0 and sum(hits(LibCS('LIRS', 16), loop)) > 0.5 * len(loop)


def test_generators_seeded():
    for name in list(traces.SMALL) + list(traces.TREND):
        a, b = traces.make(name, 0, 50_000), traces.make(name, 1, 50_000)
        assert a == traces.make(name, 0, 50_000) and a != b, name


def test_oracle_general_reader():
    import zstandard
    import realtraces
    ids = np.array([7, 3, 7, 10**15, 3], np.uint64)
    rec = np.zeros(len(ids), realtraces.REC); rec['id'] = ids; rec['next'] = 99
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 't.zst')
        with open(f, 'wb') as fh: fh.write(zstandard.ZstdCompressor().compress(rec.tobytes()))
        with open(f, 'rb') as fh: out = realtraces.read_oracle(fh, 4)
    assert out.tolist() == [1, 0, 1, 2]  # dense, order-preserving remap; only first n records


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
