"""TideFast: TIDE with a Numba hot path and optional asynchronous training.

Same policy as tide.Tide (which stays the reference implementation); only the machinery changes:
  * per-key metadata lives in a float64 array M[slot, 9] (NaN = gap not seen yet) instead of Python lists;
  * one compiled call per decision builds all candidate feature rows and scores them (feat_score);
  * the Adam step is compiled and releases the GIL (adam_step).
train='sync' trains inline exactly when Tide does (reproducible hit rates). train='async' hands training to a
background thread: the request path only fills the replay buffer and signals every `every` labels; the
trainer steps a private copy of the weights and publishes it with one reference swap (double buffering), so
decisions never wait for training and never see half-updated weights. Call close() when done.
"""
import math
import threading
import time

import numpy as np
from numba import njit

from tide import F, Tide

# fastmath without 'nnan'/'ninf': NaN marks an unseen gap, and nnan would fold math.isnan() to False.
FM = {'reassoc', 'contract', 'arcp', 'nsz', 'afn'}


@njit(cache=True)
def touch(M, i, t, kd0, kd1, kd2):
    age = t - M[i, 0]
    M[i, 3] = M[i, 2]; M[i, 2] = M[i, 1]; M[i, 1] = age
    M[i, 4] = age if math.isnan(M[i, 4]) else 0.7 * M[i, 4] + 0.3 * age
    M[i, 5] = 1.0 + M[i, 5] * math.exp(-age * kd0)
    M[i, 6] = 1.0 + M[i, 6] * math.exp(-age * kd1)
    M[i, 7] = 1.0 + M[i, 7] * math.exp(-age * kd2)
    M[i, 8] += 1.0; M[i, 0] = t


@njit(cache=True, fastmath=FM)
def feat_score(M, slots, n, nt, in_t2, t, ic, lm, kd0, kd1, kd2, p, n_in, n_hid, X, out, do_score):
    """Rows 0..nt-1 are list keys (in_t2 flag); rows nt..n-1 is the incoming key. Fills X; scores into out."""
    for r in range(n):
        i = slots[r]
        age = t - M[i, 0]
        X[r, 0] = math.log1p(age * ic)
        for j in range(1, 5):
            g = M[i, j]
            X[r, j] = lm if math.isnan(g) else math.log1p(g * ic)
        X[r, 5] = min(M[i, 8] - 1.0, 3.0) / 3.0
        X[r, 6] = math.log1p(M[i, 5] * math.exp(-age * kd0))
        X[r, 7] = math.log1p(M[i, 6] * math.exp(-age * kd1))
        X[r, 8] = math.log1p(M[i, 7] * math.exp(-age * kd2))
        X[r, 9] = math.log1p(M[i, 8])
        X[r, 10] = 0.0 if r < nt else 1.0
        X[r, 11] = in_t2 if r < nt else 0.0
    if not do_score: return
    a, b = n_in * n_hid, n_in * n_hid + n_hid
    z = np.empty(n_hid, np.float32)
    for r in range(n):
        for h in range(n_hid): z[h] = p[a + h]
        for j in range(n_in):  # inner loop over contiguous hidden units: vectorizes
            xj = X[r, j]
            for h in range(n_hid): z[h] += xj * p[j * n_hid + h]
        s = p[-1]
        for h in range(n_hid):
            if z[h] > 0: s += z[h] * p[b + h]
        out[r] = s


@njit(nogil=True, cache=True, fastmath=FM)
def adam_step(p, m, v, k, X, y, lr, n_in, n_hid):
    """One Adam step (k = step number, from 1) on MSE; same math as tide.Net.grads/step. Returns the loss."""
    B = X.shape[0]
    a, b = n_in * n_hid, n_in * n_hid + n_hid
    g = np.zeros(p.shape[0], np.float32)
    z = np.empty(n_hid, np.float32)
    dh = np.empty(n_hid, np.float32)
    loss = 0.0
    for r in range(B):
        for h in range(n_hid): z[h] = p[a + h]
        for j in range(n_in):
            xj = X[r, j]
            for h in range(n_hid): z[h] += xj * p[j * n_hid + h]
        out = p[-1]
        for h in range(n_hid):
            if z[h] > 0: out += z[h] * p[b + h]
        e = out - y[r]
        loss += e * e
        do = 2.0 * e / B
        g[-1] += do
        for h in range(n_hid):
            if z[h] > 0:
                g[b + h] += z[h] * do
                dh[h] = do * p[b + h]
            else:
                dh[h] = 0.0
            g[a + h] += dh[h]
        for j in range(n_in):
            xj = X[r, j]
            for h in range(n_hid): g[j * n_hid + h] += xj * dh[h]
    c1, c2 = 1.0 - 0.9 ** k, 1.0 - 0.999 ** k
    for q in range(p.shape[0]):
        m[q] = 0.9 * m[q] + 0.1 * g[q]
        v[q] = 0.999 * v[q] + 0.001 * g[q] * g[q]
        p[q] -= (lr / c1) * m[q] / (math.sqrt(v[q] / c2) + 1e-8)
    return loss / B


@njit(nogil=True, cache=True)
def adam_steps(p, m, v, k0, XR, yR, filled, nsteps, B, lr, n_in, n_hid, seed):
    """nsteps Adam steps on minibatches of B rows drawn from the first `filled` replay rows, without the GIL.
    Running the whole backlog in one call matters: re-acquiring the GIL after each step would cost up to one
    interpreter switch interval (5 ms) while the request thread is busy."""
    np.random.seed(seed)
    X = np.empty((B, XR.shape[1]), np.float32)
    y = np.empty(B, np.float32)
    for s in range(nsteps):
        for r in range(B):
            i = np.random.randint(0, filled)
            X[r] = XR[i]; y[r] = yR[i]
        adam_step(p, m, v, k0 + s + 1, X, y, lr, n_in, n_hid)


class TideFast(Tide):
    def __init__(s, c, train='sync', **kw):
        super().__init__(c, **kw)
        assert train in ('sync', 'async')
        s.mode = train
        cap = s.hcap + c + 64
        s.M = np.full((cap, 9), np.nan)
        s.free = list(range(cap - 1, -1, -1))
        s.slots = np.zeros(6, np.int64)
        s.sc = np.zeros(6, np.float32)
        s.n_in, s.n_hid = F, s.net.W1.shape[1]
        s.live = s.net.flat  # weights read by decisions; the Net's views (W1, ...) alias it in sync mode
        s.n_steps = 0
        if train == 'async':
            s.live = s.net.flat.copy()
            s.req, s.done, s.stop = 0, 0, False  # steps requested (request thread) / done (trainer); each has one writer
            s.ev = threading.Event()
            s.th = threading.Thread(target=s._trainer, daemon=True)
            s.th.start()

    # ---------- metadata ----------
    def _slot(s):
        if not s.free:  # grow; rarely needed since resident keys are bounded by c
            n = len(s.M)
            s.M = np.vstack([s.M, np.full((n, 9), np.nan)])
            s.free = list(range(2 * n - 1, n - 1, -1))
        return s.free.pop()

    def _touch(s, x):
        i = s.H.get(x)
        if i is None:
            i = s._slot()
            s.M[i] = (s.t, np.nan, np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0)
            s.H[x] = i
            if len(s.H) > s.hcap: s._trim()
            return
        kd = s.kd
        touch(s.M, i, float(s.t), kd[0], kd[1], kd[2])
        s.H.move_to_end(x)

    def _trim(s):
        H, T1, T2 = s.H, s.T1, s.T2
        while len(H) > s.hcap:
            k = next(iter(H))
            if k in T1 or k in T2: H.move_to_end(k)
            else: s.free.append(H.pop(k))

    # ---------- decisions ----------
    def _choose(s, L, x):
        t0 = time.perf_counter_ns() if s.lat is not None and s.active else 0
        it = iter(L)
        keys = [k for _, k in zip(range(4), it)]
        nt = len(keys)
        if len(L) > 4: keys.append(next(reversed(L)))
        nl = len(keys)
        if x is not None: keys.append(x)
        n = len(keys)
        H, sl = s.H, s.slots
        for r in range(n): sl[r] = H[keys[r]]
        kd = s.kd
        feat_score(s.M, sl, n, nl, 1.0 if L is s.T2 else 0.0, float(s.t), s.inv_c, s.lg_miss, kd[0], kd[1], kd[2],
                   s.live, s.n_in, s.n_hid, s.X, s.sc, s.active)
        pend, q, t, X = s.pend, s.q, s.t, s.X
        for r in range(n):  # label snapshots are taken even while inactive (labels don't depend on the action)
            k = keys[r]
            if k not in pend:
                e = (k, t, X[r].copy()); pend[k] = e; q.append(e)
        if not s.active: return keys[0]
        sc, d, df = s._score(s.sc, keys), s.delta, s.delta_far
        best, bi = sc[0], 0
        for i in range(1, n):
            v = sc[i] - (d if i < nt else df)
            if v > best: best, bi = v, i
        s.n_dec += 1
        if bi: s.n_over += 1
        if t0: s.lat.append(time.perf_counter_ns() - t0)
        return keys[bi]

    def _score(s, sc, keys):
        return sc

    # ---------- learning ----------
    def _batch(s, rng):
        b = rng.integers(0, s.filled, s.BATCH)
        return s.XR[b], s.yR[b]

    def _train(s):
        if s.mode == 'async':
            s.req += 1; s.ev.set()
            s._update_active()
            return
        t0 = time.perf_counter()
        net = s.net
        if net.k == 0: net.b2[0] = s.yR[:s.filled].mean()
        net.k += 1
        X, y = s._batch(s.rng)
        adam_step(net.flat, net.m, net.v, net.k, X, y, net.lr, s.n_in, s.n_hid)
        s.n_steps += 1
        s.train_s += time.perf_counter() - t0
        s._update_active()

    def _trainer(s):
        net, rng = s.net, np.random.default_rng(s.rng.integers(1 << 62))
        while True:
            s.ev.wait(); s.ev.clear()
            while s.done < s.req and not s.stop:
                n = min(s.req - s.done, 64)  # publish at least every 64 steps
                t0 = time.perf_counter()
                if net.k == 0: net.b2[0] = s.yR[:s.filled].mean()
                adam_steps(net.flat, net.m, net.v, net.k, s.XR, s.yR, s.filled, n, s.BATCH, net.lr, s.n_in, s.n_hid,
                           int(rng.integers(1 << 31)))
                net.k += n
                s.live = net.flat.copy()  # publish: one atomic reference swap
                s.n_steps += n; s.done += n
                s.train_s += time.perf_counter() - t0
            if s.stop: return

    def close(s):
        if s.mode == 'async' and s.th.is_alive():
            s.stop = True; s.ev.set(); s.th.join()


class TideOracleFast(TideFast):
    """tide.TideOracle on the fast machinery: same candidates, scored with the true next use."""

    def __init__(s, c, nxt, **kw):
        super().__init__(c, force=True, delta=0.0, delta_far=0.0, **kw)
        s.nxt, s.nu = nxt, {}

    def access(s, x):
        s.nu[x] = s.nxt[s.t]  # s.t == index of this request before increment
        return super().access(x)

    def _score(s, sc, keys):
        t, ic, nu = s.t, s.inv_c, s.nu
        return [math.log1p((nu[k] + 1 - t) * ic) for k in keys]

    def _train(s): pass
