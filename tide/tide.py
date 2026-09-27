"""TIDE: Trend-Informed Delayed-label Eviction.

ARC shell + a tiny online-trained MLP that picks the victim. ARC's REPLACE picks the list; the model
scores that list's 4 LRU-tail keys, its MRU key, and (on a new-key miss) the incoming key itself,
where choosing the incoming key means bypass. The model regresses log1p(reuse distance / c) and learns
live from delayed labels: every scored candidate is labelled when it is next requested, or censored
after W requests. A key-only shadow ARC is the safety net: if TIDE misses more over a window, it
falls back to ARC's own victim, for longer after each repeated failure.
Deviating from ARC's victim needs a score margin: delta for tail keys, delta_far for MRU/bypass.
"""
import math
import time
from array import array
from collections import OrderedDict as OD, deque
from itertools import islice

import numpy as np

from policies import ARC

F = 12  # features per candidate, see Tide._feat


class Net:
    """F -> hidden (ReLU) -> 1 regression MLP with Adam, MSE loss."""

    def __init__(s, n_in, n_hid=64, lr=1e-3, seed=0, dtype=np.float32):
        r = np.random.default_rng(seed)
        s.flat = np.zeros(n_in * n_hid + 2 * n_hid + 1, dtype)  # all params in one buffer: one Adam update per step
        a, b = n_in * n_hid, n_in * n_hid + n_hid
        s.W1, s.b1, s.w2, s.b2 = s.flat[:a].reshape(n_in, n_hid), s.flat[a:b], s.flat[b:b + n_hid], s.flat[-1:]
        s.W1[:] = r.standard_normal((n_in, n_hid)) * math.sqrt(2 / n_in)
        s.w2[:] = r.standard_normal(n_hid) * math.sqrt(1 / n_hid)
        s.P = [s.W1, s.b1, s.w2, s.b2]
        s.m, s.v = np.zeros_like(s.flat), np.zeros_like(s.flat)
        s.lr, s.k = lr, 0

    def predict(s, X):
        h = X @ s.W1; h += s.b1; np.maximum(h, 0, out=h)
        return h @ s.w2 + s.b2[0]

    def grads(s, X, y):
        # einsum, not @: with the reference BLAS numpy links here, einsum is ~2x faster at these shapes.
        h = np.einsum('ij,jk->ik', X, s.W1); h += s.b1
        a = np.maximum(h, 0)
        e = a @ s.w2 + s.b2[0] - y
        do = (2 / len(y)) * e
        dh = np.outer(do, s.w2); dh *= h > 0
        return float(e @ e) / len(y), [np.einsum('ij,ik->jk', X, dh), dh.sum(0), a.T @ do, np.array([do.sum()], s.b2.dtype)]

    def step(s, X, y, b1=0.9, b2=0.999, eps=1e-8):
        loss, G = s.grads(X, y)
        g = np.concatenate([x.ravel() for x in G])
        s.k += 1
        c1, c2 = 1 - b1 ** s.k, 1 - b2 ** s.k
        m, v = s.m, s.v
        m *= b1; m += (1 - b1) * g
        v *= b2; v += (1 - b2) * g * g
        s.flat -= (s.lr / c1) * m / (np.sqrt(v / c2) + eps)
        return loss


class Tide(ARC):
    REPLAY, BATCH, MIN_LABELS = 32768, 256, 2048

    def __init__(s, c, seed=0, W=None, delta=0.1, delta_far=0.2, hist=8, lr=1e-3, every=32, hid=64, force=False, record=False, log=False):
        super().__init__(c)
        s.every = every                                        # new labels per training step
        s.t, s.W, s.delta, s.inv_c = 0, W or 8 * c, delta, 1.0 / c
        s.delta_far = delta if delta_far is None else delta_far  # margin for MRU / bypass candidates
        s.lg_miss = math.log1p(2 * s.W / c)                     # missing-gap sentinel and censored label
        s.kd = [math.log(2) / (h * c) for h in (1, 4, 16)]    # decayed-counter half-lives c, 4c, 16c
        s.H, s.hcap = OD(), hist * c                           # key -> [t_last, g1, g2, g3, ewma, C1, C2, C3, count]
        s.shadow = ARC(c)
        s.net = Net(F, n_hid=hid, lr=lr, seed=seed)
        s.rng = np.random.default_rng(seed)
        s.X = np.zeros((6, F), np.float32)
        s.pend, s.q = {}, deque()                              # key -> (key, t0, row); q ordered by t0
        s.XR = np.zeros((s.REPLAY, F), np.float32)
        s.yR = np.zeros(s.REPLAY, np.float32)
        s.j = s.filled = s.new = 0
        s.force = s.active = force
        s.win, s.wn, s.wm, s.ws, s.cool, s.fails = max(2000, 4 * c), 0, 0, 0, 0, 0
        s.n_dec = s.n_over = s.n_byp = s.n_active_req = 0
        s.train_s = 0.0
        s.lat = array('q') if record else None
        s.log = [] if log else None

    # ---------- per-key metadata and features ----------
    def _touch(s, x):
        m, t = s.H.get(x), s.t
        if m is None:
            s.H[x] = [t, None, None, None, None, 1.0, 1.0, 1.0, 1]
            if len(s.H) > s.hcap: s._trim()
            return
        age = t - m[0]
        m[3], m[2], m[1] = m[2], m[1], age
        m[4] = age if m[4] is None else 0.7 * m[4] + 0.3 * age
        kd, exp = s.kd, math.exp
        m[5] = 1.0 + m[5] * exp(-age * kd[0])
        m[6] = 1.0 + m[6] * exp(-age * kd[1])
        m[7] = 1.0 + m[7] * exp(-age * kd[2])
        m[8] += 1; m[0] = t
        s.H.move_to_end(x)

    def _trim(s):
        H, T1, T2 = s.H, s.T1, s.T2
        while len(H) > s.hcap:
            k = next(iter(H))
            if k in T1 or k in T2: H.move_to_end(k)  # resident keys always keep metadata
            else: del H[k]

    def _feat(s, k, in_t2, is_x):
        m = s.H[k]
        age, ic, lm, kd = s.t - m[0], s.inv_c, s.lg_miss, s.kd
        log1p, exp = math.log1p, math.exp
        g1, g2, g3, ew = m[1], m[2], m[3], m[4]
        return [log1p(age * ic),
                lm if g1 is None else log1p(g1 * ic),
                lm if g2 is None else log1p(g2 * ic),
                lm if g3 is None else log1p(g3 * ic),
                lm if ew is None else log1p(ew * ic),
                min(m[8] - 1, 3) / 3,
                log1p(m[5] * exp(-age * kd[0])),
                log1p(m[6] * exp(-age * kd[1])),
                log1p(m[7] * exp(-age * kd[2])),
                log1p(m[8]), is_x, in_t2]

    # ---------- decisions ----------
    def _choose(s, L, x):
        """Victim among L's 4 LRU-tail keys, L's MRU key and (if given) the incoming key x."""
        t0 = time.perf_counter_ns() if s.lat is not None and s.active else 0
        keys = list(islice(L, 4))
        nt = len(keys)
        if len(L) > 4: keys.append(next(reversed(L)))
        in_t2 = 1.0 if L is s.T2 else 0.0
        feat = s._feat
        rows = [feat(k, in_t2, 0.0) for k in keys]
        if x is not None:
            keys.append(x); rows.append(feat(x, 0.0, 1.0))
        pend, q, t = s.pend, s.q, s.t
        for k, r in zip(keys, rows):  # label snapshots are taken even while inactive (labels don't depend on the action)
            if k not in pend:
                e = (k, t, r); pend[k] = e; q.append(e)
        if not s.active: return keys[0]
        n = len(rows)
        X = s.X[:n]; X[:] = rows
        sc = s._score(X, keys)
        best, bi, d, df = sc[0], 0, s.delta, s.delta_far
        for i in range(1, n):
            v = sc[i] - (d if i < nt else df)
            if v > best: best, bi = v, i
        s.n_dec += 1
        if bi: s.n_over += 1
        if t0: s.lat.append(time.perf_counter_ns() - t0)
        return keys[bi]

    def _score(s, X, keys):
        return s.net.predict(X).tolist()

    def _replace(s, x_in_B2):
        L, G = s._pick_list(x_in_B2)
        v = s._choose(L, None)
        del L[v]; G[v] = None

    # ---------- learning ----------
    def _label(s, e, y):
        if s.log is not None: s.log.append((e[0], e[1], None if y == s.lg_miss else s.t - e[1]))
        j = s.j
        s.XR[j] = e[2]; s.yR[j] = y
        s.j = (j + 1) % s.REPLAY
        if s.filled < s.REPLAY: s.filled += 1
        s.new += 1
        if s.new >= s.every:
            s.new = 0; s._train()

    def _train(s):
        t0 = time.perf_counter()
        if s.net.k == 0: s.net.b2[0] = s.yR[:s.filled].mean()
        b = s.rng.integers(0, s.filled, s.BATCH)
        s.net.step(s.XR[b], s.yR[b])
        s.train_s += time.perf_counter() - t0
        s._update_active()

    def _update_active(s):
        s.active = s.force or (s.cool == 0 and s.t >= s.W and s.filled >= s.MIN_LABELS)

    def _window(s):
        """Switch-on-worse vs the shadow ARC, with exponential backoff: repeated failures fall back longer."""
        if s.active and not s.force:
            if s.wm - s.ws > max(10, 0.01 * s.win):
                s.cool = 2 << min(s.fails, 4); s.fails += 1
            elif s.wm <= s.ws and s.fails:
                s.fails -= 1
        elif s.cool: s.cool -= 1
        s.wn = s.wm = s.ws = 0
        s._update_active()

    # ---------- request path ----------
    def access(s, x):
        s.t = t = s.t + 1
        q, pend = s.q, s.pend
        lim = t - s.W
        while q and q[0][1] <= lim:           # expire first: reuse at exactly d = W counts as censored
            e = q.popleft()
            if pend.get(e[0]) is e:
                del pend[e[0]]; s._label(e, s.lg_miss)
        e = pend.pop(x, None)
        if e is not None: s._label(e, math.log1p((t - e[1]) * s.inv_c))
        s._touch(x)
        sh = s.shadow.access(x)

        T1, T2, B1, B2, c = s.T1, s.T2, s.B1, s.B2, s.c
        hit = False
        if x in T1:
            del T1[x]; T2[x] = None; hit = True
        elif x in T2:
            T2.move_to_end(x); hit = True
        elif x in B1:
            s.p = min(c, s.p + max(1, len(B2) / len(B1)))
            s._replace(False); del B1[x]; T2[x] = None
        elif x in B2:
            s.p = max(0, s.p - max(1, len(B1) / len(B2)))
            s._replace(True); del B2[x]; T2[x] = None
        elif len(T1) + len(T2) >= c:          # new key, full cache: the incoming key is a candidate too
            L1 = len(T1) + len(B1)
            L, G = (T1, None) if L1 == c and len(T1) == c else s._pick_list(False)
            v = s._choose(L, x)
            if v == x:
                s.n_byp += 1                   # ponytail: bypassed keys leave no ARC ghost
            else:
                if L1 == c:
                    if len(T1) < c: B1.popitem(last=False)
                elif L1 + len(T2) + len(B2) == 2 * c:
                    B2.popitem(last=False)
                del L[v]
                if G is not None: G[v] = None
                T1[x] = None
        else:                                  # new key, cache not full: plain ARC case IV
            L1 = len(T1) + len(B1)
            if L1 == c:
                if len(T1) < c: B1.popitem(last=False); s._replace(False)
                else: T1.popitem(last=False)
            else:
                tot = L1 + len(T2) + len(B2)
                if tot >= c:
                    if tot == 2 * c: B2.popitem(last=False)
                    s._replace(False)
            T1[x] = None

        s.wn += 1
        if not hit: s.wm += 1
        if not sh: s.ws += 1
        if s.active: s.n_active_req += 1
        if s.wn >= s.win: s._window()
        return hit


class TideOracle(Tide):
    """Same candidate set and plumbing, but scores with the true next use: the ceiling of the design."""

    def __init__(s, c, nxt, **kw):
        super().__init__(c, force=True, delta=0.0, delta_far=0.0, **kw)
        s.nxt, s.nu = nxt, {}

    def access(s, x):
        s.nu[x] = s.nxt[s.t]  # s.t == index of this request before increment
        return super().access(x)

    def _score(s, X, keys):
        t, ic, nu = s.t, s.inv_c, s.nu
        return [math.log1p((nu[k] + 1 - t) * ic) for k in keys]

    def _train(s): pass
