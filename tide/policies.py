"""Baseline eviction policies. Every policy exposes access(key) -> bool (True on hit)."""
import heapq
from collections import OrderedDict as OD, deque


class ARC:
    """Megiddo & Modha, FAST'03 Fig. 4, uniform sizes. OrderedDict: first = LRU end, last = MRU end."""

    def __init__(s, c):
        s.c, s.p = c, 0.0
        s.T1, s.T2, s.B1, s.B2 = OD(), OD(), OD(), OD()

    def _pick_list(s, x_in_B2):
        # `not s.T2` is libCacheSim's guard; never fires for plain ARC with uniform sizes.
        if s.T1 and (len(s.T1) > s.p or (x_in_B2 and len(s.T1) == s.p) or not s.T2):
            return s.T1, s.B1
        return s.T2, s.B2

    def _replace(s, x_in_B2):
        L, B = s._pick_list(x_in_B2)
        k, _ = L.popitem(last=False)
        B[k] = None

    def access(s, x):
        if x in s.T1:
            del s.T1[x]; s.T2[x] = None; return True
        if x in s.T2:
            s.T2.move_to_end(x); return True
        if x in s.B1:
            s.p = min(s.c, s.p + max(1, len(s.B2) / len(s.B1)))
            s._replace(False); del s.B1[x]; s.T2[x] = None; return False
        if x in s.B2:
            s.p = max(0, s.p - max(1, len(s.B1) / len(s.B2)))
            s._replace(True); del s.B2[x]; s.T2[x] = None; return False
        L1 = len(s.T1) + len(s.B1)
        if L1 == s.c:
            if len(s.T1) < s.c:
                s.B1.popitem(last=False); s._replace(False)
            else:
                s.T1.popitem(last=False)
        else:
            tot = L1 + len(s.T2) + len(s.B2)
            if tot >= s.c:
                if tot == 2 * s.c: s.B2.popitem(last=False)
                s._replace(False)
        s.T1[x] = None
        return False


class LRU:
    def __init__(s, c): s.c, s.d = c, OD()

    def access(s, x):
        d = s.d
        if x in d: d.move_to_end(x); return True
        if len(d) >= s.c: d.popitem(last=False)
        d[x] = None
        return False


class LFU:
    """Standard in-cache LFU: O(1) frequency buckets, LRU tie-break, counts forgotten on eviction."""

    def __init__(s, c): s.c, s.f, s.b, s.minf = c, {}, {}, 0

    def access(s, x):
        f, b = s.f, s.b
        if x in f:
            n = f[x]; del b[n][x]
            if not b[n]:
                del b[n]
                if s.minf == n: s.minf = n + 1
            f[x] = n + 1; b.setdefault(n + 1, OD())[x] = None
            return True
        if len(f) >= s.c:
            k, _ = b[s.minf].popitem(last=False)
            if not b[s.minf]: del b[s.minf]
            del f[k]
        f[x] = 1; b.setdefault(1, OD())[x] = None; s.minf = 1
        return False


class LFUPerfect:
    """LFU with global (never forgotten) counts. Unbounded metadata; information-only baseline."""

    def __init__(s, c): s.c, s.cnt, s.cache, s.h, s.t = c, {}, {}, [], 0

    def access(s, x):
        s.t += 1
        n = s.cnt.get(x, 0) + 1; s.cnt[x] = n
        hit = x in s.cache
        if not hit and len(s.cache) >= s.c:
            while True:  # lazy heap: skip stale entries
                n0, t0, k = heapq.heappop(s.h)
                if s.cache.get(k) == t0: del s.cache[k]; break
        s.cache[x] = s.t
        heapq.heappush(s.h, (n, s.t, x))
        return hit


class S3FIFO:
    """Yang et al., SOSP'23: 10% small FIFO, main FIFO with 2-bit reinsertion, ghost FIFO."""

    def __init__(s, c):
        s.c, s.sc = c, max(1, int(0.1 * c)); s.mc = c - s.sc
        s.S, s.M, s.G, s.f, s.inS, s.inM = deque(), deque(), OD(), {}, set(), set()

    def _evictM(s):
        while True:
            t = s.M.popleft()
            if s.f[t] > 0: s.f[t] -= 1; s.M.append(t)
            else: s.inM.discard(t); del s.f[t]; return

    def _evictS(s):
        while s.S:
            t = s.S.popleft(); s.inS.discard(t)
            if s.f[t] > 1:
                s.f[t] = 0; s.M.append(t); s.inM.add(t)
                if len(s.M) > s.mc: s._evictM()
            else:
                del s.f[t]; s.G[t] = None
                if len(s.G) > s.mc: s.G.popitem(last=False)
                return

    def access(s, x):
        if x in s.inS or x in s.inM:
            s.f[x] = min(s.f[x] + 1, 3); return True
        while len(s.inS) + len(s.inM) >= s.c:
            if len(s.S) >= s.sc: s._evictS()
            else: s._evictM()
        s.f[x] = 0
        if x in s.G: del s.G[x]; s.M.append(x); s.inM.add(x)
        else: s.S.append(x); s.inS.add(x)
        return False


class SIEVE:
    """Zhang et al., NSDI'24. Queue as a list (index 0 = oldest); hand sweeps old -> new."""

    def __init__(s, c): s.c, s.q, s.vis, s.hand = c, [], {}, 0

    def access(s, x):
        vis = s.vis
        if x in vis: vis[x] = 1; return True
        q = s.q
        if len(q) >= s.c:
            h = s.hand
            if h >= len(q): h = 0
            while vis[q[h]]:
                vis[q[h]] = 0; h += 1
                if h == len(q): h = 0
            del vis[q.pop(h)]; s.hand = h
        q.append(x); vis[x] = 0
        return False


class LibCS:
    """A libCacheSim reference implementation (LIRS, LeCaR, Cacheus, LRB, ...) behind access(key) -> bool.
    Unit object sizes; clock_time = request index."""

    def __init__(s, name, c):
        import libcachesim as lcs
        s.cache, s.r, s.t = getattr(lcs, name)(cache_size=c), lcs.Request(), 0
        s.r.obj_size = 1
        s.get = s.cache.get

    def access(s, x):
        r = s.r
        r.obj_id = x; r.clock_time = s.t; s.t += 1
        return s.get(r)


def next_use(trace):
    """nxt[i] = index of the next request for trace[i], or len(trace) + 10**9 if none."""
    n, last = len(trace), {}
    nxt = [0] * n
    never = n + 10**9
    for i in range(n - 1, -1, -1):
        x = trace[i]; nxt[i] = last.get(x, never); last[x] = i
    return nxt


def belady(trace, c, nxt):
    """Belady's MIN. Returns per-request hits as a bytearray."""
    cache, heap, cur, hits = set(), [], {}, bytearray(len(trace))
    for i, x in enumerate(trace):
        if x in cache: hits[i] = 1
        elif len(cache) >= c:
            while True:
                negn, k = heapq.heappop(heap)
                if k in cache and cur[k] == -negn: cache.discard(k); break
        cache.add(x); cur[x] = nxt[i]; heapq.heappush(heap, (-nxt[i], x))
    return hits
