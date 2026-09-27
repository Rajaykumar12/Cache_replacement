"""Benchmark TIDE against LRU, LFU, ARC, S3-FIFO, SIEVE, Belady and the TIDE oracle.

  python3 bench.py                 # full suite: 12 CSV traces + 3 x 1M-request trend traces, 3 cache sizes
  python3 bench.py --quick         # trend traces at 200k requests
  python3 bench.py --seeds 3       # TIDE over 3 seeds
  python3 bench.py --only cycle,phases
Writes bench_results.csv and bench_curves.csv (windowed hit rate of TIDE vs ARC).
"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from policies import ARC, LFU, LRU, S3FIFO, SIEVE, LFUPerfect, belady, next_use
from tide import Tide, TideOracle
from traces import SUITE, TREND

ROOT = os.path.dirname(os.path.abspath(__file__))
FRACS = (0.001, 0.01, 0.1)
BASE = {'LRU': LRU, 'LFU': LFU, 'LFU*': LFUPerfect, 'ARC': ARC, 'S3-FIFO': S3FIFO, 'SIEVE': SIEVE}
HEAD = ['LRU', 'LFU', 'ARC', 'S3-FIFO', 'SIEVE', 'TIDE']  # contenders; LFU* (global counts) is info-only
WINDOWS = 100


def run(policy, trace):
    hits, acc = bytearray(len(trace)), policy.access
    t0 = time.perf_counter()
    for i, x in enumerate(trace): hits[i] = acc(x)
    return hits, time.perf_counter() - t0


def rec(name, n, fp, c, f, pol, seed, hits, dt, t=None):
    h = np.frombuffer(bytes(hits), np.uint8)
    r = dict(trace=name, n=n, footprint=fp, c=c, frac=f, policy=pol, seed=seed,
             hit=100 * h.mean(), hit_1st=100 * h[:n // 2].mean(), hit_2nd=100 * h[n // 2:].mean(),
             req_per_s=n / dt if dt else '', p50_us='', p99_us='', train_share='', override_pct='', bypass_pct='', active_pct='')
    if t is not None:
        lat = np.array(t.lat) / 1e3
        r.update(p50_us=np.percentile(lat, 50) if len(lat) else '', p99_us=np.percentile(lat, 99) if len(lat) else '',
                 train_share=t.train_s / dt, override_pct=100 * t.n_over / max(1, t.n_dec),
                 bypass_pct=100 * t.n_byp / max(1, t.n_dec), active_pct=100 * t.n_active_req / n)
    return r


def curve(name, c, pol, hits):
    h = np.frombuffer(bytes(hits), np.uint8)
    w = [100 * a.mean() for a in np.array_split(h, WINDOWS)]
    return [dict(trace=name, c=c, policy=pol, window=i, hit=v) for i, v in enumerate(w)]


def cell(args):
    name, f, n_trend, seeds = args
    trace = SUITE[name](n_trend) if name in TREND else SUITE[name]()
    n, fp = len(trace), len(set(trace))
    c = max(16, round(f * fp))
    nxt = next_use(trace)
    out, curves = [], []
    for pol, mk in BASE.items():
        hits, dt = run(mk(c), trace)
        out.append(rec(name, n, fp, c, f, pol, 0, hits, dt))
        if pol == 'ARC': curves += curve(name, c, pol, hits)
    out.append(rec(name, n, fp, c, f, 'Belady', 0, belady(trace, c, nxt), 0))
    o = TideOracle(c, nxt)
    hits, dt = run(o, trace)
    out.append(rec(name, n, fp, c, f, 'TIDE-oracle', 0, hits, dt))
    for s in range(seeds):
        t = Tide(c, seed=s, record=True)
        hits, dt = run(t, trace)
        out.append(rec(name, n, fp, c, f, 'TIDE', s, hits, dt, t))
        if s == 0: curves += curve(name, c, 'TIDE', hits)
    return out, curves


def summarize(rows):
    cells = {}
    for r in rows:
        k = (r['trace'], r['c'])
        cells.setdefault(k, {}).setdefault(r['policy'], []).append(r)
    mean = lambda rs, key='hit': float(np.mean([r[key] for r in rs]))
    print('\nHit rate (%) per cell; TIDE = mean over seeds, dARC = TIDE - ARC')
    cols = HEAD + ['LFU*', 'TIDE-oracle', 'Belady']
    print('%-18s %6s ' % ('trace', 'c') + ' '.join('%8s' % p[:8] for p in cols) + '    dARC')
    norm = {p: [] for p in HEAD}
    wins = {p: 0 for p in HEAD}
    d_arc, gap_share, best_trend = [], [], {}
    for (tr, c), P in cells.items():
        h = {p: mean(P[p]) for p in cols}
        d = h['TIDE'] - h['ARC']
        d_arc.append((d, tr, c))
        print('%-18s %6d ' % (tr, c) + ' '.join('%8.2f' % h[p] for p in cols) + '  %+6.2f' % d)
        den = h['Belady'] - h['LRU']
        if den > 0.5:
            for p in HEAD: norm[p].append((h[p] - h['LRU']) / den)
        wins[max(HEAD, key=lambda p: h[p])] += 1
        if tr in TREND:
            best_trend[tr] = max(best_trend.get(tr, -1e9), d)
            if h['TIDE-oracle'] - h['ARC'] > 0.5: gap_share.append(d / (h['TIDE-oracle'] - h['ARC']))
    tide = [r for r in rows if r['policy'] == 'TIDE']
    ge = sum(d >= 0 for d, _, _ in d_arc)
    worst = min(d_arc)
    print('\nMean normalized hit rate (r - LRU)/(Belady - LRU) over %d cells:' % len(norm['TIDE']))
    for p in sorted(HEAD, key=lambda p: -np.mean(norm[p])): print('  %-8s %.3f' % (p, np.mean(norm[p])))
    print('Best policy per cell (wins):', wins)
    print('TIDE >= ARC in %d/%d cells (%.0f%%); worst TIDE - ARC = %+.2f (%s, c=%d)' % (ge, len(d_arc), 100 * ge / len(d_arc), *worst))
    print('Best TIDE - ARC per trend trace:', {k: round(v, 2) for k, v in best_trend.items()})
    if gap_share: print('Share of the (oracle - ARC) gap realized on trend cells: %.0f%%' % (100 * np.mean(gap_share)))
    for r in (r for r in tide if r['trace'] == 'phases' and r['seed'] == 0):
        a = next(x for x in rows if x['trace'] == 'phases' and x['c'] == r['c'] and x['policy'] == 'ARC')
        print('phases c=%d: TIDE - ARC round 1 %+.2f, round 2 %+.2f' % (r['c'], r['hit_1st'] - a['hit_1st'], r['hit_2nd'] - a['hit_2nd']))


def speed(names, n_trend):
    """Serial pass (no CPU contention) at the 1% size: throughput, decision latency, training share."""
    print('\nSpeed (serial, 1%% cache size)\n%-18s %7s %10s %10s %8s %8s %8s' % ('trace', 'c', 'ARC req/s', 'TIDE req/s', 'p50 us', 'p99 us', 'train %'))
    rows = []
    for name in names:
        trace = SUITE[name](n_trend) if name in TREND else SUITE[name]()
        c = max(16, round(0.01 * len(set(trace))))
        _, da = run(ARC(c), trace)
        t = Tide(c, record=True)
        _, dt = run(t, trace)
        lat = np.array(t.lat) / 1e3
        r = dict(trace=name, n=len(trace), c=c, arc_req_per_s=len(trace) / da, tide_req_per_s=len(trace) / dt,
                 p50_us=np.percentile(lat, 50), p99_us=np.percentile(lat, 99), train_pct=100 * t.train_s / dt)
        rows.append(r)
        print('%-18s %7d %9.0fk %9.0fk %8.1f %8.1f %8.1f' % (name, c, r['arc_req_per_s'] / 1e3, r['tide_req_per_s'] / 1e3,
              r['p50_us'], r['p99_us'], r['train_pct']))
    with open(os.path.join(ROOT, 'bench_speed.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--only', default='')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    names = a.only.split(',') if a.only else list(SUITE)
    n_trend = 200_000 if a.quick else 1_000_000
    jobs = sorted(((n, f, n_trend, a.seeds) for n in names for f in FRACS), key=lambda j: j[0] not in TREND)
    rows, curves = [], []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (r, cv) in enumerate(ex.map(cell, jobs, chunksize=1)):
            rows += r; curves += cv
            print('\r%d/%d cells, %.0fs' % (i + 1, len(jobs), time.time() - t0), end='', flush=True)
    for fn, data in (('bench_results.csv', rows), ('bench_curves.csv', curves)):
        with open(os.path.join(ROOT, fn), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    summarize(rows)
    speed([n for n in names if n in TREND or n in ('zipf_100k', 'web_traffic_100k')], n_trend)


if __name__ == '__main__':
    main()
