"""Benchmark TIDE against classic and learned baselines, Belady and the TIDE oracle.

Configuration behind the published results (about 10 minutes on 12 cores):
  python3 bench.py --set synth --seeds 3 --instances 2 --trend-n 500000 --workers 11   # -> bench_results_synth.csv
  python3 bench.py --set real --seeds 3 --real-n 1000000 --workers 11                  # -> bench_results_real.csv
  python3 bench.py --latency --latency-n 500000                                        # -> bench_speed.csv
  python3 bench.py --async-hits --seeds 3 --trend-n 500000 --real-n 1000000 --workers 5  # -> bench_async.csv
The full-scale run (--seeds 5 --instances 3, 1M trend traces, 10M real requests with --workers 3 for RAM) takes
about 2.5 hours; --quick shortens the trend traces to 200k requests for smoke tests.
TIDE runs on TideFast (train='sync'), which test_tide.py checks against the reference tide.Tide.
"""
import argparse
import csv
import os
import time
from array import array
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import traces
from policies import ARC, LFU, LRU, S3FIFO, SIEVE, LFUPerfect, LibCS, belady, next_use
from tide import Tide
from tide_fast import TideFast, TideOracleFast

ROOT = os.path.dirname(os.path.abspath(__file__))
FRACS = (0.001, 0.01, 0.1)
LEARNED = ('LIRS', 'LeCaR', 'Cacheus', 'LRB')  # libCacheSim reference implementations
BASE = {'LRU': LRU, 'LFU': LFU, 'LFU*': LFUPerfect, 'ARC': ARC, 'S3-FIFO': S3FIFO, 'SIEVE': SIEVE,
        **{n: (lambda c, n=n: LibCS(n, c)) for n in LEARNED}}
HEAD = ['LRU', 'LFU', 'ARC', 'S3-FIFO', 'SIEVE', *LEARNED, 'TIDE']  # contenders; LFU* (global counts) is info-only
WINDOWS = 100
FIELDS = ['trace', 'inst', 'n', 'footprint', 'c', 'frac', 'policy', 'seed', 'hit', 'hit_1st', 'hit_2nd', 'req_per_s',
          'p50_us', 'p99_us', 'train_share', 'override_pct', 'bypass_pct', 'active_pct']


def run(policy, trace):
    hits, acc = bytearray(len(trace)), policy.access
    t0 = time.perf_counter()
    for i, x in enumerate(trace): hits[i] = acc(x)
    return hits, time.perf_counter() - t0


def rec(name, inst, n, fp, c, f, pol, seed, hits, dt, t=None):
    h = np.frombuffer(bytes(hits), np.uint8)
    r = dict(trace=name, inst=inst, n=n, footprint=fp, c=c, frac=f, policy=pol, seed=seed,
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
    return [dict(trace=name, c=c, policy=pol, window=i, hit=100 * a.mean()) for i, a in enumerate(np.array_split(h, WINDOWS))]


def job(args):
    """One (trace, instance, size) and one policy: 'b:<baseline>', 'ref' (Belady + oracle) or a TIDE seed."""
    name, inst, f, n_trend, n_real, what = args
    trace = traces.make(name, inst, n_trend, n_real)
    n, fp = len(trace), len(set(trace))
    c = max(16, round(f * fp))
    out, curves = [], []
    trend0 = name in traces.TREND and inst == 0
    if what.startswith('b:'):
        pol = what[2:]
        hits, dt = run(BASE[pol](c), trace)
        out.append(rec(name, inst, n, fp, c, f, pol, 0, hits, dt))
        if pol == 'ARC' and trend0: curves += curve(name, c, pol, hits)
    elif what == 'ref':
        nxt = next_use(trace)
        out.append(rec(name, inst, n, fp, c, f, 'Belady', 0, belady(trace, c, nxt), 0))
        hits, dt = run(TideOracleFast(c, nxt), trace)
        out.append(rec(name, inst, n, fp, c, f, 'TIDE-oracle', 0, hits, dt))
    else:
        seed = int(what)
        t = TideFast(c, seed=seed, record=True)
        hits, dt = run(t, trace)
        out.append(rec(name, inst, n, fp, c, f, 'TIDE', seed, hits, dt, t))
        if seed == 0 and trend0: curves += curve(name, c, 'TIDE', hits)
    return out, curves


def write(fn, rows, fields=None):
    with open(os.path.join(ROOT, fn), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields or list(rows[0])); w.writeheader(); w.writerows(rows)


def suite(a):
    names = a.only.split(',') if a.only else list(traces.REAL if a.set == 'real' else [*traces.TREND, *traces.SMALL])
    n_trend = 200_000 if a.quick else a.trend_n
    inst = 1 if a.set == 'real' else a.instances
    heavy = lambda nm: nm in traces.REAL or nm in traces.TREND
    jobs = [(nm, i, f, n_trend, a.real_n, w) for nm in names for i in range(inst) for f in FRACS
            for w in (*(f'b:{p}' for p in BASE), 'ref', *map(str, range(a.seeds)))]
    slow = lambda w: w == 'b:LRB' or not w.startswith('b:')
    jobs.sort(key=lambda j: (not heavy(j[0]), not slow(j[5]), -j[2]))  # biggest first for better packing
    rows, curves = [], []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (r, cv) in enumerate(ex.map(job, jobs, chunksize=1)):
            rows += r; curves += cv
            print('\r%d/%d jobs, %.0fs' % (i + 1, len(jobs), time.time() - t0), end='', flush=True)
    print()
    write(f'bench_results_{a.set}.csv', rows, FIELDS)
    if curves: write('bench_curves.csv', curves)
    import analysis
    analysis.summarize(a.set)


# ---------- latency ----------
def timed(policy, trace):
    """Per-request latency (ns) of every access, plus throughput."""
    lat, acc, ns = array('q', bytes(8 * len(trace))), policy.access, time.perf_counter_ns
    t0 = time.perf_counter()
    for i, x in enumerate(trace):
        s = ns(); acc(x); lat[i] = ns() - s
    return np.frombuffer(lat, np.int64) / 1e3, time.perf_counter() - t0


def latency(a):
    """Serial (one process at a time, no CPU contention), 1% cache size, first --latency-n requests of each trace."""
    names = a.only.split(',') if a.only else list(traces.REAL)
    rows = []
    print('%-12s %-11s %9s %8s %8s %9s %9s %8s' % ('trace', 'policy', 'req/s', 'p50 us', 'p99 us', 'p99.9 us', 'dec p99', 'train %'))
    for name in names:
        trace = traces.make(name, 0, a.trend_n, a.latency_n)[:a.latency_n]
        c = max(16, round(0.01 * len(set(trace))))
        TideFast(16).access(0)  # JIT warm-up outside the timed runs
        for pol, mk in (('ARC', lambda: ARC(c)), ('TIDE-ref', lambda: Tide(c, record=True)),
                        ('TIDE-fast', lambda: TideFast(c, record=True)), ('TIDE-async', lambda: TideFast(c, train='async', record=True))):
            p = mk()
            lat, dt = timed(p, trace)
            if hasattr(p, 'close'): p.close()
            dec = np.array(getattr(p, 'lat', None) or [0]) / 1e3
            r = dict(trace=name, n=len(trace), c=c, policy=pol, req_per_s=len(trace) / dt,
                     p50_us=np.percentile(lat, 50), p99_us=np.percentile(lat, 99), p999_us=np.percentile(lat, 99.9),
                     dec_p50_us=np.percentile(dec, 50), dec_p99_us=np.percentile(dec, 99),
                     train_pct=100 * getattr(p, 'train_s', 0) / dt, steps=getattr(p, 'n_steps', getattr(getattr(p, 'net', None), 'k', 0)))
            rows.append(r)
            print('%-12s %-11s %8.0fk %8.2f %8.2f %9.2f %9.2f %8.1f' % (name, pol, r['req_per_s'] / 1e3, r['p50_us'], r['p99_us'],
                  r['p999_us'], r['dec_p99_us'], r['train_pct']), flush=True)
    write('bench_speed.csv', rows)


# ---------- async hit-rate cost ----------
def async_job(args):
    name, f, seed, n_trend, n_real = args
    trace = traces.make(name, 0, n_trend, n_real)
    c = max(16, round(f * len(set(trace))))
    out = []
    for mode in ('sync', 'async'):
        t = TideFast(c, seed=seed, train=mode)
        hits, _ = run(t, trace)
        t.close()
        out.append(dict(trace=name, c=c, frac=f, seed=seed, mode=mode, hit=100 * np.frombuffer(bytes(hits), np.uint8).mean(),
                        steps=t.n_steps))
    return out


def async_hits(a):
    names = a.only.split(',') if a.only else ['cycle', 'phases', *traces.REAL]
    jobs = [(nm, f, s, a.trend_n, a.real_n) for nm in names for f in (0.01,) for s in range(a.seeds)]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:  # each job uses 2 threads in async mode
        for i, r in enumerate(ex.map(async_job, jobs, chunksize=1)):
            rows += r; print('\r%d/%d' % (i + 1, len(jobs)), end='', flush=True)
    print()
    write('bench_async.csv', rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=('synth', 'real'), default='synth')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--instances', type=int, default=3)
    ap.add_argument('--only', default='')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--trend-n', type=int, default=1_000_000, help='requests per trend trace')
    ap.add_argument('--real-n', type=int, default=10_000_000, help='prefix of each real trace')
    ap.add_argument('--latency-n', type=int, default=2_000_000, help='requests per trace in --latency')
    ap.add_argument('--latency', action='store_true')
    ap.add_argument('--async-hits', action='store_true')
    a = ap.parse_args()
    if a.latency: latency(a)
    elif a.async_hits: async_hits(a)
    else: suite(a)


if __name__ == '__main__':
    main()
