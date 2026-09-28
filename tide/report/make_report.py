"""Build the results PDF from bench_results_{synth,real}.csv, bench_curves.csv, bench_speed.csv and bench_async.csv.

  python3 report/make_report.py      # writes report/gen/* and compiles report/report.pdf (needs pdflatex + pgfplots)
"""
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import analysis  # noqa: E402

GEN = os.path.join(HERE, 'gen')
TREND = ['lifecycle', 'cycle', 'phases']
SHORT = {'zipf_100k': 'zipf-100k', 'web_traffic_100k': 'web', 'msr_hm_0': 'MSR hm\\_0', 'twitter_c50': 'Twitter c50',
         'meta_reag': 'Meta CDN reag'}
short = lambda t: SHORT.get(t, t.replace('zipf_alpha_', 'zipf-a').replace('_', '\\_'))
neg = lambda x: x.replace('-', '$-$')
pm = lambda v, ci, d=2: f'{v:.{d}f}' if ci == 0 else f'{v:.{d}f}\\,$\\pm$\\,{ci:.{d}f}'
tex_table = lambda spec, head, body: (f'\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head} \\\\\n\\midrule\n'
                                      + '\n'.join(body) + '\n\\bottomrule\n\\end{tabular}\n')


def write(name, text):
    with open(os.path.join(GEN, name), 'w') as f: f.write(text)


def summary_table(r):
    rows = []
    for _, x in r.iterrows():
        d = '---' if x.policy == 'TIDE' else neg(f'{x.tide_minus:+.2f}') + f' [{neg(f"{x.tide_minus_lo:+.2f}")}, {neg(f"{x.tide_minus_hi:+.2f}")}]'
        norm = neg(f'{x.norm:.3f}') + ('' if x.norm_hi - x.norm_lo < 5e-4 else f' [{neg(f"{x.norm_lo:.3f}")}, {neg(f"{x.norm_hi:.3f}")}]')
        name = f'\\textbf{{{x.policy}}}' if x.policy == 'TIDE' else x.policy
        rows.append(f'{name} & {norm} & {x.hit:.2f} & {x.wins} & {d} \\\\')
    return tex_table('lrrrr', r'Policy & Normalised [95\% CI] & Hit \% & Wins & TIDE $-$ policy, points [95\% CI]', rows)


def main():
    os.makedirs(GEN, exist_ok=True)
    m = {}
    res = {}
    for which in ('synth', 'real'):
        h = analysis.cells(analysis.load(which))
        s = analysis.per_size(h)
        r = analysis.ranking(h)
        res[which] = (h, s, r)
        W = which.capitalize()
        get = lambda p, col='norm': float(r.loc[r.policy == p, col].iloc[0])
        worst = s.loc[s.dARC.idxmin()]
        m.update({f'NGroups{W}': len(s), f'NCells{W}': len(h), f'NTraces{W}': s.trace.nunique(),
                  f'Norm{W}TIDE': f'{get("TIDE"):.3f}', f'Norm{W}ARC': f'{get("ARC"):.3f}', f'Norm{W}SthreeFIFO': f'{get("S3-FIFO"):.3f}',
                  f'Norm{W}LIRS': f'{get("LIRS"):.3f}', f'Norm{W}Cacheus': f'{get("Cacheus"):.3f}',
                  f'Wins{W}TIDE': int(get('TIDE', 'wins')), f'GeARC{W}': int((s.dARC >= 0).sum()),
                  f'Worst{W}': neg(f'{worst.dARC:+.2f}'), f'WorstCell{W}': f'{short(worst.trace)}, $c={worst.c}$',
                  f'Rank{W}TIDE': int(r.index[r.policy == 'TIDE'][0]) + 1})
        for p, tag in (('ARC', 'ARC'), ('S3-FIFO', 'Sthree'), ('LIRS', 'LIRS'), ('Cacheus', 'Cacheus'), ('LRB', 'LRB')):
            x = r[r.policy == p].iloc[0]
            m[f'D{W}{tag}'] = neg(f'{x.tide_minus:+.2f}')
            m[f'D{W}{tag}CI'] = f'[{neg(f"{x.tide_minus_lo:+.2f}")}, {neg(f"{x.tide_minus_hi:+.2f}")}]'
        write(f'tab_summary_{which}.tex', summary_table(r))
    h, s, r = res['synth']
    m['Seeds'] = int(analysis.load('synth').query('policy == "TIDE"').seed.nunique())
    m['Instances'] = int(h.inst.nunique())
    for t in TREND:
        m[f'Gain{t.capitalize()}'] = neg(f'{s[s.trace == t].dARC.max():+.1f}')
    m['AdvGain'] = neg(f'{s[s.trace == "adversarial"].dARC.max():+.1f}')

    # per-trace mean gain over ARC, with CI over (instance, size) means -> bar chart data
    rows = []
    for t, g in h.groupby('trace', sort=False):
        d = [sum(v) / len(v) - a for v, a in zip(g.tide_seeds, g.ARC)]
        rows.append((short(t).replace('\\_', '-').replace(' ', '-'), sum(d) / len(d), analysis.tci(d)))
    rows.sort(key=lambda x: x[1])
    write('gain.dat', 'trace gain ci\n' + ''.join(f'{t} {v:.3f} {c:.3f}\n' for t, v, c in rows))

    # full synthetic table, grouped per (trace, size)
    head = ['LRU', 'ARC', 'S3-FIFO', 'LIRS', 'Cacheus', 'LRB', 'TIDE']
    body, prev = [], None
    for _, x in s.iterrows():
        if prev is not None and x.trace != prev: body.append(r'\midrule')
        best = max(x[p] for p in head)
        cells = [short(x.trace) if x.trace != prev else '', f'{100 * x.frac:g}', str(x.c)]
        cells += [(rf'\textbf{{{x[p]:.2f}}}' if abs(x[p] - best) < 1e-9 else f'{x[p]:.2f}') for p in head[:-1]]
        tv = pm(x.TIDE, x.TIDE_ci)
        cells += [rf'\textbf{{{tv}}}' if abs(x.TIDE - best) < 1e-9 else tv, f"{x['TIDE-oracle']:.2f}", f'{x.Belady:.2f}',
                  neg(f'{x.dARC:+.2f}')]
        body.append(' & '.join(cells) + r' \\'); prev = x.trace
    write('tab_results.tex', tex_table('lrr' + 'r' * 10, r'Trace & Size\% & $c$ & ' + ' & '.join(head) + r' & Oracle & Belady & $\Delta$ARC', body))

    # real traces, one row per size
    hr, sr, rr = res['real']
    real = analysis.load('real')
    act = real[real.policy == 'TIDE'].groupby(['trace', 'c']).active_pct.mean()
    head = ['LRU', 'ARC', 'S3-FIFO', 'LIRS', 'Cacheus', 'LRB', 'TIDE']
    body, prev = [], None
    for _, x in sr.iterrows():
        if prev is not None and x.trace != prev: body.append(r'\midrule')
        best = max(x[p] for p in head)
        cells = [short(x.trace) if x.trace != prev else '', f'{100 * x.frac:g}', f'{x.c:,}']
        cells += [(rf'\textbf{{{x[p]:.2f}}}' if abs(x[p] - best) < 1e-9 else f'{x[p]:.2f}') for p in head]
        cells += [f"{x['TIDE-oracle']:.2f}", f'{x.Belady:.2f}', neg(f'{x.dARC:+.2f}'), f'{act[(x.trace, x.c)]:.0f}']
        body.append(' & '.join(cells) + r' \\'); prev = x.trace
    write('tab_real.tex', tex_table('lrr' + 'r' * 11, r'Trace & Size\% & $c$ & ' + ' & '.join(head) +
                                    r' & Oracle & Belady & $\Delta$ARC & Active\%', body))
    fp = real.drop_duplicates('trace').set_index('trace')
    m['RealN'] = f"{int(fp.n.iloc[0]):,}".replace(',', '{,}')
    m['OracleGapReal'] = f"{(sr['TIDE-oracle'] - sr.ARC).mean():.1f}"
    m['ActiveRealMin'], m['ActiveRealMax'] = f'{act.min():.0f}', f'{act.max():.0f}'
    write('tab_realtraces.tex', tex_table('llrr', r'Trace & Domain & Requests & Distinct keys', [
        f'{short(t)} & {d} & {int(fp.n[t]):,} & {int(fp.footprint[t]):,} \\\\'
        for t, (_, d) in analysis_real_domains().items()]))

    # speed
    sp = pd.read_csv(os.path.join(ROOT, 'bench_speed.csv'))
    lab = {'ARC': 'ARC', 'TIDE-ref': 'TIDE, NumPy', 'TIDE-fast': 'TIDE, Numba sync', 'TIDE-async': 'TIDE, Numba async'}
    body, prev = [], None
    for _, x in sp.iterrows():
        if prev is not None and x.trace != prev: body.append(r'\midrule')
        body.append(f"{short(x.trace) if x.trace != prev else ''} & {lab[x.policy]} & {x.req_per_s / 1e3:,.0f}k & "
                    f'{x.p50_us:.2f} & {x.p99_us:.1f} & {x.p999_us:.1f} \\\\'); prev = x.trace
    write('tab_speed.tex', tex_table('llrrrr', r'Trace & Implementation & Req/s & p50 ($\mu$s) & p99 ($\mu$s) & p99.9 ($\mu$s)', body))
    g = sp.set_index(['trace', 'policy'])
    rng = lambda pol, col: (g.xs(pol, level='policy')[col].min(), g.xs(pol, level='policy')[col].max())
    m['PNineNineRef'] = '{:.0f}--{:.0f}'.format(*rng('TIDE-ref', 'p99_us'))
    m['PNineNineSync'] = '{:.0f}--{:.0f}'.format(*rng('TIDE-fast', 'p99_us'))
    m['PNineNineAsync'] = '{:.0f}--{:.0f}'.format(*rng('TIDE-async', 'p99_us'))
    m['PFiftyAsync'] = '{:.1f}--{:.1f}'.format(*rng('TIDE-async', 'p50_us'))
    m['PFiftyARC'] = '{:.2f}--{:.2f}'.format(*rng('ARC', 'p50_us'))
    sx = (g.xs('TIDE-async', level='policy').req_per_s / g.xs('TIDE-ref', level='policy').req_per_s)
    m['SpeedupAsync'] = f'{sx.min():.1f}--{sx.max():.1f}'
    sa = (g.xs('ARC', level='policy').req_per_s / g.xs('TIDE-async', level='policy').req_per_s)
    m['SlowdownAsync'] = f'{sa.min():.0f}--{sa.max():.0f}'
    m['TrainPctSync'] = '{:.0f}--{:.0f}'.format(*rng('TIDE-fast', 'train_pct'))

    # async vs sync hit rate
    a = pd.read_csv(os.path.join(ROOT, 'bench_async.csv'))
    t = a.pivot_table(index='trace', columns='mode', values=['hit', 'steps'], aggfunc='mean')
    body = [f"{short(tr)} & {t.loc[tr, ('hit', 'sync')]:.2f} & {t.loc[tr, ('hit', 'async')]:.2f} & "
            f"{neg(f'{t.loc[tr, ('hit', 'async')] - t.loc[tr, ('hit', 'sync')]:+.2f}')} & "
            f"{100 * t.loc[tr, ('steps', 'async')] / t.loc[tr, ('steps', 'sync')]:.0f} \\\\" for tr in t.index]
    write('tab_async.tex', tex_table('lrrrr', r'Trace & Sync hit \% & Async hit \% & Change & Async steps (\% of sync)', body))
    dd = t[('hit', 'async')] - t[('hit', 'sync')]
    m['AsyncWorst'] = neg(f'{dd.min():+.2f}'); m['AsyncWorstTrace'] = short(dd.idxmin())
    m['AsyncOthers'] = f'{dd.drop(dd.idxmin()).abs().max():.2f}'

    write('macros.tex', ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in m.items()))

    # report-only plot data: hit rate vs size and learning curves on the trend traces (instance 0)
    h0 = h[h.inst == 0]
    for tr in TREND:
        x = h0[h0.trace == tr].sort_values('frac')
        cols = ['LRU', 'ARC', 'S3-FIFO', 'LIRS', 'Cacheus', 'TIDE', 'Belady']
        write(f'size_{tr}.dat', 'frac ' + ' '.join(p.replace('-', '') for p in cols) + '\n' + ''.join(
            f'{100 * r.frac:g} ' + ' '.join(f'{r[p]:.3f}' for p in cols) + '\n' for _, r in x.iterrows()))
    curves = pd.read_csv(os.path.join(ROOT, 'bench_curves.csv'))
    caps = ''
    for tr in TREND:
        x = s[s.trace == tr].sort_values('dARC').iloc[-1]
        cv = curves[(curves.trace == tr) & (curves.c == x.c)].pivot(index='window', columns='policy', values='hit')
        write(f'curve_{tr}.dat', 'pct ARC TIDE\n' + ''.join(f'{w + 0.5:g} {q.ARC:.3f} {q.TIDE:.3f}\n' for w, q in cv.iterrows()))
        caps += f'\\newcommand{{\\Curve{tr.capitalize()}}}{{{x.c}}}\n'
    write('curve_caps.tex', caps)

    subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'report.tex'], cwd=HERE, check=True,
                   capture_output=True)
    print('wrote', os.path.join(HERE, 'report.pdf'))


def analysis_real_domains():
    import realtraces
    return realtraces.REAL


if __name__ == '__main__':
    main()
