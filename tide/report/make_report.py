"""Build the results PDF from bench_results.csv, bench_curves.csv and bench_speed.csv.

  python3 report/make_report.py      # writes report/gen/* and compiles report/report.pdf (needs latexmk + pgfplots)
"""
import os
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEN = os.path.join(HERE, 'gen')
TREND = ['lifecycle', 'cycle', 'phases']
HEAD = ['LRU', 'LFU', 'ARC', 'S3-FIFO', 'SIEVE', 'TIDE']
SHORT = {'zipf_100k': 'zipf-100k', 'web_traffic_100k': 'web', 'adversarial': 'adversarial', 'bursty': 'bursty',
         'gaussian': 'gaussian', 'periodic': 'periodic', 'uniform': 'uniform', 'lifecycle': 'lifecycle',
         'cycle': 'cycle', 'phases': 'phases'}
short = lambda t: SHORT.get(t, t.replace('zipf_alpha_', 'zipf-a'))
tex = lambda t: short(t).replace('_', r'\_')
neg = lambda x: x.replace('-', '$-$')  # typographic minus in tables


def write(name, text):
    with open(os.path.join(GEN, name), 'w') as f: f.write(text)


def main():
    os.makedirs(GEN, exist_ok=True)
    df = pd.read_csv(os.path.join(ROOT, 'bench_results.csv'))
    curves = pd.read_csv(os.path.join(ROOT, 'bench_curves.csv'))
    speed = pd.read_csv(os.path.join(ROOT, 'bench_speed.csv'))
    seeds = df[df.policy == 'TIDE'].seed.nunique()
    h = df.groupby(['trace', 'frac', 'c', 'policy'])['hit'].mean().unstack('policy').reset_index()
    order = TREND + [t for t in dict.fromkeys(df.trace) if t not in TREND]
    h['rank'] = h.trace.map(order.index)
    h = h.sort_values(['rank', 'frac']).drop(columns='rank')
    h['dARC'] = h.TIDE - h.ARC
    d = h.pivot_table(index='trace', columns='frac', values='dARC').reindex(order)
    h = h.drop_duplicates(['trace', 'c'])  # tiny traces round several sizes to the same c: count that cell once

    # --- summary numbers ---
    ok = h[h.Belady - h.LRU > 0.5]
    norm = {p: ((ok[p] - ok.LRU) / (ok.Belady - ok.LRU)).mean() for p in HEAD}
    wins = h[HEAD].idxmax(axis=1).value_counts().reindex(HEAD, fill_value=0)
    worst = h.loc[h.dARC.idxmin()]
    tr = h[h.trace.isin(TREND)]
    gap = tr[tr['TIDE-oracle'] - tr.ARC > 0.5]
    gap_share = ((gap.TIDE - gap.ARC) / (gap['TIDE-oracle'] - gap.ARC)).mean()
    best_trend = tr.groupby('trace').dARC.max()
    m = {
        'NCells': len(h), 'NNorm': len(ok), 'Seeds': seeds, 'NTraces': h.trace.nunique(),
        'NormTIDE': f"{norm['TIDE']:.3f}", 'NormARC': f"{norm['ARC']:.3f}", 'NormSthree': f"{norm['S3-FIFO']:.3f}",
        'NormLFU': f"${norm['LFU']:.3f}$", 'WinsTIDE': wins['TIDE'], 'GeARC': int((h.dARC >= 0).sum()),
        'Worst': f"${worst.dARC:+.2f}$", 'WorstCell': f"{tex(worst.trace)}, $c={worst.c}$",
        'GapShare': f"{100 * gap_share:.0f}", 'AdvGain': f"{h[h.trace == 'adversarial'].dARC.max():+.1f}",
        **{f'Gain{t.capitalize()}': f'{best_trend[t]:+.1f}' for t in TREND},
        'PFifty': f'{speed.p50_us.median():.0f}', 'PNineNineMax': f'{speed.p99_us.max():.0f}',
        'RpsMin': f'{speed.tide_req_per_s.min() / 1e3:.0f}', 'RpsMax': f'{speed.tide_req_per_s.max() / 1e3:.0f}',
        'TrainPct': f'{speed.train_pct.mean():.0f}',
    }
    write('macros.tex', ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in m.items()))

    # --- plot data ---
    pols = sorted(HEAD, key=lambda p: -norm[p])
    write('norm.dat', 'policy norm\n' + ''.join(f'{p} {norm[p]:.4f}\n' for p in pols))
    CLIP = 12
    write('delta.dat', 'trace s1 s2 s3\n' + ''.join(
        f'{short(t)} ' + ' '.join(f'{min(v, CLIP):.3f}' for v in row) + '\n' for t, row in d.iterrows()))
    notes = []
    for k, (t, row) in enumerate(d.iterrows()):
        groups = {}  # equal clipped values share one label, centered over their bars
        for i, v in enumerate(row):
            if v > CLIP: groups.setdefault(f'{v:+.0f}', []).append((i - 1) * 3)
        notes += [rf'\node[above,font=\tiny,inner sep=1pt] at ([xshift={sum(x) / len(x):g}pt]axis cs:{k},{CLIP}) {{{v}}};'
                  for v, x in groups.items()]
    write('delta_notes.tex', '\n'.join(notes))
    for t in TREND:
        s = h[h.trace == t]
        write(f'size_{t}.dat', 'frac ' + ' '.join(p.replace('-', '') for p in HEAD + ['Belady']) + '\n' + ''.join(
            f'{100 * r.frac:g} ' + ' '.join(f'{r[p]:.3f}' for p in HEAD + ['Belady']) + '\n' for _, r in s.iterrows()))
    picks = {}
    for t in TREND:  # the size where TIDE gains most over ARC
        c = int(h[h.trace == t].sort_values('dARC').iloc[-1].c)
        cv = curves[(curves.trace == t) & (curves.c == c)].pivot(index='window', columns='policy', values='hit')
        write(f'curve_{t}.dat', 'pct ARC TIDE\n' + ''.join(f'{w + 0.5:g} {r.ARC:.3f} {r.TIDE:.3f}\n' for w, r in cv.iterrows()))
        picks[t] = c
    write('curve_caps.tex', ''.join(f'\\newcommand{{\\Curve{t.capitalize()}}}{{{picks[t]}}}\n' for t in TREND))

    # --- tables ---
    rows = []
    for i, (_, r) in enumerate(h.iterrows()):
        best = max(r[p] for p in HEAD)
        first = i == 0 or h.iloc[i - 1].trace != r.trace
        if first and i: rows.append(r'\midrule')
        cells = [tex(r.trace) if first else '', f'{100 * r.frac:g}', str(r.c)]
        cells += [(rf'\textbf{{{r[p]:.2f}}}' if r[p] == best else f'{r[p]:.2f}') for p in HEAD]
        cells += [f"{r['TIDE-oracle']:.2f}", f'{r.Belady:.2f}', neg(f'{r.dARC:+.2f}')]
        rows.append(' & '.join(cells) + r' \\')
    table = lambda spec, head, body: (f'\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head} \\\\\n\\midrule\n'
                                      + '\n'.join(body) + '\n\\bottomrule\n\\end{tabular}\n')
    write('tab_results.tex', table('lrrrrrrrrrrr', r'Trace & Size\% & $c$ & LRU & LFU & ARC & S3-FIFO & SIEVE & TIDE & Oracle & Belady & $\Delta$ARC', rows))
    write('tab_summary.tex', table('lrrr', r'Policy & Norm. & Wins & Mean hit\%',
                                   [f'{p} & {neg(f"{norm[p]:.3f}")} & {wins[p]} & {h[p].mean():.2f} \\\\' for p in pols]))
    write('tab_speed.tex', table('lrrrrrrr', r'Trace & Requests & $c$ & ARC req/s & TIDE req/s & p50 (\textmu s) & p99 (\textmu s) & Training \%', [
        f'{tex(r.trace)} & {r.n / 1e6:g}M & {r.c} & {r.arc_req_per_s / 1e3:,.0f}k & {r.tide_req_per_s / 1e3:.0f}k & '
        f'{r.p50_us:.1f} & {r.p99_us:.1f} & {r.train_pct:.0f} \\\\' for _, r in speed.iterrows()]))

    subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', '-quiet', 'report.tex'], cwd=HERE, check=True)
    subprocess.run(['latexmk', '-c', 'report.tex'], cwd=HERE, check=True, capture_output=True)
    print('wrote', os.path.join(HERE, 'report.pdf'))


if __name__ == '__main__':
    main()
