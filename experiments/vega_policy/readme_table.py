"""The improvement table, in the layout of Vega's README: released Vega vs Vega + our feedback adapter, per size.

Reads results/compare (0.8B) and results_4b/compare (4B), whatever exists. A 4B column that has not been run yet
falls back to the README's published number, marked (README). Jev is always copied from the README.
ECE uses the README's estimator (15 equal-width bins). Gains are paired over the same 1,075 rows, bootstrap 95% CI.

    python readme_table.py            # prints markdown, also written to results/readme_table.md
"""
import glob, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = {'0.8B': os.path.join(HERE, 'results', 'compare'), '4B': os.path.join(HERE, 'results_4b', 'compare')}
# The README's numbers for the same rows (Vega 4B and Jev columns of its benchmark table).
README = {'4B': {'privacy': 1.000, 'public_sector': 0.960, 'acc': 0.633, 'ece15': 0.098, 'nll': 1.420, 'kl_soft': 0.854, 'answered': 1.000},
          'Jev': {'privacy': 0.960, 'public_sector': 0.880, 'acc': 0.752, 'ece15': 0.122, 'nll': 3.196, 'kl_soft': 6.099, 'answered': 0.997}}
ROWS = [('privacy', 'DecisionBench domain:privacy, 25 items', 'acc'), ('public_sector', 'DecisionBench domain:public_sector, 25 items', 'acc'),
        ('acc', 'DecisionBench, accuracy', 'acc'), ('ece15', 'DecisionBench, calibration error (ECE)', 'err'),
        ('nll', 'DecisionBench, negative log likelihood', 'err'), ('kl_soft', 'DecisionBench, KL to gold distribution (10 rows)', 'err'),
        ('answered', 'DecisionBench, rows answered', 'acc')]
rng = np.random.default_rng(0)


def ece15(conf, correct, bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float); e = np.linspace(0, 1, bins + 1); out = 0.0
    for i in range(bins):
        sel = (conf > e[i]) & (conf <= e[i + 1]) if i else (conf >= e[i]) & (conf <= e[i + 1])
        if sel.any():
            out += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(out)


def runs(d, method):
    """One result per seed. Prefers the file that kept per-row confidences (needed for the 15-bin ECE)."""
    by_seed = {}
    for f in sorted(glob.glob(os.path.join(d, f'{method}_s*.json'))):
        m = re.fullmatch(rf'{method}_s(\d+)(_v2)?\.json', os.path.basename(f))
        if not m:
            continue                                   # other tags (_kl1, ablations) are not this method
        t = json.load(open(f))['test']
        if 'ece15' not in t and 'conf' in t['rows']:
            t['ece15'] = ece15(t['rows']['conf'], t['rows']['correct'])
        if m.group(1) not in by_seed or ('ece15' in t and 'ece15' not in by_seed[m.group(1)]):
            by_seed[m.group(1)] = t
    out = []
    for t in by_seed.values():
        t = dict(t, **t['by_domain']); t['answered'] = len(t['rows']['correct']) / t['n']
        out.append(t)
    return out


def cell(rs, k):
    x = [r[k] for r in rs if k in r]
    if not x:
        return None, '—'
    return float(np.mean(x)), f'{np.mean(x):.3f}' + (f' ± {np.std(x):.3f}' if len(x) > 1 and np.std(x) > 0 else '')


def gain(base, ours):
    d = np.mean([r['rows']['correct'] for r in ours], 0) - np.array(base[0]['rows']['correct'])
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    return f'{d.mean() * 100:+.1f} [{np.percentile(bs, 2.5) * 100:+.1f}, {np.percentile(bs, 97.5) * 100:+.1f}]'


def main():
    cols, gains = [], []          # (header, {row: (value, text)})
    for size, d in SIZES.items():
        zero, ours = runs(d, 'zero'), runs(d, 'feedback')
        if zero:
            cols.append((f'Vega {size}', {k: cell(zero, k) for k, _, _ in ROWS}))
        elif size in README:
            cols.append((f'Vega {size} (README)', {k: (v, f'{v:.3f}') for k, v in README[size].items()}))
        cols.append((f'**{size} + ours**', {k: cell(ours, k) for k, _, _ in ROWS} if ours else {}))
        if zero and ours:
            gains.append(f'{size}: {gain(zero, ours)} points accuracy, {len(ours)} seeds')
    cols.append(('Jev (README)', {k: (v, f'{v:.3f}') for k, v in README['Jev'].items()}))
    lines = ['| benchmark | ' + ' | '.join(h for h, _ in cols) + ' |', '|---|' + '---|' * len(cols)]
    for k, name, kind in ROWS:
        vals = [c.get(k, (None, 'not run'))[0] for _, c in cols]
        best = (max if kind == 'acc' else min)([v for v in vals if v is not None])
        txt = [c.get(k, (None, 'not run'))[1] for _, c in cols]
        txt = [f'**{t}**' if v is not None and abs(v - best) < 5e-4 else t for v, t in zip(vals, txt)]
        lines.append(f'| {name} | ' + ' | '.join(txt) + ' |')
    lines += ['', 'Paired accuracy gain of ours over the released model, same 1,075 rows, bootstrap 95% CI:'] + [f'- {g}' for g in gains]
    out = '\n'.join(lines)
    print(out)
    open(os.path.join(HERE, 'results', 'readme_table.md'), 'w').write(out + '\n')


if __name__ == '__main__':
    main()
