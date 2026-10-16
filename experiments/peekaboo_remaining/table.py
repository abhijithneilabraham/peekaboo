"""TABLES.md for the remaining experiments: base model vs + Peekaboo for Vega 0.8B, Vega 4B and Laya, in the layout
of VegaML's README, plus every method per suite and the realistic-feedback table. Metrics are computed with VegaML's
own benchmark code (via experiments/vega_policy/table.py). Reads and writes VEGA_OUT.
"""
import glob, importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
VP = os.path.join(HERE, '..', 'vega_policy')
sys.path.insert(0, VP)
import numpy as np

# experiments/vega_policy/table.py (same file name as this one, so loaded by path): VegaML-harness metrics,
# the README rows, the paired bootstrap
_spec = importlib.util.spec_from_file_location('vega_table', os.path.join(VP, 'table.py'))
VT = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(VT)

ROOT = VT.ROOT
MODELS = [('0.8b', 'Vega 0.8B', 'results'), ('4b', 'Vega 4B', 'results_4b'), ('laya', 'Laya', 'results_laya')]
SUITES = ('public', 'jevbench', 'decisionbench')
PRE = {'public': 'pub', 'jevbench': 'jev', 'decisionbench': 'db'}
REALISTIC = [('feedback', 'clean feedback (as in the paper)'), ('feedback_noise10', '10% of reports flipped'),
             ('feedback_noise20', '20% flipped'), ('feedback_noise30', '30% flipped'), ('feedback_half', 'only half the answers rated'),
             ('feedback_quarter', 'only a quarter rated'), ('feedback_complaints', 'complaints only (all wrong, 20% of right reported)'),
             ('logged_ips', 'logged-feedback baseline (off-policy)')]
METHODS = {'0.8b': [('released', 'base Vega 0.8B', 'none'), ('ttt', "VegaML's TTT head", 'answer key'), ('sup_ts', 'answer-key adapter', 'answer key'),
                    ('linear_fb', 'no-engine control', 'right/wrong'), ('feedback_kl1', 'Peekaboo, KL 1', 'right/wrong'), ('feedback', '**Vega 0.8B + Peekaboo**', 'right/wrong')],
           'laya': [('released', 'base Laya', 'none'), ('rlcd', 'Laya + RLCD (its own fine-tuning)', 'answer key'), ('feedback', '**Laya + Peekaboo**', 'right/wrong')]}
METHODS['4b'] = [(m, n.replace('0.8B', '4B'), s) for m, n, s in METHODS['0.8b']]


def runs(model, suite, method):
    res = os.path.join(ROOT, dict((m, r) for m, _, r in MODELS)[model])
    if method == 'released':
        f = os.path.join(res, f'released_{suite}.json')
        if not os.path.exists(f):
            return []
        recs, its = json.load(open(f)), VT.items(suite)
        return [[recs[it['id']] for it in its]] if its and all(it['id'] in recs for it in its) else []
    return [json.load(open(f))['records'] for f in sorted(glob.glob(os.path.join(res, 'methods', suite, f'{method}_s[0-9].json')))]


def summary(model, suite, method):
    rs = runs(model, suite, method)
    vals = {}
    for recs in rs:
        for k, x in VT.metrics(suite, recs).items():
            vals.setdefault(k, []).append(x)
    return vals


def mean(v):
    return None if not v or any(x is None for x in v) else float(np.mean(v))


def cell(v, key):
    m = mean(v)
    if m is None:
        return 'not run'
    return VT.fmt(m, key) + (f' ± {np.std(v):.3f}' if len(v) > 1 and not key.endswith('p50') else '')


def gain(model, suite, method):
    base, ours = runs(model, suite, 'released'), runs(model, suite, method)
    if not base or not ours:
        return None
    g = VT.gain(suite, base, ours)
    return f'{g[0]:+.1f} [{g[1]:+.1f}, {g[2]:+.1f}]'


def main():
    S = {(m, s, meth): summary(m, s, meth) for m, _, _ in MODELS for s in SUITES
         for meth in {x for x, _, _ in METHODS[m]} | {x for x, _ in REALISTIC}}
    row_val = lambda m, meth, key: next((S[(m, s, meth)].get(key) for s in SUITES if key in S[(m, s, meth)]), None)
    L = ['# Peekaboo on Vega 0.8B, Vega 4B and Laya', '',
         'Base = the model as released, measured in this run. + Peekaboo = the same model plus our adapter, trained only from '
         'right / wrong feedback (mean ± s.d. over 3 seeds). README = VegaML\'s published number. Metrics from VegaML\'s own '
         'benchmark code. Bold = best in the row.', '', '## 1. The README benchmark table, with Peekaboo', '']
    head = ['benchmark', 'Vega 0.8B (README)', 'base Vega 0.8B', '**Vega 0.8B + Peekaboo**', 'Vega 4B (README)', 'base Vega 4B',
            '**Vega 4B + Peekaboo**', 'base Laya', '**Laya + Peekaboo**', 'Laya + RLCD', 'Jev (README)']
    L += ['| ' + ' | '.join(head) + ' |', '|---' * len(head) + '|']
    for row in VT.ROWS:
        if row[0] == 'h':
            L.append(f'| **{row[1]}** |' + ' |' * (len(head) - 1)); continue
        key, name, hib, r08, r4, rj = row
        cols = [(r08, [r08]), (None, row_val('0.8b', 'released', key)), (None, row_val('0.8b', 'feedback', key)),
                (r4, [r4]), (None, row_val('4b', 'released', key)), (None, row_val('4b', 'feedback', key)),
                (None, row_val('laya', 'released', key)), (None, row_val('laya', 'feedback', key)), (None, row_val('laya', 'rlcd', key)),
                (rj, [rj])]
        vals = [mean(v) for _, v in cols]
        txt = [VT.fmt(r, key) if r is not None else cell(v, key) for r, v in cols]
        ok = [x for x in vals if x is not None]
        best = (max if hib else min)(ok) if ok else None
        txt = [f'**{t}**' if x is not None and best is not None and abs(x - best) < 5e-4 else t for x, t in zip(vals, txt)]
        L.append(f'| {name} | ' + ' | '.join(txt) + ' |')

    L += ['', '## 2. Accuracy gain of Peekaboo over the base model, same rows (points, bootstrap 95% CI)', '',
          '| suite | Vega 0.8B | Vega 4B | Laya |', '|---|---|---|---|']
    for s in SUITES:
        L.append(f'| {s} | ' + ' | '.join(gain(m, s, 'feedback') or 'not run' for m, _, _ in MODELS) + ' |')
    L.append('\nDecisionBench for Vega with every baseline is in experiments/vega_policy/results*/compare (the paper\'s Section 5.1).')

    L += ['', '## 3. Every method, per suite (accuracy / ECE / NLL, and paired gain over the base model)', '']
    for s in SUITES:
        L += [f'### {s}', '', '| model | method | learns from | accuracy | ECE | NLL | gain vs base |', '|---|---|---|---|---|---|---|']
        for m, mname, _ in MODELS:
            for meth, name, src in METHODS[m]:
                v = S[(m, s, meth)]
                if not v:
                    continue
                n = lambda k: cell(v.get(f'{PRE[s]}:{k}'), k)
                L.append(f'| {mname} | {name} | {src} | {n("accuracy")} | {n("ece")} | {n("nll") if s != "jevbench" else "—"} | '
                         f'{"—" if meth == "released" else gain(m, s, meth) or "—"} |')
        L.append('')

    L += ['## 4. Peekaboo under realistic feedback (DecisionBench, 1,075 rows)', '',
          'The paper\'s feedback was perfect. Here it is noisy, missing, or biased the way real users are, and the last row '
          'learns from feedback logged on the base model\'s own answers instead of trying answers itself.', '',
          '| feedback | ' + ' | '.join(f'{n} accuracy | {n} ECE | {n} gain' for _, n, _ in MODELS) + ' |',
          '|---|' + '---|---|---|' * len(MODELS)]
    base_row = []
    for m, _, _ in MODELS:
        v = S[(m, 'decisionbench', 'released')]
        base_row += [cell(v.get('db:accuracy'), 'accuracy'), cell(v.get('db:ece'), 'ece'), '—']
    L.append('| base model, no feedback | ' + ' | '.join(base_row) + ' |')
    for meth, name in REALISTIC:
        row = []
        for m, _, _ in MODELS:
            v = S[(m, 'decisionbench', meth)]
            row += [cell(v.get('db:accuracy'), 'accuracy'), cell(v.get('db:ece'), 'ece'), gain(m, 'decisionbench', meth) or '—']
        L.append(f'| {name} | ' + ' | '.join(row) + ' |')

    L += ['', '## 5. Checks', '', '| model | suite | replay vs official answers, max abs. probability difference | rows the model could not answer |', '|---|---|---|---|']
    for m, mname, res in MODELS:
        for s in SUITES:
            f = os.path.join(ROOT, res, 'methods', s, 'replay_s0.json')
            rel = os.path.join(ROOT, res, f'released_{s}.json')
            d = json.load(open(f))['info'].get('max_abs_dp') if os.path.exists(f) else None
            errs = sum('error' in r for r in json.load(open(rel)).values()) if os.path.exists(rel) else None
            L.append(f'| {mname} | {s} | {"not run" if d is None else f"{d:.1e}"} | {"not run" if errs is None else errs} |')
    text = '\n'.join(L) + '\n'
    open(os.path.join(ROOT, 'TABLES.md'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
