"""TABLES.md: Peekaboo against Vega's and Laya's own results, on their own benchmarks.

  1. Vega's README benchmark table (43 public tasks, JevBench, DecisionBench) with base Vega and Vega + Peekaboo
     for 0.8B and 4B. DecisionBench comes from the finished runs in experiments/vega_policy/results*/compare.
  2. Laya's README benchmark (typed-decisions): base Laya, Laya's own answer-key fine-tune, Laya + Peekaboo,
     and Vega with VegaML's TTT head, Vega's answer-key adapter and Peekaboo on the same rows.
  3. The 43 public tasks with VegaML's own adaptation (TTT) and the answer-key adapter next to Peekaboo.
  4. One line per comparison: Peekaboo against the base model and against their improvement, paired 95% CIs.
Metrics are computed with VegaML's own benchmark code (via experiments/vega_policy/table.py). Reads/writes VEGA_OUT.
"""
import glob, importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
VP = os.path.join(HERE, '..', 'vega_policy')
sys.path.insert(0, VP)
import numpy as np

# experiments/vega_policy/table.py (same file name as this one, so loaded by path)
_spec = importlib.util.spec_from_file_location('vega_table', os.path.join(VP, 'table.py'))
VT = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(VT)
ROOT = VT.ROOT
RES = {'0.8b': 'results', '4b': 'results_4b', 'laya': 'results_laya', 'laya_td': 'results_laya_td'}


def runs(model, suite, method):
    res = os.path.join(ROOT, RES[model])
    if method == 'released':
        f = os.path.join(res, f'released_{suite}.json')
        its = VT.items(suite)
        if not os.path.exists(f) or not its:
            return []
        recs = json.load(open(f))
        return [[recs[it['id']] for it in its]] if all(it['id'] in recs for it in its) else []
    return [json.load(open(f))['records'] for f in sorted(glob.glob(os.path.join(res, 'methods', suite, f'{method}_s[0-9].json')))]


def summary(model, suite, method):
    vals = {}
    for recs in runs(model, suite, method):
        for k, x in VT.metrics(suite, recs).items():
            vals.setdefault(k, []).append(x)
    return vals


def db_from_repo(size, method):
    """DecisionBench from the finished comparison (experiments/vega_policy/results*/compare): 'zero' or 'feedback'."""
    d = os.path.join(VP, RES[size], 'compare')
    vals = {}
    for sd in range(3 if method == 'feedback' else 1):
        f = next((p for p in (os.path.join(d, f'{method}_s{sd}_v2.json'), os.path.join(d, f'{method}_s{sd}.json')) if os.path.exists(p)), None)
        if f is None:
            continue
        t = json.load(open(f))['test']
        for k, x in {'db:accuracy': t['acc'], 'db:ece': t.get('ece15'), 'db:nll': t['nll'], 'db:kl': t['kl_soft'], 'db:coverage': 1.0,
                     'db:privacy': t['by_domain'].get('privacy'), 'db:public_sector': t['by_domain'].get('public_sector')}.items():
            vals.setdefault(k, []).append(x)
    return vals


def mean(v):
    return None if not v or any(x is None for x in v) else float(np.mean(v))


def cell(v, key='x'):
    m = mean(v)
    if m is None:
        return 'not run'
    return VT.fmt(m, key) + (f' ± {np.std(v):.3f}' if len(v) > 1 and not key.endswith('p50') else '')


def paired(suite, a_runs, b_runs, reps=2000):
    """Accuracy of b minus a on the same rows, each averaged over its seeds (points, bootstrap 95% CI). The public menu
    is a macro average over tasks, so it resamples rows within each task."""
    if not a_runs or not b_runs:
        return 'not run'
    its = VT.items(suite); rng = np.random.default_rng(0)
    c = lambda rs: np.mean([[float(r.get('pred') == r['gold']) for r in recs] for recs in rs], 0)
    d = c(b_runs) - c(a_runs)
    groups = ([np.array([i for i, it in enumerate(its) if it['task'] == t]) for t in sorted({it['task'] for it in its})]
              if suite == 'public' else [np.arange(len(its))])
    stat = 100 * np.mean([d[g].mean() for g in groups])
    bs = [100 * np.mean([d[rng.choice(g, len(g))].mean() for g in groups]) for _ in range(reps)]
    return f'{stat:+.1f} [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]'


def bold(vals, txt, hib):
    ok = [x for x in vals if x is not None]
    best = (max if hib else min)(ok) if ok else None
    return [f'**{t}**' if x is not None and best is not None and abs(x - best) < 5e-4 else t for x, t in zip(vals, txt)]


def main():
    L = ['# Peekaboo against Vega and Laya, on their own benchmarks', '',
         '"base" = the model as released, measured in this run. "+ Peekaboo" = the same model plus our adapter, trained '
         'only from right / wrong feedback (mean ± s.d. over 3 seeds). "README" = the number Vega or Laya publish. '
         'Bold = best in the row.', '']

    # 1. Vega's README table
    S = {(m, s, meth): summary(m, s, meth) for m in ('0.8b', '4b') for s in ('public', 'jevbench') for meth in ('released', 'feedback', 'ttt', 'sup_ts')}
    for m in ('0.8b', '4b'):
        S[(m, 'decisionbench', 'released')] = db_from_repo(m, 'zero'); S[(m, 'decisionbench', 'feedback')] = db_from_repo(m, 'feedback')
    val = lambda m, meth, key: next((S[(m, s, meth)][key] for s in ('public', 'jevbench', 'decisionbench') if key in S[(m, s, meth)]), None)
    L += ['## 1. Vega\'s README benchmark table, with Peekaboo', '']
    head = ['benchmark', 'Vega 0.8B (README)', 'base Vega 0.8B', '**Vega 0.8B + Peekaboo**', 'Vega 4B (README)', 'base Vega 4B',
            '**Vega 4B + Peekaboo**', 'Jev (README)']
    L += ['| ' + ' | '.join(head) + ' |', '|---' * len(head) + '|']
    for row in VT.ROWS:
        if row[0] == 'h':
            L.append(f'| **{row[1]}** |' + ' |' * (len(head) - 1)); continue
        key, name, hib, r08, r4, rj = row
        cols = [[r08], val('0.8b', 'released', key), val('0.8b', 'feedback', key), [r4], val('4b', 'released', key), val('4b', 'feedback', key), [rj]]
        vals = [mean(v) for v in cols]
        txt = [VT.fmt(c[0], key) if i in (0, 3, 6) else cell(c, key) for i, c in enumerate(cols)]
        L.append(f'| {name} | ' + ' | '.join(bold(vals, txt, hib)) + ' |')
    L += ['', 'DecisionBench rows: the finished runs in experiments/vega_policy/results*/compare (latency there is the README\'s).', '']

    # 2. Laya's README benchmark: typed-decisions
    T = [('laya', 'released', 'base Laya', 'none', 0.362), ('laya_td', 'released', "laya-typed-decisions (Laya's own fine-tune)", 'answer key', 0.766),
         ('laya', 'feedback', '**Laya + Peekaboo**', 'right / wrong', None)]
    for m, nm in (('0.8b', 'Vega 0.8B'), ('4b', 'Vega 4B')):
        T += [(m, 'released', f'base {nm}', 'none', None), (m, 'ttt', f"{nm} + VegaML's TTT head", 'answer key', None),
              (m, 'sup_ts', f'{nm} + answer-key adapter', 'answer key', None), (m, 'feedback', f'**{nm} + Peekaboo**', 'right / wrong', None)]
    L += ['## 2. Laya\'s README benchmark: typed-decisions (2,000 test decisions; feedback / labels from the 6,000 training decisions)', '',
          '| model | learns from | accuracy (README) | accuracy | soft accuracy | ECE | NLL | gain vs base |', '|---|---|---|---|---|---|---|---|']
    for m, meth, name, src, readme in T:
        v = summary(m, 'typed', meth)
        base = runs('laya' if m.startswith('laya') else m, 'typed', 'released')
        g = '—' if meth == 'released' and m != 'laya_td' else paired('typed', base, runs(m, 'typed', meth))
        L.append(f'| {name} | {src} | {"" if readme is None else f"{readme:.3f}"} | {cell(v.get("td:accuracy"))} | {cell(v.get("td:soft_accuracy"))} | '
                 f'{cell(v.get("td:ece"))} | {cell(v.get("td:nll"))} | {g} |')
    L += ['| Jev 1.13.0 (dataset card) | none | 0.727 | | | | | |', '',
          'Vega\'s README also reports typed-decisions (0.763 / 0.803 with its task adapter) on six other workflows whose rows '
          'are not published, so it cannot be rerun; the rows above are the public benchmark Laya reports.', '']

    # 3. public tasks, every way to improve Vega
    L += ['## 3. The 43 public tasks: Peekaboo against VegaML\'s own ways to improve Vega', '',
          '| model | method | learns from | accuracy | skill | ECE | NLL | gain vs base |', '|---|---|---|---|---|---|---|---|']
    for m, nm in (('0.8b', 'Vega 0.8B'), ('4b', 'Vega 4B')):
        for meth, name, src in (('released', 'base', 'none'), ('ttt', "VegaML's TTT head (the notebook's 50 labelled rows per task)", 'answer key'),
                                ('sup_ts', 'answer-key adapter', 'answer key'), ('feedback', '**Peekaboo**', 'right / wrong')):
            v = S[(m, 'public', meth)]
            g = '—' if meth == 'released' else paired('public', runs(m, 'public', 'released'), runs(m, 'public', meth))
            L.append(f'| {nm} | {name} | {src} | {cell(v.get("pub:accuracy"))} | {cell(v.get("pub:skill"))} | {cell(v.get("pub:ece"))} | '
                     f'{cell(v.get("pub:nll"))} | {g} |')
    L.append('')

    # 4. the head-to-head lines
    L += ['## 4. Peekaboo head to head (accuracy points, paired 95% CI; positive = Peekaboo better)', '',
          '| benchmark | model | Peekaboo vs base model | Peekaboo vs their improvement (answer key) |', '|---|---|---|---|']
    for m, nm in (('0.8b', 'Vega 0.8B'), ('4b', 'Vega 4B')):
        for s in ('public', 'typed'):
            pk = runs(m, s, 'feedback')
            L.append(f'| {s} | {nm} | {paired(s, runs(m, s, "released"), pk)} | TTT: {paired(s, runs(m, s, "ttt"), pk)} · '
                     f'adapter: {paired(s, runs(m, s, "sup_ts"), pk)} |')
        L.append(f'| jevbench | {nm} | {paired("jevbench", runs(m, "jevbench", "released"), runs(m, "jevbench", "feedback"))} | — (Vega has none) |')
    pk = runs('laya', 'typed', 'feedback')
    L += [f'| typed | Laya | {paired("typed", runs("laya", "typed", "released"), pk)} | laya-typed-decisions: '
          f'{paired("typed", runs("laya_td", "typed", "released"), pk)} |', '',
          'DecisionBench (Vega 0.8B / 4B) is in the paper: +10.2 / +6.0 over base, +8.0 / +5.8 over TTT, -1.6 / +1.2 against the '
          'answer-key adapter.', '']

    # 5. checks
    L += ['## 5. Checks', '', '| model | suite | replay vs official answers, max abs. probability difference | rows the model could not answer |', '|---|---|---|---|']
    for m in ('0.8b', '4b', 'laya', 'laya_td'):
        for s in (('typed',) if m.startswith('laya') else ('public', 'jevbench', 'typed')):
            f = os.path.join(ROOT, RES[m], 'methods', s, 'replay_s0.json'); rel = os.path.join(ROOT, RES[m], f'released_{s}.json')
            d = json.load(open(f))['info'].get('max_abs_dp') if os.path.exists(f) else None
            errs = sum('error' in r for r in json.load(open(rel)).values()) if os.path.exists(rel) else None
            L.append(f'| {m} | {s} | {"—" if d is None else f"{d:.1e}"} | {"not run" if errs is None else errs} |')
    text = '\n'.join(L) + '\n'
    open(os.path.join(ROOT, 'TABLES.md'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
