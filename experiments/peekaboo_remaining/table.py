"""TABLES.md: Peekaboo against Vega's and Laya's own results, on their own benchmarks.

  1. Vega's README benchmark table (43 public tasks, JevBench, DecisionBench): base and + Peekaboo for Vega 0.8B,
     Vega 4B and Laya, plus laya-typed-decisions. Vega's DecisionBench comes from experiments/vega_policy/results*.
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


def runs(model, suite, method, key='records'):
    res = os.path.join(ROOT, RES[model])
    if method == 'released':
        f = os.path.join(res, f'released_{suite}.json')
        its = VT.items(suite)
        if not os.path.exists(f) or not its:
            return []
        recs = json.load(open(f))
        return [[recs[it['id']] for it in its]] if all(it['id'] in recs for it in its) else []
    out = []
    for f in sorted(glob.glob(os.path.join(res, 'methods', suite, f'{method}_s[0-9].json'))):
        d = json.load(open(f))
        if key in d:
            out.append(d[key])
    return out


def summary(model, suite, method, key='records'):
    vals = {}
    for recs in runs(model, suite, method, key):
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


# ───────────────────────────── the full grid ─────────────────────────────
SUITES = ('public', 'jevbench', 'decisionbench', 'typed', 'emotion', 'agnews')
PREFIX = {'public': 'pub', 'jevbench': 'jev', 'decisionbench': 'db', 'typed': 'td', 'emotion': 'em', 'agnews': 'ag'}
VARIANTS = [('released', 'records', 'base model', 'none'),
            ('noship_released', 'records', 'base model, shipped adapters off (typed-decisions)', 'none'),
            ('ttt', 'records', "VegaML's TTT head (vegaml 0.8.0 rule: head where it beats chance)", 'answer key'),
            ('ttt', 'records_guard', "VegaML's TTT head (vegaml 0.12.3 rule: head where it beats chance and the engine)", 'answer key'),
            ('sup_ts', 'records', 'answer-key adapter, same recipe (Laya: answer-key head)', 'answer key'),
            ('sup_ts', 'records_es', 'answer-key adapter, early-stopped on calibration labels', 'answer key'),
            ('sup_1pass', 'records', 'answer-key adapter, one pass', 'answer key'),
            ('laya_td', 'records', "laya-typed-decisions (Laya's released fine-tune, typed-decisions only)", 'answer key (typed-decisions)'),
            ('feedback', 'records', '**Peekaboo** (8 passes)', 'right / wrong'),
            ('feedback_1pass', 'records', '**Peekaboo, one pass** (each row rated once)', 'right / wrong'),
            ('feedback', 'records_switch', 'Peekaboo + switch (best answers compared)', 'right / wrong'),
            ('feedback', 'records_raw', 'Peekaboo without the honesty fix', 'right / wrong'),
            ('feedback_kl0', 'records', 'Peekaboo without the KL term', 'right / wrong'),
            ('logged_argmax', 'records', "logged feedback: the released model's best answers, rated once", 'right / wrong (logged)'),
            ('logged_ips', 'records', "logged feedback: the released model's sampled answers, rated once (clipped IPS)", 'right / wrong (logged)'),
            ('feedback_noise20', 'records', 'Peekaboo, 20% of ratings flipped', 'right / wrong (noisy)'),
            ('feedback_half', 'records', 'Peekaboo, half of the answers rated', 'right / wrong (missing)'),
            ('feedback_complaints', 'records', 'Peekaboo, every wrong answer reported, 20% of right ones', 'right / wrong (complaints)'),
            ('sup_ts_noship', 'records', 'answer-key adapter, shipped adapters off', 'answer key'),
            ('feedback_noship', 'records', 'Peekaboo, shipped adapters off', 'right / wrong'),
            ('feedback_1pass_noship', 'records', 'Peekaboo, one pass, shipped adapters off', 'right / wrong'),
            ('feedback_enc', 'records', '**Peekaboo, encoder unfrozen** (8 passes)', 'right / wrong'),
            ('feedback_enc_1pass', 'records', 'Peekaboo, encoder unfrozen, one pass', 'right / wrong'),
            ('sup_enc', 'records', 'answer-key fine-tune, encoder unfrozen, same recipe', 'answer key'),
            ('sup_enc', 'records_es', 'answer-key fine-tune, encoder unfrozen, early-stopped', 'answer key'),
            ('feedback_enc', 'records_switch', 'Peekaboo, encoder unfrozen + switch', 'right / wrong'),
            ('feedback_enc', 'records_raw', 'Peekaboo, encoder unfrozen, without the honesty fix', 'right / wrong'),
            ('pooled', 'records', '**pooled Peekaboo** (one add-on, all suites\' feedback)', 'right / wrong'),
            ('pooled', 'records_switch', 'pooled Peekaboo + switch', 'right / wrong'),
            ('pooled', 'records_raw', 'pooled Peekaboo without the honesty fix', 'right / wrong')]


def variant_runs(model, suite, method, key):
    if method == 'laya_td':
        return runs('laya_td', suite, 'released') if model == 'laya' else []
    return runs(model, suite, method, key)


def grid():
    L = ['## 0. Everything: every model, method and ablation on every benchmark', '',
         'Accuracy (mean ± s.d. over 3 seeds), then calibration error, then the paired accuracy gain over the base model in '
         'points with a 95% bootstrap CI. JevBench: per-suite Peekaboo learns by 5-fold cross-fitting; pooled Peekaboo never '
         'sees a JevBench row. "switch" = the add-on is used only where held-out right / wrong feedback says it beats the base model.', '']
    for title, what in (('accuracy', 'accuracy'), ('calibration error (ECE, lower is better)', 'ece'), ('gain over the base model', 'gain')):
        L += [f'### {title}', '', '| model | method | learns from | ' + ' | '.join(SUITES) + ' |', '|---|---|---|' + '---|' * len(SUITES)]
        for model, mname in (('0.8b', 'Vega 0.8B'), ('4b', 'Vega 4B'), ('laya', 'Laya')):
            for method, key, name, src in VARIANTS:
                if (model == 'laya' and method == 'ttt') or (model != 'laya' and method == 'laya_td'):
                    continue
                cells = []
                for s in SUITES:
                    rs = variant_runs(model, s, method, key)
                    if not rs:
                        cells.append(''); continue
                    if what == 'gain':
                        cells.append('—' if method == 'released' else paired(s, runs(model, s, 'released'), rs))
                    else:
                        vals = [VT.metrics(s, r).get(f'{PREFIX[s]}:{what}') for r in rs]
                        cells.append(cell(vals))
                if any(cells):
                    L.append(f'| {mname} | {name} | {src} | ' + ' | '.join(c or '·' for c in cells) + ' |')
        L.append('')
    return L


def main():
    L = ['# Peekaboo against Vega and Laya, on their own benchmarks', '',
         '"base" = the model as released, measured in this run. "+ Peekaboo" = the same model plus our adapter, trained '
         'only from right / wrong feedback (mean ± s.d. over 3 seeds). "README" = the number Vega or Laya publish. '
         'Bold = best in the row.', ''] + grid()

    # 1. Vega's README table
    S = {(m, s, meth): summary(m, s, meth) for m in ('0.8b', '4b') for s in ('public', 'jevbench') for meth in ('released', 'feedback', 'ttt', 'sup_ts')}
    for m in ('0.8b', '4b'):                   # this run's DecisionBench when it has it, else the finished runs in the repo
        here = summary(m, 'decisionbench', 'released')
        S[(m, 'decisionbench', 'released')] = here or db_from_repo(m, 'zero')
        S[(m, 'decisionbench', 'feedback')] = summary(m, 'decisionbench', 'feedback') if here else db_from_repo(m, 'feedback')
    for s_ in ('public', 'jevbench', 'decisionbench'):
        S[('laya', s_, 'released')] = summary('laya', s_, 'released'); S[('laya', s_, 'feedback')] = summary('laya', s_, 'feedback')
        S[('laya_td', s_, 'released')] = summary('laya_td', s_, 'released')
    val = lambda m, meth, key: next((S[(m, s, meth)][key] for s in ('public', 'jevbench', 'decisionbench') if key in S[(m, s, meth)]), None)
    L += ['## 1. Vega\'s README benchmark table, with Peekaboo', '']
    head = ['benchmark', 'Vega 0.8B (README)', 'base Vega 0.8B', '**Vega 0.8B + Peekaboo**', 'Vega 4B (README)', 'base Vega 4B',
            '**Vega 4B + Peekaboo**', 'base Laya', 'laya-typed-decisions', '**Laya + Peekaboo**', 'Jev (README)']
    L += ['| ' + ' | '.join(head) + ' |', '|---' * len(head) + '|']
    for row in VT.ROWS:
        if row[0] == 'h':
            L.append(f'| **{row[1]}** |' + ' |' * (len(head) - 1)); continue
        key, name, hib, r08, r4, rj = row
        cols = [[r08], val('0.8b', 'released', key), val('0.8b', 'feedback', key), [r4], val('4b', 'released', key), val('4b', 'feedback', key),
                val('laya', 'released', key), val('laya_td', 'released', key), val('laya', 'feedback', key), [rj]]
        vals = [mean(v) for v in cols]
        txt = [VT.fmt(c[0], key) if i in (0, 3, len(cols) - 1) else cell(c, key) for i, c in enumerate(cols)]
        L.append(f'| {name} | ' + ' | '.join(bold(vals, txt, hib)) + ' |')
    L += ['', 'Vega\'s DecisionBench rows: the finished runs in experiments/vega_policy/results*/compare (its latency is the README\'s). '
          'Laya is measured in this run on every row. laya-typed-decisions is Laya\'s released answer-key fine-tune (on typed-decisions).', '']

    # 2. Laya's README benchmark: typed-decisions
    T = [('laya', 'released', 'base Laya', 'none', 0.362), ('laya_td', 'released', "laya-typed-decisions (Laya's own fine-tune)", 'answer key', 0.766),
         ('laya', 'feedback', '**Laya + Peekaboo**', 'right / wrong', None),
         ('laya', 'feedback_enc', '**Laya + Peekaboo, encoder unfrozen**', 'right / wrong', None)]
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
    for s_ in ('public', 'jevbench', 'decisionbench', 'typed'):
        pk = runs('laya', s_, 'feedback')
        L.append(f'| {s_} | Laya | {paired(s_, runs("laya", s_, "released"), pk)} | laya-typed-decisions: '
                 f'{paired(s_, runs("laya_td", s_, "released"), pk)} |')
    for s_ in SUITES:
        pk = runs('laya', s_, 'feedback_enc')
        if pk:
            L.append(f'| {s_} | Laya, encoder unfrozen | {paired(s_, runs("laya", s_, "released"), pk)} | laya-typed-decisions: '
                     f'{paired(s_, runs("laya_td", s_, "released"), pk)} |')
    L += ['',
          'DecisionBench (Vega 0.8B / 4B) is in the paper: +10.2 / +6.0 over base, +8.0 / +5.8 over TTT, -1.6 / +1.2 against the '
          'answer-key adapter.', '']

    # 5. checks
    L += ['## 5. Checks', '', '| model | suite | replay vs official answers, max abs. probability difference | rows the model could not answer |', '|---|---|---|---|']
    for m in ('0.8b', '4b', 'laya', 'laya_td'):
        for s in (('public', 'jevbench', 'decisionbench', 'typed') if m.startswith('laya') else ('public', 'jevbench', 'typed')):
            f = os.path.join(ROOT, RES[m], 'methods', s, 'replay_s0.json'); rel = os.path.join(ROOT, RES[m], f'released_{s}.json')
            d = json.load(open(f))['info'].get('max_abs_dp') if os.path.exists(f) else None
            errs = sum('error' in r for r in json.load(open(rel)).values()) if os.path.exists(rel) else None
            L.append(f'| {m} | {s} | {"—" if d is None else f"{d:.1e}"} | {"not run" if errs is None else errs} |')
    text = '\n'.join(L) + '\n'
    open(os.path.join(ROOT, 'TABLES.md'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
