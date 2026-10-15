"""The final tables, in the layout of VegaML's README benchmark section, for every size that has results.

For each size: the README's published number, the released model measured in this run (decide()'s own records),
and the released model + our feedback adapter (mean ± s.d. over seeds). Jev is copied from the README. All metrics
are computed by VegaML's own harness code (harness/bench_common.py, bench_run.py, bench_suites.py).

    python table.py          # reads VEGA_OUT, writes VEGA_OUT/TABLES.md
"""
import glob, json, os, re, subprocess, sys
import numpy as np

import paths
paths.harness()
import bench_common as bc
import bench_run as BR
import bench_suites as BS

ROOT = paths.ROOT
SIZES = ['0.8b', '4b']
SFX = {'0.8b': '', '4b': '_4b'}
METHODS = [('released', 'released Vega', 'none'), ('ttt', "VegaML's TTT head, 'auto'", 'gold labels'),
           ('sup_ts', 'supervised adapter + temp. scaling', 'gold labels'), ('linear_fb', 'no-engine control', 'right/wrong'),
           ('feedback_kl1', 'ours, KL 1', 'right/wrong'), ('feedback', '**ours**', 'right/wrong')]
README_TASKS = ['unfair_tos', 'tweet_stance_abortion', 'prompt_injections', 'tweet_sentiment', 'tweet_offensive', 'enron_spam',
                'jailbreak_cls', 'glue_rte', 'emotion', 'toxic_chat_toxicity', 'superglue_cb', 'clinc150_oos']
# (row key, label, higher is better, README 0.8B, README 4B, README Jev)
ROWS = [('h', 'Public tasks the README lists (accuracy)')] + [
    (f'task:{k}', k, True, *v) for k, v in zip(README_TASKS, [
        (.737, .877, .667), (.364, .545, .436), (.746, .855, .764), (.424, .593, .525), (.780, .780, .729), (1.0, 1.0, .950),
        (.983, 1.0, .966), (.633, .883, .850), (.550, .650, .617), (.950, 1.0, .983), (.667, .822, .800), (.683, .900, .883)])] + [
    ('h', 'Slices'),
    ('jev:temporal', 'JevBench family:temporal_numeric', True, .467, .400, .200),
    ('db:privacy', 'DecisionBench domain:privacy', True, .880, 1.0, .960),
    ('db:public_sector', 'DecisionBench domain:public_sector', True, .760, .960, .880),
    ('h', 'Public menu, 43 tasks, macro average'),
    ('pub:accuracy', 'accuracy', True, .556, .717, .836), ('pub:skill', 'skill, chance corrected', True, .388, .594, .755),
    ('pub:macro_f1', 'macro F1', True, .486, .649, .796), ('pub:ece', 'expected calibration error', False, .151, .148, .115),
    ('pub:nll', 'negative log likelihood', False, 1.099, .768, .878), ('pub:aurc', 'area under risk-coverage', False, .362, .169, .077),
    ('pub:p50', 'latency per decision, p50 ms', False, 74, 106, 162),
    ('h', 'JevBench, 231 items'),
    ('jev:accuracy', 'accuracy', True, .545, .662, .866), ('jev:hard', 'accuracy, hard tier', True, .414, .513, .730),
    ('jev:paraphrase', 'paraphrase consistency', True, .694, .722, .972), ('jev:ece', 'expected calibration error', False, .178, .169, .067),
    ('h', 'DecisionBench, 1,075 items'),
    ('db:accuracy', 'accuracy', True, .509, .633, .752), ('db:ece', 'expected calibration error', False, .103, .098, .122),
    ('db:nll', 'negative log likelihood', False, 1.628, 1.420, 3.196), ('db:kl', 'KL to the gold distribution, 10 rows', False, 1.012, .854, 6.099),
    ('db:coverage', 'coverage, rows answered', True, 1.0, 1.0, .997), ('db:p50', 'latency per decision, p50 ms', False, 85, 245, 163)]


# ───────────────────────────── loading ─────────────────────────────
def items(suite):
    f = os.path.join(ROOT, f'items_{suite}.json')
    return [it for it in json.load(open(f))['items'] if it['split'] == 'test'] if os.path.exists(f) else []


def runs(size, suite, method):
    """List of record lists, one per seed (released: one, decide()'s own records)."""
    res = os.path.join(ROOT, 'results' + SFX[size])
    if method == 'released':
        f = os.path.join(res, f'released_{suite}.json')
        if not os.path.exists(f):
            return []
        recs = json.load(open(f)); its = items(suite)
        return [[recs[it['id']] for it in its]] if all(it['id'] in recs for it in its) else []
    return [json.load(open(f))['records'] for f in sorted(glob.glob(os.path.join(res, 'methods', suite, f'{method}_s*.json')))]


# ───────────────────────────── metrics, VegaML's harness ─────────────────────────────
def metrics(suite, recs):
    its = items(suite); m = {}
    if suite == 'public':
        per = {}
        for t in sorted({it['task'] for it in its}):
            idx = [i for i, it in enumerate(its) if it['task'] == t]
            labels = its[idx[0]]['labels']
            per[t] = bc.score_metrics([recs[i] for i in idx], labels, its[idx[0]]['qtype'], 1.0 / max(len(labels), 2))
            m[f'task:{t}'] = per[t]['accuracy']
        for k, f in [('accuracy', 'accuracy'), ('skill', 'skill'), ('macro_f1', 'macro_f1'), ('ece', 'ece'), ('nll', 'nll'), ('aurc', 'aurc')]:
            vals = [p[f] for p in per.values() if p.get(f) is not None]
            m[f'pub:{k}'] = float(np.mean(vals)) if vals else None
        lat = [p['latency_ms_p50'] for p in per.values() if p.get('latency_ms_p50') is not None]
        m['pub:p50'] = float(np.median(lat)) if lat else None
        m['tasks'] = len(per)
    elif suite == 'jevbench':
        al = BR.sliced(its, recs, lambda it: 'all'); m['jev:accuracy'] = al['all'].get('accuracy'); m['jev:ece'] = al['all'].get('ece')
        m['jev:hard'] = BR.sliced(its, recs, lambda it: 'tier:' + str(it['tier'])).get('tier:hard', {}).get('accuracy')
        m['jev:temporal'] = BR.sliced(its, recs, lambda it: 'family:' + str(it['family'])).get('family:temporal_numeric', {}).get('accuracy')
        m['jev:paraphrase'] = BS.paraphrase_consistency(its, recs).get('paraphrase_consistency')
    else:
        al = BR.sliced(its, recs, lambda it: 'all')['all']
        m.update({'db:accuracy': al.get('accuracy'), 'db:ece': al.get('ece'), 'db:nll': al.get('nll'), 'db:coverage': al.get('coverage'),
                  'db:p50': al.get('latency_ms_p50'), 'db:kl': BS.soft_gold_kl(its, recs).get('soft_gold_kl_mean')})
        dom = BR.sliced(its, recs, lambda it: 'domain:' + str(it['domain']), min_rows=1)
        m['db:privacy'] = dom.get('domain:privacy', {}).get('accuracy'); m['db:public_sector'] = dom.get('domain:public_sector', {}).get('accuracy')
    return m


def gain(suite, base, ours, reps=2000):
    """Paired accuracy gain (points) of ours (mean over seeds) over the released model on the same rows, with a
    bootstrap 95% CI. The public menu is a macro average over tasks, so it resamples rows within each task."""
    its = items(suite); rng = np.random.default_rng(0)
    c = lambda recs: np.array([float(r.get('pred') == r['gold']) for r in recs])
    d = np.mean([c(o) for o in ours], 0) - c(base[0])
    groups = [np.array([i for i, it in enumerate(its) if it['task'] == t]) for t in sorted({it['task'] for it in its})] if suite == 'public' else [np.arange(len(its))]
    stat = lambda dd: np.mean([dd[g].mean() for g in groups])
    bs = [np.mean([d[rng.choice(g, len(g))].mean() for g in groups]) for _ in range(reps)]
    return 100 * stat(d), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


# ───────────────────────────── tables ─────────────────────────────
def fmt(x, key):
    if x is None:
        return 'not run'
    return f'{x:.0f}' if key.endswith('p50') else f'{x:.3f}'


def main():
    M = {}                                  # (size, method) -> {row key: [values per seed]}
    G = {}                                  # (size, suite, method) -> gain
    for size in SIZES:
        for method, _, _ in METHODS:
            vals = {}
            for suite in ('public', 'jevbench', 'decisionbench'):
                rs = runs(size, suite, method)
                for recs in rs:
                    for k, x in metrics(suite, recs).items():
                        vals.setdefault(k, []).append(x)
                base = runs(size, suite, 'released')
                if method != 'released' and rs and base:
                    G[(size, suite, method)] = gain(suite, base, rs)
            M[(size, method)] = vals
    cell = lambda size, method, key: M[(size, method)].get(key)
    mean = lambda v: None if not v or any(x is None for x in v) else float(np.mean(v))

    L = ['# Vega + feedback adapter: reproduction tables', '',
         'Columns per size: the README\'s published number, the released model measured in this run, and the released model plus our feedback adapter (mean ± s.d. over seeds). Jev is copied from the README. Metrics are computed with VegaML\'s own benchmark code. Bold = best in the row.', '']
    head = ['benchmark']
    for size in SIZES:
        head += [f'Vega {size.upper()} (README)', f'Vega {size.upper()} (this run)', f'**{size.upper()} + ours**']
    head += ['Jev (README)']
    L += ['| ' + ' | '.join(head) + ' |', '|---' * len(head) + '|']
    for row in ROWS:
        if row[0] == 'h':
            L.append(f'| **{row[1]}** |' + ' |' * (len(head) - 1)); continue
        key, name, hib, r08, r4, rj = row
        vals, txt = [], []
        for size, rd in (('0.8b', r08), ('4b', r4)):
            rel, ours = cell(size, 'released', key), cell(size, 'feedback', key)
            for x, t in [(rd, fmt(rd, key)), (mean(rel), fmt(mean(rel), key)),
                         (mean(ours), fmt(mean(ours), key) + (f' ± {np.std(ours):.3f}' if ours and len(ours) > 1 and mean(ours) is not None and not key.endswith('p50') else ''))]:
                vals.append(x); txt.append(t)
        vals.append(rj); txt.append(fmt(rj, key))
        ok = [x for x in vals if x is not None]
        best = (max if hib else min)(ok) if ok else None
        txt = [f'**{t}**' if x is not None and best is not None and abs(x - best) < 5e-4 else t for x, t in zip(vals, txt)]
        L.append(f'| {name} | ' + ' | '.join(txt) + ' |')

    L += ['', '## Paired accuracy gain of ours over the released model, same rows (points, bootstrap 95% CI)', '',
          '| suite | ' + ' | '.join(s.upper() for s in SIZES) + ' |', '|---|' + '---|' * len(SIZES)]
    for suite in ('public', 'jevbench', 'decisionbench'):
        row = []
        for size in SIZES:
            g = G.get((size, suite, 'feedback'))
            row.append(f'{g[0]:+.1f} [{g[1]:+.1f}, {g[2]:+.1f}]' if g else 'not run')
        L.append(f'| {suite} | ' + ' | '.join(row) + ' |')

    L += ['', '## Every method, accuracy / ECE / NLL (mean over seeds) and paired gain vs released', '']
    for suite, pre in (('public', 'pub'), ('jevbench', 'jev'), ('decisionbench', 'db')):
        L += [f'### {suite}', '', '| method | learns from | ' + ' | '.join(f'{s.upper()} acc | {s.upper()} ECE | {s.upper()} NLL | {s.upper()} gain' for s in SIZES) + ' |',
              '|---|---|' + '---|---|---|---|' * len(SIZES)]
        for method, name, src in METHODS:
            row = []
            for size in SIZES:
                v = M[(size, method)]
                n = lambda k: fmt(mean(v.get(f'{pre}:{k}')), k)
                g = G.get((size, suite, method))
                row += [n('accuracy'), n('ece'), n('nll') if suite != 'jevbench' else '—', f'{g[0]:+.1f} [{g[1]:+.1f}, {g[2]:+.1f}]' if g else '—']
            if any(x not in ('not run', '—') for x in row):
                L.append(f'| {name} | {src} | ' + ' | '.join(row) + ' |')
        L.append('')

    for size in SIZES:                      # Section 5.5 and Section 7 summaries, when those steps ran
        res = os.path.join(ROOT, 'results' + SFX[size])
        for title, script, pat in [('feedback amount, one pass, unseen tasks (DecisionBench)', 'dbgroup.py', 'db_*_big*.json'),
                                   ('KL weight pilot (DecisionBench)', 'dbgroup.py', 'db_*_s[0-9]_kl*.json'), ('Kuhn poker', 'group.py', 'kuhn_*.json')]:
            if glob.glob(os.path.join(res, pat)):
                out = subprocess.run([sys.executable, os.path.join(paths.HERE, script), os.path.join(res, pat)], capture_output=True, text=True).stdout
                L += [f'## {size.upper()}: {title}', '', '```', out.rstrip(), '```', '']
    text = '\n'.join(L) + '\n'
    open(os.path.join(ROOT, 'TABLES.md'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
