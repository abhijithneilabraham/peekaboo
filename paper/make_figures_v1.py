"""Figures, LaTeX tables and number macros for the Peekaboo paper, from the finished results in
experiments/peekaboo_remaining/output (and nothing else: every number in the paper is produced here).

    python paper/make_figures.py          # writes paper/figures/*.pdf, paper/tables.tex, paper/numbers.tex, paper/numbers.json

Metrics are VegaML's own (experiments/vega_policy/table.py -> harness). Gains are paired accuracy differences on the same
rows, each method averaged over its seeds, with a 95% bootstrap CI over rows (within task for the public menu).
"""
import json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PR = os.path.join(REPO, 'experiments', 'peekaboo_remaining')
os.environ.setdefault('VEGA_OUT', os.path.join(PR, 'output'))
sys.path.insert(0, PR)
import numpy as np
import matplotlib, matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import table as T          # experiments/peekaboo_remaining/table.py (loads VegaML's metrics)

FIG = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)
SUITES = ['public', 'jevbench', 'decisionbench', 'typed', 'emotion', 'agnews']
SNAME = {'public': '43 public tasks', 'jevbench': 'JevBench', 'decisionbench': 'DecisionBench', 'typed': 'typed-decisions',
         'emotion': 'emotion', 'agnews': 'AG News'}
SSHORT = {'public': 'public', 'jevbench': 'JevBench', 'decisionbench': 'DecisionB.', 'typed': 'typed-dec.', 'emotion': 'emotion', 'agnews': 'AG News'}
MODELS = [('0.8b', 'Vega 0.8B'), ('4b', 'Vega 4B'), ('laya', 'Laya')]
PFX = T.PREFIX
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'figure.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False})
C = {'base': '#9e9e9e', 'theirs': '#e07b39', 'sup': '#4c72b0', 'pk': '#2a9d3f', 'pk_sw': '#7bc96f', 'raw': '#b5d99c',
     'kl0': '#c44e52', 'pooled': '#8172b2', 'enc': '#1b5e20'}


# ───────────────────────────── numbers ─────────────────────────────
def paired_num(suite, a_runs, b_runs, reps=2000, seed=0):
    """b minus a in accuracy points on the same rows, mean over seeds, with a 95% bootstrap CI (None if either is missing)."""
    if not a_runs or not b_runs:
        return None
    its = T.VT.items(suite); rng = np.random.default_rng(seed)
    c = lambda rs: np.mean([[float(r.get('pred') == r['gold']) for r in recs] for recs in rs], 0)
    d = c(b_runs) - c(a_runs)
    groups = ([np.array([i for i, it in enumerate(its) if it['task'] == t]) for t in sorted({it['task'] for it in its})]
              if suite == 'public' else [np.arange(len(its))])
    stat = 100 * np.mean([d[g].mean() for g in groups])
    bs = [100 * np.mean([d[rng.choice(g, len(g))].mean() for g in groups]) for _ in range(reps)]
    return {'gain': float(stat), 'lo': float(np.percentile(bs, 2.5)), 'hi': float(np.percentile(bs, 97.5))}


def metric(model, suite, method, what, key='records'):
    """Per-seed values of one metric (accuracy, ece, nll, ...) for a (model, suite, method)."""
    rs = T.variant_runs(model, suite, method, key)
    vals = [T.VT.metrics(suite, r).get(f'{PFX[suite]}:{what}') for r in rs]
    return [v for v in vals if v is not None]


VARIANTS = [(m, k) for m, k, _, _ in T.VARIANTS]        # (method, record key) as TABLES.md lists them
NAMES = {(m, k): n.replace('**', '') for m, k, n, _ in T.VARIANTS}
NAMES[('feedback_kl0', 'records')] = 'Peekaboo without the KL term ($\\beta = 0$)'
NAMES[('feedback_enc', 'records')] = 'Peekaboo, encoder unfrozen'
NAMES[('feedback_enc', 'records_switch')] = 'Peekaboo, encoder unfrozen + switch'
NAMES[('feedback_enc', 'records_raw')] = 'Peekaboo, encoder unfrozen, no honesty fix'
for v in [('feedback_enc', 'records'), ('feedback_enc', 'records_switch'), ('feedback_enc', 'records_raw')]:
    if v not in VARIANTS:
        VARIANTS.append(v)


def compute():
    cache = os.path.join(HERE, 'numbers.json')
    N = {'grid': {}, 'rel': {}, 'reward': {}, 'T': {}, 'switch': {}, 'wf': {}, 'tasks': {}}
    for model, mname in MODELS:
        for method, key in VARIANTS:
            if (model == 'laya' and method == 'ttt') or (model != 'laya' and method in ('laya_td', 'feedback_enc')):
                continue
            for s in SUITES:
                rs = T.variant_runs(model, s, method, key)
                if not rs:
                    continue
                acc, ece = metric(model, s, method, 'accuracy', key), metric(model, s, method, 'ece', key)
                nll = metric(model, s, method, 'nll', key)
                g = None if method == 'released' else paired_num(s, T.runs(model, s, 'released'), rs)
                N['grid'][f'{model}|{method}|{key}|{s}'] = {'acc': acc, 'ece': ece, 'nll': nll, 'n': len(rs), 'gain': g}
                print(f'{mname:9s} {NAMES[(method, key)][:40]:40s} {s:13s} acc {np.mean(acc):.3f}'
                      + (f'  gain {g["gain"]:+.1f} [{g["lo"]:+.1f}, {g["hi"]:+.1f}]' if g else ''), flush=True)
    # head-to-head: Peekaboo against their improvement
    for model, _ in MODELS:
        theirs = 'laya_td' if model == 'laya' else 'ttt'
        for s in SUITES:
            for ours in (['feedback', 'feedback_enc'] if model == 'laya' else ['feedback']):
                g = paired_num(s, T.variant_runs(model, s, theirs, 'records'), T.runs(model, s, ours))
                if g:
                    N['grid'][f'{model}|{ours}-vs-{theirs}|records|{s}'] = {'gain': g}
            for ours in (['feedback', 'feedback_enc'] if model == 'laya' else ['feedback']):
                g = paired_num(s, T.runs(model, s, 'sup_ts'), T.runs(model, s, ours))
                if g:
                    N['grid'][f'{model}|{ours}-vs-sup_ts|records|{s}'] = {'gain': g}
    # reliability diagrams (10 equal-width bins on the confidence of the predicted answer)
    for model, _ in MODELS:
        for s in ('decisionbench', 'public', 'typed'):
            for method, key in (('released', 'records'), ('feedback', 'records'), ('feedback', 'records_raw')):
                rs = T.variant_runs(model, s, method, key)
                if not rs:
                    continue
                conf = np.concatenate([[max(r['probs'].values()) if r.get('probs') else 0.0 for r in recs] for recs in rs])
                corr = np.concatenate([[float(r.get('pred') == r['gold']) for r in recs] for recs in rs])
                edges = np.linspace(0, 1, 11); b = np.clip(np.digitize(conf, edges) - 1, 0, 9)
                N['rel'][f'{model}|{method}|{key}|{s}'] = [{'conf': float(conf[b == i].mean()), 'acc': float(corr[b == i].mean()), 'n': int((b == i).sum())}
                                                           for i in range(10) if (b == i).sum() >= 10]
    # reward curves, temperatures and the switch, from the run infos
    for model, _ in MODELS:
        for s in SUITES:
            for method in ('feedback', 'feedback_kl0', 'feedback_enc'):
                if model != 'laya' and method == 'feedback_enc':
                    continue
                import glob
                fs = sorted(glob.glob(os.path.join(T.ROOT, T.RES[model], 'methods', s, f'{method}_s[0-9].json')))
                curves, Ts, sw = [], [], []
                for f in fs:
                    for fo in json.load(open(f))['info'].get('folds', []):
                        if fo.get('reward_curve'):
                            curves.append(fo['reward_curve'])
                        if fo.get('T'):
                            Ts.append(fo['T'])
                        if fo.get('switch'):
                            sw.append(fo['switch'])
                if curves:
                    N['reward'][f'{model}|{method}|{s}'] = np.mean(curves, 0).tolist()
                if Ts:
                    N['T'][f'{model}|{method}|{s}'] = Ts
                if sw:
                    N['switch'][f'{model}|{method}|{s}'] = {'on': float(np.mean([x['on'] for x in sw])), 'adapter': float(np.mean([x['adapter'] for x in sw])),
                                                              'base': float(np.mean([x['base'] for x in sw])), 'n': len(sw)}
    # typed-decisions per workflow, public per task
    for model, _ in MODELS:
        for method in ('released', 'ttt', 'sup_ts', 'feedback', 'laya_td', 'feedback_enc'):
            rs = T.variant_runs(model, 'typed', method, 'records')
            if rs:
                N['wf'][f'{model}|{method}'] = {w: float(np.mean([T.VT.metrics('typed', r)[f'td:wf:{w}'] for r in rs]))
                                                for w in ('agent_trace_observability', 'customer_service', 'invoice_processing', 'security_incidents')}
            rs = T.variant_runs(model, 'public', method, 'records')
            if rs:
                ms = [T.VT.metrics('public', r) for r in rs]
                N['tasks'][f'{model}|{method}'] = {k[5:]: float(np.mean([m[k] for m in ms])) for k in ms[0] if k.startswith('task:')}
    json.dump(N, open(cache, 'w'), indent=0)
    return N


# ───────────────────────────── figures ─────────────────────────────
def G(N, model, method, s, key='records'):
    return N['grid'].get(f'{model}|{method}|{key}|{s}')




def fig_reliability(N):
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.6))
    for row, s in enumerate(('decisionbench', 'typed')):
        for ax, (model, mname) in zip(axes[row], MODELS):
            for method, key, label, col in (('released', 'records', 'base model', C['base']), ('feedback', 'records_raw', 'Peekaboo, no honesty fix', C['raw']),
                                            ('feedback', 'records', 'Peekaboo', C['pk'])):
                r = N['rel'].get(f'{model}|{method}|{key}|{s}')
                if r:
                    ax.plot([b['conf'] for b in r], [b['acc'] for b in r], 'o-', ms=3, color=col, label=label)
            ax.plot([0, 1], [0, 1], '--', color='k', lw=0.6); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f'{mname}, {SNAME[s]}')
            if row == 1:
                ax.set_xlabel('stated confidence')
            if ax is axes[row][0]:
                ax.set_ylabel('fraction correct')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(FIG, 'reliability.pdf'), bbox_inches='tight'); plt.close(fig)


def fig_ablation(N):
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.4))
    var = [('feedback', 'records', 'Peekaboo', C['pk']), ('feedback', 'records_switch', '+ switch', C['pk_sw']),
           ('feedback', 'records_raw', 'no honesty fix', C['raw']), ('feedback_kl0', 'records', 'no KL term ($\\beta = 0$)', C['kl0'])]
    w = 0.8 / len(var); x = np.arange(len(SUITES))
    for col_i, (model, mname) in enumerate(MODELS[:2]):
        for row, what in enumerate(('gain', 'ece')):
            ax = axes[row][col_i]
            for i, (method, key, label, col) in enumerate(var):
                g = [G(N, model, method, s, key) for s in SUITES]
                if what == 'gain':
                    y = [v['gain']['gain'] if v and v['gain'] else np.nan for v in g]
                else:
                    y = [np.mean(v['ece']) if v and v['ece'] else np.nan for v in g]
                ax.bar(x + (i - (len(var) - 1) / 2) * w, y, w, color=col, label=label)
            if what == 'gain':
                ax.axhline(0, color='k', lw=0.6)
                b = [G(N, model, 'released', s) for s in SUITES]
            else:
                ax.plot(x, [np.mean(G(N, model, 'released', s)['ece']) for s in SUITES], 'k_', ms=14, mew=1.2, label='base model')
            ax.set_xticks(x); ax.set_xticklabels([SSHORT[s] for s in SUITES], rotation=25, ha='right')
            ax.set_title(mname if row == 0 else '')
            ax.set_ylabel('accuracy gain (points)' if what == 'gain' else 'calibration error (ECE)')
    h, l = axes[1][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=5, loc='lower center', bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(FIG, 'ablation.pdf')); plt.close(fig)




def fig_reward(N):
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9), sharey=False)
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, (model, mname) in zip(axes, MODELS):
        for i, s in enumerate(SUITES):
            c = N['reward'].get(f'{model}|feedback|{s}')
            if c:
                ax.plot(range(1, len(c) + 1), c, 'o-', ms=3, color=cols[i], label=SNAME[s])
        ax.set_title(mname); ax.set_xlabel('pass over the feedback rows'); ax.set_xticks(range(1, 9))
    axes[0].set_ylabel('mean reward of sampled answers')
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, 'reward.pdf')); plt.close(fig)


def fig_laya(N):
    fig, ax = plt.subplots(figsize=(7, 3.0))
    bars = [('released', 'base Laya', C['base']), ('laya_td', 'laya-typed-decisions (answer key, whole model)', C['theirs']),
            ('feedback', 'Peekaboo, head only (right/wrong)', C['pk']), ('feedback_enc', 'Peekaboo, encoder unfrozen (right/wrong)', C['enc'])]
    w = 0.8 / len(bars); x = np.arange(len(SUITES))
    for i, (method, label, col) in enumerate(bars):
        y = [np.mean(G(N, 'laya', method, s)['acc']) if G(N, 'laya', method, s) else np.nan for s in SUITES]
        e = [np.std(G(N, 'laya', method, s)['acc']) if G(N, 'laya', method, s) and len(G(N, 'laya', method, s)['acc']) > 1 else 0 for s in SUITES]
        ax.bar(x + (i - (len(bars) - 1) / 2) * w, y, w, yerr=e, color=col, label=label, error_kw={'lw': 0.7, 'capsize': 1.5})
    ax.set_xticks(x); ax.set_xticklabels([SNAME[s] for s in SUITES], rotation=15, ha='right'); ax.set_ylabel('accuracy'); ax.set_ylim(0.3, 1.0)
    ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.22), fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'laya.pdf'), bbox_inches='tight'); plt.close(fig)




def fig_switch_T(N):
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    x = np.arange(len(SUITES)); w = 0.26
    for i, (model, mname) in enumerate(MODELS):
        Ts = [np.mean(N['T'].get(f'{model}|feedback|{s}', [np.nan])) for s in SUITES]
        axes[0].bar(x + (i - 1) * w, Ts, w, label=mname, color=[C['pk'], C['enc'], C['pooled']][i])
        on = [N['switch'].get(f'{model}|feedback|{s}', {}).get('on', np.nan) for s in SUITES]
        axes[1].bar(x + (i - 1) * w, on, w, label=mname, color=[C['pk'], C['enc'], C['pooled']][i])
    axes[0].axhline(1, color='k', lw=0.6, ls='--'); axes[0].set_ylabel('fitted temperature T'); axes[0].set_title('honesty fix: temperature fitted from right/wrong')
    axes[1].set_ylabel('share of runs keeping the add-on'); axes[1].set_title('feedback-checked switch'); axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([SSHORT[s] for s in SUITES], rotation=25, ha='right')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, 'switch_T.pdf'), bbox_inches='tight'); plt.close(fig)


MARK = {'theirs': ('D', C['theirs']), 'sup': ('s', C['sup']), 'pk': ('o', C['pk']), 'enc': ('o', C['enc']), 'pooled': ('s', C['pooled'])}


def forest(ax, N, model, series, title):
    """One panel of a forest plot: benchmarks on the y axis, gain over the base model on the x axis, one marker per
    method with its paired 95% bootstrap interval as a horizontal bar."""
    y = np.arange(len(SUITES))[::-1]
    k = len(series); off = np.linspace(0.3, -0.3, k) if k > 1 else [0.0]
    for (method, label, style), dy in zip(series, off):
        mk, col = MARK[style]
        for yi, s in zip(y, SUITES):
            g = G(N, model, method, s)
            if not g or not g['gain']:
                continue
            v = g['gain']
            ax.errorbar(v['gain'], yi + dy, xerr=[[v['gain'] - v['lo']], [v['hi'] - v['gain']]], fmt=mk, ms=4.5, color=col,
                        ecolor=col, elinewidth=1.0, capsize=2, label=label if yi == y[0] else None)
    ax.axvline(0, color='k', lw=0.7); ax.set_yticks(y); ax.set_yticklabels([SNAME[s] for s in SUITES]); ax.set_title(title)
    ax.set_ylim(-0.6, len(SUITES) - 0.4); ax.grid(axis='x', lw=0.4, alpha=0.5); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    for yi in y[1:]:
        ax.axhline(yi + 0.5, color='#dddddd', lw=0.5)


def fig_gains(N):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    forest(axes[0], N, '0.8b', [('ttt', 'their own method, answer key (TTT head / Laya fine-tune)', 'theirs'), ('sup_ts', 'our add-on trained on the answer key', 'sup'),
                                ('feedback', 'Peekaboo, right / wrong only', 'pk')], 'Vega 0.8B')
    forest(axes[1], N, '4b', [('ttt', '', 'theirs'), ('sup_ts', '', 'sup'), ('feedback', '', 'pk')], 'Vega 4B')
    forest(axes[2], N, 'laya', [('laya_td', '', 'theirs'), ('sup_ts', '', 'sup'), ('feedback', '', 'pk'),
                                ('feedback_enc', 'Peekaboo, encoder unfrozen, right / wrong only', 'enc')], 'Laya')
    axes[2].set_xlim(-12, 45)
    for ax in axes:
        ax.set_xlabel('accuracy gain over the base model (points)')
    h, l = axes[0].get_legend_handles_labels(); h2, l2 = axes[2].get_legend_handles_labels()
    h += [x for x, lab in zip(h2, l2) if lab]; l += [lab for lab in l2 if lab]
    fig.legend(h, l, frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(os.path.join(FIG, 'gains.pdf'), bbox_inches='tight'); plt.close(fig)


def fig_pooled(N):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, (model, mname) in zip(axes, MODELS):
        forest(ax, N, model, [('feedback', 'Peekaboo, one add-on per benchmark', 'pk'), ('pooled', 'pooled Peekaboo, one add-on for all benchmarks', 'pooled')], mname)
        ax.set_xlabel('accuracy gain over the base model (points)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(FIG, 'pooled.pdf'), bbox_inches='tight'); plt.close(fig)


def fig_benchmarks():
    import collections
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), gridspec_kw={'width_ratios': [1, 1.3]})
    y = np.arange(len(SUITES))[::-1]
    # left: test rows by question type
    types = ['choice', 'boolean', 'score']; tn = {'choice': 'pick one', 'boolean': 'yes / no', 'score': 'rating scale'}
    counts = {s: collections.Counter(it['question'].get('type') for it in T.VT.items(s)) for s in SUITES}
    left = np.zeros(len(SUITES))
    for t, col in zip(types, ('#4c72b0', '#55a868', '#c44e52')):
        v = np.array([counts[s].get(t, 0) for s in SUITES])
        axes[0].barh(y, v, left=left, color=col, label=tn[t], height=0.7); left += v
    axes[0].set_yticks(y); axes[0].set_yticklabels([SNAME[s] for s in SUITES]); axes[0].set_xlabel('test rows'); axes[0].set_title('test rows by question type')
    axes[0].legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.22))
    # right: share of test rows by number of options, one stacked bar per benchmark
    bins = [(2, 2, '2'), (3, 3, '3'), (4, 4, '4'), (5, 5, '5'), (6, 8, '6 to 8'), (9, 10**9, '9 or more')]
    shades = plt.cm.Blues(np.linspace(0.3, 0.95, len(bins)))
    left = np.zeros(len(SUITES))
    for (lo, hi, lab), col in zip(bins, shades):
        v = []
        for s in SUITES:
            k = np.array([len(it['labels']) for it in T.VT.items(s)])
            v.append(((k >= lo) & (k <= hi)).mean())
        v = np.array(v)
        axes[1].barh(y, v, left=left, color=col, label=lab, height=0.7, edgecolor='white', lw=0.5)
        for yi, vi, li in zip(y, v, left):
            if vi >= 0.12:
                axes[1].text(li + vi / 2, yi, f'{100 * vi:.0f}%', ha='center', va='center', fontsize=7, color='white' if col[2] < 0.8 else 'black')
        left += v
    axes[1].set_yticks(y); axes[1].set_yticklabels([]); axes[1].set_xlim(0, 1); axes[1].set_xlabel('share of test rows')
    axes[1].set_title('number of options per question'); axes[1].tick_params(axis='y', length=0)
    axes[1].legend(frameon=False, ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.22), title='options', title_fontsize=8, handlelength=1.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'benchmarks.pdf'), bbox_inches='tight'); plt.close(fig)


def fig_curve():
    """Learning curve from the earlier DecisionBench study (experiments/vega_policy/results, results_4b): gain and ECE
    on the fixed 1,720-row test set against the number of feedback rows per task."""
    import glob
    VP = os.path.join(REPO, 'experiments', 'vega_policy')
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6))
    series = [('kl0.1', 'Peekaboo, right / wrong only', C['pk']), ('kl1', 'right / wrong only, stronger KL term ($\\beta = 1$)', C['pk_sw']),
              ('sup', 'answer-key adapter, no temperature step', C['sup'])]
    for row, (d, mname) in enumerate((('results', 'Vega 0.8B'), ('results_4b', 'Vega 4B'))):
        R = os.path.join(VP, d)
        Z = [json.load(open(f))['final']['test'] for f in glob.glob(os.path.join(R, 'db_zero_rows_s*_big.json'))]
        z, ze = np.mean([x['acc'] for x in Z]), np.mean([x['ece'] for x in Z])
        for tag, label, col in series:
            xs, g, gs, e = [], [], [], []
            for n in (5, 15, 30, 60):
                fs = glob.glob(os.path.join(R, f'db_*_rows_s*_big_n{n}_{tag}.json'))
                if not fs:
                    continue
                key = 'test' if tag == 'sup' else 'test_recal'
                acc = [json.load(open(f))['final'][key]['acc'] for f in fs]; ece = [json.load(open(f))['final'][key]['ece'] for f in fs]
                xs.append(n); g.append(100 * (np.mean(acc) - z)); gs.append(100 * np.std(acc)); e.append(np.mean(ece))
            axes[row][0].errorbar(xs, g, gs, marker='o', ms=4, color=col, label=label, capsize=2, lw=1.2)
            axes[row][1].plot(xs, e, marker='o', ms=4, color=col, lw=1.2)
        axes[row][0].axhline(0, color='k', lw=0.6); axes[row][1].axhline(ze, color=C['base'], lw=1.0, ls='--', label='base model')
        for ax in axes[row]:
            ax.set_xscale('log'); ax.set_xticks([5, 15, 30, 60]); ax.set_xticklabels(['5', '15', '30', '60']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[row][0].set_ylabel(f'{mname}\ngain over base (points)'); axes[row][1].set_ylabel('ECE (lower is better)')
    for ax in axes[1]:
        ax.set_xlabel('feedback rows per task (43 tasks)')
    axes[0][0].set_title('accuracy'); axes[0][1].set_title('calibration')
    h, l = axes[0][0].get_legend_handles_labels(); h2, l2 = axes[0][1].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(os.path.join(FIG, 'curve.pdf'), bbox_inches='tight'); plt.close(fig)


# ───────────────────────────── LaTeX ─────────────────────────────
def f3(v):
    return '' if not v else f'{np.mean(v):.3f}' + (f' {{\\scriptsize$\\pm${np.std(v):.3f}}}' if len(v) > 1 else '')


def fg(g, ci=True):
    if not g:
        return '--'
    return f'{g["gain"]:+.1f}' + (f' {{\\scriptsize[{g["lo"]:+.1f}, {g["hi"]:+.1f}]}}' if ci else '')


def tex_tables(N):
    OUT = {}
    L = []
    # main table: accuracy, every model, their method, answer key, Peekaboo (+ variants), per suite
    main_rows = {'0.8b': [('released', 'records', 'base model', 'none'), ('ttt', 'records', "VegaML's TTT head", 'answer key'),
                          ('sup_ts', 'records', 'answer-key adapter', 'answer key'), ('feedback', 'records', '\\textbf{Peekaboo}', 'right / wrong'),
                          ('pooled', 'records', 'pooled Peekaboo', 'right / wrong')],
                 '4b': None,
                 'laya': [('released', 'records', 'base model', 'none'), ('laya_td', 'records', "laya-typed-decisions (Laya's fine-tune)", 'answer key'),
                          ('sup_ts', 'records', 'answer-key head', 'answer key'), ('feedback', 'records', '\\textbf{Peekaboo}, head only', 'right / wrong'),
                          ('feedback_enc', 'records', '\\textbf{Peekaboo}, encoder unfrozen', 'right / wrong'), ('pooled', 'records', 'pooled Peekaboo, head only', 'right / wrong')]}
    main_rows['4b'] = main_rows['0.8b']
    for what, caption, label in (('acc', 'Accuracy', 'tab:acc'), ('gain', 'Gain over the base model, in accuracy points', 'tab:gain'),
                                 ('ece', 'Expected calibration error (lower is better)', 'tab:ece')):
        L += [f'% ---- {what}', '\\begin{table}[tbp]\\centering\\scriptsize', f'\\caption{{{caption}. Mean over 3 seeds ($\\pm$ s.d.) where a method is stochastic. '
              + ('Each gain is paired on the same rows against the model\'s own base; the CIs are in Table~\\ref{tab:h2h} and Figure~\\ref{fig:gains}. ' if what == 'gain' else '')
              + 'JevBench has no training split: Peekaboo learns by 5-fold cross-fitting (paraphrase groups kept together), pooled Peekaboo never sees it, and VegaML\'s TTT and the answer-key methods are not defined for it.}',
              f'\\label{{{label}}}', '\\setlength{\\tabcolsep}{2.5pt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{llr' + 'r' * len(SUITES) + '}', '\\toprule',
              'model & method & learns from & ' + ' & '.join(SSHORT[s] for s in SUITES) + ' \\\\', '\\midrule']
        for model, mname in MODELS:
            for i, (method, key, name, src) in enumerate(main_rows[model]):
                cells = []
                for s in SUITES:
                    g = G(N, model, method, s, key)
                    if not g:
                        cells.append('$\\cdot$'); continue
                    if what == 'gain':
                        cells.append('--' if method == 'released' else fg(g['gain'], ci=False))
                    else:
                        cells.append(f3(g[what]))
                L.append(f'{mname if i == 0 else ""} & {name} & {src} & ' + ' & '.join(cells) + ' \\\\')
            if model != 'laya':
                L.append('\\addlinespace')
        L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}', '']
    OUT['tables_main.tex'] = L; L = []
    # ablations: accuracy and ECE for every variant
    L += ['% ---- ablations', '\\begin{table}[tbp]\\centering\\scriptsize', '\\caption{Every variant and ablation: accuracy (top) and expected calibration error (bottom), mean over 3 seeds. '
          '``switch\'\': the add-on is used only where held-out right / wrong feedback says it beats the base model. ``no honesty fix\'\': the trained add-on without the feedback-fitted temperature. ``no KL term\'\': $\\beta = 0$.}',
          '\\label{tab:ablation}', '\\setlength{\\tabcolsep}{3pt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{ll' + 'r' * len(SUITES) + '}', '\\toprule',
          'model & variant & ' + ' & '.join(SSHORT[s] for s in SUITES) + ' \\\\']
    for what in ('acc', 'ece'):
        L += ['\\midrule', f'\\multicolumn{{{2 + len(SUITES)}}}{{l}}{{\\textit{{{"accuracy" if what == "acc" else "calibration error (ECE)"}}}}} \\\\', '\\midrule']
        for model, mname in MODELS:
            first = True
            for method, key in VARIANTS:
                if method in ('ttt', 'sup_ts', 'laya_td') or (model == 'laya' and method == 'ttt') or (model != 'laya' and method in ('laya_td', 'feedback_enc')):
                    continue
                cells = [f3(G(N, model, method, s, key)[what]) if G(N, model, method, s, key) else '$\\cdot$' for s in SUITES]
                if not any(c not in ('', '$\\cdot$') for c in cells):
                    continue
                L.append(f'{mname if first else ""} & {NAMES[(method, key)]} & ' + ' & '.join(cells) + ' \\\\'); first = False
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}', '']
    OUT['tables_ablation.tex'] = L; L = []
    # head to head
    L += ['% ---- head to head', '\\begin{table}[tbp]\\centering\\small', '\\caption{Peekaboo head to head, in accuracy points with a paired 95\\% bootstrap CI. Positive means Peekaboo is better. '
          '``their improvement\'\' is VegaML\'s TTT head for Vega and the released laya-typed-decisions fine-tune for Laya, both trained with the answer key.}',
          '\\label{tab:h2h}', '\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{llrrr}', '\\toprule', 'model & benchmark & vs. base & vs. their improvement & vs. answer-key adapter \\\\', '\\midrule']
    blocks = [('0.8b', 'Vega 0.8B', 'feedback', 'ttt'), ('4b', 'Vega 4B', 'feedback', 'ttt'),
              ('laya', 'Laya, head only', 'feedback', 'laya_td'), ('laya', 'Laya, encoder unfrozen', 'feedback_enc', 'laya_td')]
    for bi, (model, mname, ours, theirs) in enumerate(blocks):
        first = True
        for s in SUITES:
            g1 = G(N, model, ours, s); g2 = N['grid'].get(f'{model}|{ours}-vs-{theirs}|records|{s}'); g3 = N['grid'].get(f'{model}|{ours}-vs-sup_ts|records|{s}')
            if not g1:
                continue
            L.append(f'{mname if first else ""} & {SNAME[s]} & {fg(g1["gain"])} & {fg(g2["gain"]) if g2 else "--"} & {fg(g3["gain"]) if g3 else "--"} \\\\'); first = False
        if bi < len(blocks) - 1:
            L.append('\\addlinespace')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    OUT['tables_h2h.tex'] = L; L = []
    # typed-decisions per workflow
    L += ['% ---- workflows', '\\begin{table}[tbp]\\centering\\small', '\\caption{typed-decisions accuracy per workflow (500 test decisions each).}', '\\label{tab:wf}',
          '\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{llrrrr}', '\\toprule', 'model & method & agent traces & customer service & invoices & security \\\\', '\\midrule']
    for model, mname in MODELS:
        rows = [('released', 'base'), ('ttt', "TTT head"), ('sup_ts', 'answer-key adapter'), ('feedback', 'Peekaboo')] if model != 'laya' else \
               [('released', 'base'), ('laya_td', 'laya-typed-decisions'), ('sup_ts', 'answer-key head'), ('feedback', 'Peekaboo, head only'), ('feedback_enc', 'Peekaboo, encoder unfrozen')]
        first = True
        for method, name in rows:
            w = N['wf'].get(f'{model}|{method}')
            if w:
                L.append(f'{mname if first else ""} & {name} & ' + ' & '.join(f'{w[k]:.3f}' for k in ('agent_trace_observability', 'customer_service', 'invoice_processing', 'security_incidents')) + ' \\\\'); first = False
        if model != 'laya':
            L.append('\\addlinespace')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    OUT['tables_wf.tex'] = L; L = []
    # the 43 public tasks, one row each
    cols = [('0.8b', 'released', 'base'), ('0.8b', 'ttt', 'TTT'), ('0.8b', 'sup_ts', 'key'), ('0.8b', 'feedback', 'Pk'),
            ('4b', 'released', 'base'), ('4b', 'ttt', 'TTT'), ('4b', 'sup_ts', 'key'), ('4b', 'feedback', 'Pk'),
            ('laya', 'released', 'base'), ('laya', 'laya_td', 'fine-tune'), ('laya', 'feedback', 'Pk head'), ('laya', 'feedback_enc', 'Pk enc')]
    cols = [c for c in cols if f'{c[0]}|{c[1]}' in N['tasks']]
    tasks = sorted(N['tasks']['0.8b|released'])
    L += ['% ---- per task', '\\begin{table}[tbp]\\centering\\scriptsize',
          '\\caption{Accuracy on each of the 43 public tasks (up to 60 test rows each; mean over seeds). ``key\'\' is the answer-key adapter / head, ``Pk\'\' is Peekaboo, '
          '``fine-tune\'\' is laya-typed-decisions, ``Pk enc\'\' is Peekaboo with the encoder unfrozen. Bold: the best method for that model on that task.}', '\\label{tab:tasks}',
          '\\setlength{\\tabcolsep}{2.2pt}', '\\begin{tabular}{l' + 'r' * len(cols) + '}', '\\toprule',
          'task & ' + ' & '.join(f'\\multicolumn{{{sum(1 for c in cols if c[0] == m)}}}{{c}}{{{mn}}}' for m, mn in MODELS if any(c[0] == m for c in cols)) + ' \\\\',
          ' & ' + ' & '.join(c[2] for c in cols) + ' \\\\', '\\midrule']
    for t in tasks:
        row = []
        for m, _ in MODELS:
            vals = [N['tasks'][f'{c[0]}|{c[1]}'].get(t) for c in cols if c[0] == m]
            best = max(v for v in vals if v is not None) if any(v is not None for v in vals) else None
            row += [('' if v is None else (f'\\textbf{{{v:.2f}}}' if abs(v - best) < 5e-4 else f'{v:.2f}')) for v in vals]
        L.append(t.replace('_', '\\_') + ' & ' + ' & '.join(row) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    OUT['tables_tasks.tex'] = L
    for name, lines in OUT.items():
        open(os.path.join(HERE, name), 'w').write('\n'.join(lines) + '\n')


def tex_numbers(N):
    """\\num{...} style macros for the text: \\acc{model}{method}{suite}, \\gain..., written as plain commands."""
    def nm(*parts):
        tr = {'0.8b': 'Small', '4b': 'Big', 'laya': 'Laya', 'released': 'Base', 'ttt': 'Ttt', 'sup_ts': 'Sup', 'feedback': 'Pk', 'pooled': 'Pooled',
              'feedback_kl0': 'Klzero', 'feedback_enc': 'Enc', 'laya_td': 'Td', 'public': 'Pub', 'jevbench': 'Jev', 'decisionbench': 'Db', 'typed': 'Typed',
              'emotion': 'Emo', 'agnews': 'Ag', 'records': '', 'records_switch': 'Sw', 'records_raw': 'Raw'}
        return ''.join(tr.get(p, p) for p in parts)
    L = ['% generated by paper/make_figures.py; every number in the text comes from here']
    for k, v in N['grid'].items():
        model, method, key, s = k.split('|')
        if '-vs-' in method:
            a, b = method.split('-vs-')
            if v.get('gain'):
                L.append(f'\\newcommand{{\\g{nm(model, a)}Vs{nm(b, s)}}}{{{v["gain"]["gain"]:+.1f}}}')
                L.append(f'\\newcommand{{\\ci{nm(model, a)}Vs{nm(b, s)}}}{{[{v["gain"]["lo"]:+.1f}, {v["gain"]["hi"]:+.1f}]}}')
            continue
        base = nm(model, method, key, s)
        if v['acc']:
            L.append(f'\\newcommand{{\\acc{base}}}{{{np.mean(v["acc"]):.3f}}}')
        if v['ece']:
            L.append(f'\\newcommand{{\\ece{base}}}{{{np.mean(v["ece"]):.3f}}}')
        if v['nll']:
            L.append(f'\\newcommand{{\\nll{base}}}{{{np.mean(v["nll"]):.3f}}}')
        if v.get('gain'):
            L.append(f'\\newcommand{{\\gain{base}}}{{{v["gain"]["gain"]:+.1f}}}')
            L.append(f'\\newcommand{{\\ci{base}}}{{[{v["gain"]["lo"]:+.1f}, {v["gain"]["hi"]:+.1f}]}}')
    open(os.path.join(HERE, 'numbers.tex'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    N = json.load(open(os.path.join(HERE, 'numbers.json'))) if '--tables-only' in sys.argv else compute()
    fig_gains(N); fig_reliability(N); fig_ablation(N); fig_pooled(N); fig_reward(N); fig_laya(N); fig_benchmarks(); fig_switch_T(N); fig_curve()
    for f in ('vega_feedback_curve.png', 'vega_feedback_curve_4b.png'):
        if os.path.exists(os.path.join(FIG, f)):
            os.remove(os.path.join(FIG, f))
    tex_tables(N); tex_numbers(N)
    print('wrote', FIG, 'tables.tex numbers.tex numbers.json')
