"""Figures, LaTeX tables and number macros for the paper, from a finished grid and its analysis.json
(experiments/peekaboo_remaining/analysis.py). Every number in the paper comes from here.

    VEGA_OUT=experiments/peekaboo_remaining/output_v2 python paper/make_figures.py
"""
import json, os, sys
import numpy as np
import matplotlib, matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PR = os.path.join(REPO, 'experiments', 'peekaboo_remaining')
os.environ.setdefault('VEGA_OUT', os.path.join(PR, 'output_v2'))
sys.path.insert(0, PR)
import analysis as A   # noqa: E402

OUT = A.OUT
N = json.load(open(os.path.join(OUT, 'analysis.json')))
FIG = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)
SUITES = A.SUITES
SNAME = {'public': '43 public tasks', 'jevbench': 'JevBench', 'decisionbench': 'DecisionBench', 'typed': 'typed-decisions', 'emotion': 'emotion', 'agnews': 'AG News'}
SSHORT = {'public': 'public', 'jevbench': 'JevBench', 'decisionbench': 'DecisionB.', 'typed': 'typed-dec.', 'emotion': 'emotion', 'agnews': 'AG News'}
MODELS = [('0.8b', 'Vega 0.8B'), ('4b', 'Vega 4B'), ('laya', 'Laya')]
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'figure.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False})
C = {'base': '#9e9e9e', 'theirs': '#e07b39', 'sup': '#4c72b0', 'sup_es': '#7fa6d8', 'pk': '#2a9d3f', 'pk1': '#8fd19e', 'raw': '#b5d99c',
     'kl0': '#c44e52', 'pooled': '#8172b2', 'enc': '#1b5e20', 'enc1': '#66bb6a', 'prior': '#b58900', 'logged': '#6c757d'}
MARK = {'theirs': 'D', 'sup': 's', 'sup_es': 's', 'pk': 'o', 'pk1': 'o', 'enc': 'o', 'enc1': 'o', 'prior': '^', 'logged': 'v', 'pooled': 's', 'kl0': 'x'}


PK = lambda model: 'feedback_enc' if model == 'laya' else 'feedback'          # the main Peekaboo arm per model in this grid
SUP = lambda model: 'sup_enc' if model == 'laya' else 'sup_ts'


# ───────────────────────────── access ─────────────────────────────
def cmp(model, meth, key, suite, vs='released', vkey='records', suffix=''):
    k = f'{model}|{meth}|{key}|{suite}|vs|{vs}' + ('' if vs == 'released' else f'|{vkey}') + suffix
    return N['compare'].get(k)


def sc(model, meth, key, suite, what):
    v = N['scores'].get(f'{model}|{meth}|{key}|{suite}')
    return None if not v or not v.get(what) or v[what][0] is None else v[what]


def fg(g, ci=True, mark=True):
    if not g:
        return '--'
    s = f'{g["gain"]:+.1f}'
    if ci:
        s += f' {{\\scriptsize[{g["lo"]:+.1f}, {g["hi"]:+.1f}]}}'
    if mark and g.get('sig_holm'):
        s += '$^{\\dagger}$'
    return s


def f3(v):
    return '--' if not v else f'{np.mean(v):.3f}' + (f' {{\\scriptsize$\\pm${np.std(v, ddof=1):.3f}}}' if len(v) > 1 else '')


# ───────────────────────────── figures ─────────────────────────────
def forest(ax, model, series, title, xlim=None):
    y = np.arange(len(SUITES))[::-1]
    off = np.linspace(0.32, -0.32, len(series)) if len(series) > 1 else [0.0]
    for (meth, key, label, style), dy in zip(series, off):
        for yi, s in zip(y, SUITES):
            g = cmp(model, meth, key, s)
            if not g:
                continue
            ax.errorbar(g['gain'], yi + dy, xerr=[[g['gain'] - g['lo']], [g['hi'] - g['gain']]], fmt=MARK[style], ms=4.2, color=C[style],
                        ecolor=C[style], elinewidth=1.0, capsize=2, label=label if yi == y[0] else None)
    ax.axvline(0, color='k', lw=0.7); ax.set_yticks(y); ax.set_yticklabels([SNAME[s] for s in SUITES]); ax.set_title(title)
    ax.set_ylim(-0.6, len(SUITES) - 0.4); ax.grid(axis='x', lw=0.4, alpha=0.5); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    for yi in y[1:]:
        ax.axhline(yi + 0.5, color='#dddddd', lw=0.5)
    if xlim:
        ax.set_xlim(*xlim)


def fig_gains():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    vega = [('ttt', 'records_guard', "their own method, answer key (TTT head, 0.12.3 rule / Laya's typed-decisions fine-tune)", 'theirs'),
            ('sup_ts', 'records_es', 'answer-key add-on, same recipe, early-stopped', 'sup'),
            ('feedback', 'records', 'Peekaboo, eight passes', 'pk'), ('feedback_1pass', 'records', 'Peekaboo, one pass (each row rated once)', 'pk1')]
    forest(axes[0], '0.8b', vega, 'Vega 0.8B')
    forest(axes[1], '4b', [(m, k, '', st) for m, k, _, st in vega], 'Vega 4B')
    forest(axes[2], 'laya', [('laya_td', 'records', '', 'theirs'), ('sup_1pass', 'records', 'answer-key add-on, one pass (Laya head)', 'sup_es'), ('feedback_1pass', 'records', '', 'pk1')], 'Laya, head only (one pass)')
    for ax in axes:
        ax.set_xlabel('accuracy gain over the released model (points)')
    h, l = axes[0].get_legend_handles_labels(); h2, l2 = axes[2].get_legend_handles_labels()
    h += [x for x, lab in zip(h2, l2) if lab]; l += [lab for lab in l2 if lab]
    fig.legend(h, l, frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.16, 1, 1)); fig.savefig(os.path.join(FIG, 'gains.pdf'), bbox_inches='tight'); plt.close(fig)


def fig_laya():
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    series = [('laya_td', 'records', "Laya's typed-decisions fine-tune (in-domain only on typed-decisions)", 'theirs'),
              ('sup_enc', 'records_es', 'answer-key fine-tune, encoder unfrozen, early-stopped', 'sup'),
              ('feedback_enc', 'records', 'Peekaboo, encoder unfrozen, eight passes', 'enc'), ('feedback_enc_1pass', 'records', 'Peekaboo, encoder unfrozen, one pass', 'enc1'),
              ('feedback_1pass', 'records', 'Peekaboo, head only, one pass', 'pk')]
    forest(ax, 'laya', series, 'Laya, encoder unfrozen')
    ax.set_xlabel('accuracy gain over released Laya (points)')
    ax.legend(frameon=False, ncol=1, loc='upper center', bbox_to_anchor=(0.5, -0.18), fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'laya.pdf'), bbox_inches='tight'); plt.close(fig)


def fig_passes():
    """One pass against eight, feedback and labels, per model and benchmark."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for ax, (model, mname) in zip(axes, MODELS):
        y = np.arange(len(SUITES))[::-1]
        for yi, s in zip(y, SUITES):
            pairs = ((('feedback_enc_1pass', 'feedback_enc', C['pk'], 'Peekaboo: one pass $\\rightarrow$ eight passes'),) if model == 'laya' else
                     (('feedback_1pass', 'feedback', C['pk'], 'Peekaboo: one pass $\\rightarrow$ eight passes'), ('sup_1pass', 'sup_ts', C['sup'], 'answer key: one pass $\\rightarrow$ eight passes')))
            for dy, (m1, m8, col, lab) in zip((0.18, -0.18), pairs):
                a, b = cmp(model, m1, 'records', s), cmp(model, m8, 'records', s)
                if a and b:
                    ax.plot([a['gain'], b['gain']], [yi + dy, yi + dy], '-', color=col, lw=1.2)
                    ax.plot(a['gain'], yi + dy, 'o', ms=4, mfc='white', color=col, label=lab if yi == y[0] else None)
                    ax.plot(b['gain'], yi + dy, 'o', ms=4, color=col)
        ax.axvline(0, color='k', lw=0.7); ax.set_yticks(y); ax.set_yticklabels([SNAME[s] for s in SUITES]); ax.set_title(mname + (', encoder unfrozen' if model == 'laya' else ''))
        ax.grid(axis='x', lw=0.4, alpha=0.5); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
        ax.set_xlabel('gain over the released model (points)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(FIG, 'passes.pdf'), bbox_inches='tight'); plt.close(fig)


def fig_realistic():
    arms = [('feedback', 'records', 'eight passes, clean', 'pk'), ('feedback_1pass', 'records', 'one pass, clean', 'pk1'),
            ('logged_argmax', 'records', 'logged: best answers rated once', 'logged'), ('logged_ips', 'records', 'logged: sampled answers rated once', 'logged'),
            ('feedback_noise20', 'records', '20% of verdicts flipped', 'kl0'), ('feedback_half', 'records', 'half of the answers rated', 'pooled'),
            ('feedback_complaints', 'records', 'all wrong answers reported, 20% of right ones', 'prior')]
    suites = [s for s in ('decisionbench', 'typed', 'emotion') if any(cmp(m, a[0], a[1], s) for m, _ in MODELS for a in arms[2:])]
    if not suites:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    y = np.arange(len(arms))[::-1]
    for ax, (model, mname) in zip(axes, MODELS[:2]):
        for j, s in enumerate(suites):
            dy = (j - (len(suites) - 1) / 2) * -0.22
            for yi, (meth, key, label, style) in zip(y, arms):
                g = cmp(model, meth, key, s)
                if g:
                    ax.errorbar(g['gain'], yi + dy, xerr=[[g['gain'] - g['lo']], [g['hi'] - g['gain']]], fmt='osD^v<>'[j], ms=4, color=plt.cm.tab10(j),
                                ecolor=plt.cm.tab10(j), elinewidth=0.9, capsize=1.5, label=SNAME[s] if yi == y[0] else None)
        ax.axvline(0, color='k', lw=0.7); ax.set_yticks(y); ax.set_yticklabels([a[2] for a in arms]); ax.set_title(mname)
        ax.grid(axis='x', lw=0.4, alpha=0.5); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0); ax.set_xlabel('gain over the released model (points)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=len(suites), loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, 'realistic.pdf'), bbox_inches='tight'); plt.close(fig)


def fig_anatomy():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ranks = ['fix_rate_rank2', 'fix_rate_rank3', 'fix_rate_rank4plus']; x = np.arange(len(ranks)); w = 0.13
    i = 0
    for model, mname in MODELS[:2]:
        for meth, label, col in (('feedback', 'Peekaboo', C['pk']), ('sup_ts', 'answer key', C['sup'])):
            vals = [np.nanmean([N['fixbreak'].get(f'{model}|{meth}|{s}', {}).get(r) or np.nan for s in SUITES if s != 'jevbench']) for r in ranks]
            axes[0].bar(x + (i - 1.5) * w, [100 * v for v in vals], w, color=col, alpha=1.0 if model == '0.8b' else 0.55, label=f'{mname}, {label}'); i += 1
    axes[0].set_xticks(x); axes[0].set_xticklabels(['gold ranked 2nd', 'ranked 3rd', 'ranked 4th or lower']); axes[0].set_ylabel('% of base-wrong rows fixed')
    axes[0].set_title('which errors get fixed'); axes[0].legend(frameon=False, fontsize=7)
    y = np.arange(len(SUITES))[::-1]
    for model, dy, alpha in (('0.8b', 0.2, 1.0), ('4b', -0.2, 0.55)):
        for yi, s in zip(y, SUITES):
            p = N['prior'].get(f'{model}|{s}')
            g = cmp(model, 'feedback', 'records', s); a = cmp(model, 'sup_ts', 'records', s)
            first = yi == y[0] and model == '0.8b'
            if p and p['feedback'].get('vs_released'):
                axes[1].plot(p['feedback']['vs_released']['gain'], yi + dy, '^', color=C['prior'], alpha=alpha, ms=5, label='label prior from verdicts' if first else None)
                axes[1].plot(p['labels']['vs_released']['gain'], yi + dy, '^', mfc='white', color=C['prior'], alpha=alpha, ms=5, label='label prior from labels' if first else None)
            if g:
                axes[1].plot(g['gain'], yi + dy, 'o', color=C['pk'], alpha=alpha, ms=4.5, label='Peekaboo' if first else None)
            if a:
                axes[1].plot(a['gain'], yi + dy, 's', color=C['sup'], alpha=alpha, ms=4.5, label='answer-key adapter' if first else None)
    axes[1].axvline(0, color='k', lw=0.7); axes[1].set_yticks(y); axes[1].set_yticklabels([SNAME[s] for s in SUITES]); axes[1].grid(axis='x', lw=0.4, alpha=0.5)
    axes[1].spines['left'].set_visible(False); axes[1].tick_params(axis='y', length=0); axes[1].set_xlabel('gain over the released model (points; solid 0.8B, faint 4B)')
    axes[1].set_title('how much a label prior alone recovers'); axes[1].legend(frameon=False, fontsize=7, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'anatomy.pdf'), bbox_inches='tight'); plt.close(fig)


def fig_calibration():
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
    bars = [('released', 'records', 'released', C['base']), ('baseT', 'feedback', 'released + verdict-fitted T', '#c0c0c0'),
            ('feedback', 'records_raw', 'Peekaboo before the fix', C['raw']), ('feedback', 'records', 'Peekaboo', C['pk']), ('sup_ts', 'records', 'answer key + label T', C['sup'])]
    w = 0.8 / len(bars); x = np.arange(len(SUITES))
    for col_i, (model, mname) in enumerate(MODELS):
        for row, what in enumerate(('ece', 'aurc')):
            ax = axes[row][col_i]
            for i, (meth, key, label, colr) in enumerate(bars):
                meth = {'feedback': PK(model), 'sup_ts': SUP(model)}.get(meth, meth)
                if meth == 'baseT':
                    y = [(N['base_T'].get(f'{model}|{s}') or {}).get(key, {}).get(what, np.nan) for s in SUITES]
                else:
                    y = [np.mean(sc(model, meth, key, s, what)) if sc(model, meth, key, s, what) else np.nan for s in SUITES]
                ax.bar(x + (i - (len(bars) - 1) / 2) * w, y, w, color=colr, label=label)
            ax.set_xticks(x); ax.set_xticklabels([SSHORT[s] for s in SUITES], rotation=25, ha='right'); ax.set_title(mname if row == 0 else '')
            ax.set_ylabel('ECE (lower is better)' if what == 'ece' else 'AURC (lower is better)')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=5, loc='lower center', bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(FIG, 'calibration.pdf'), bbox_inches='tight'); plt.close(fig)


def fig_reliability():
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.6))
    for row, s in enumerate(('decisionbench', 'typed')):
        for ax, (model, mname) in zip(axes[row], MODELS):
            for meth, key, label, col in (('released', 'records', 'released model', C['base']), (PK(model), 'records_raw', 'Peekaboo before the honesty fix', C['raw']), (PK(model), 'records', 'Peekaboo', C['pk'])):
                R = A.runs(model, s, meth, key)
                if not R:
                    continue
                conf = np.concatenate([[max(r['probs'].values()) if r.get('probs') else 0.0 for r in recs] for recs in R])
                corr = np.concatenate([A.correct(recs) for recs in R])
                edges = np.linspace(0, 1, 11); b = np.clip(np.digitize(conf, edges) - 1, 0, 9)
                pts = [(conf[b == i].mean(), corr[b == i].mean()) for i in range(10) if (b == i).sum() >= 10]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', ms=3, color=col, label=label)
            ax.plot([0, 1], [0, 1], '--', color='k', lw=0.6); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(f'{mname}, {SNAME[s]}')
            if row == 1:
                ax.set_xlabel('stated confidence')
            if ax is axes[row][0]:
                ax.set_ylabel('fraction correct')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(FIG, 'reliability.pdf'), bbox_inches='tight'); plt.close(fig)


def fig_switch_T():
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
    x = np.arange(len(SUITES)); w = 0.26
    for i, (model, mname) in enumerate(MODELS):
        Ts = [np.mean([f['T'] for inf in A.infos(model, s, PK(model)) for f in inf.get('folds', []) if f.get('T')] or [np.nan]) for s in SUITES]
        axes[0].bar(x + (i - 1) * w, Ts, w, label=mname, color=[C['pk'], C['enc'], C['pooled']][i])
        on = [(N['switch'].get(f'{model}|{PK(model)}|{s}') or {}).get('on', np.nan) for s in SUITES]
        so = [(N['switch'].get(f'{model}|{PK(model)}|{s}') or {}).get('sampled_on') or np.nan for s in SUITES]
        axes[1].bar(x + (i - 1) * w, on, w, label=mname, color=[C['pk'], C['enc'], C['pooled']][i])
        axes[1].plot(x + (i - 1) * w, so, 'k_', ms=9, mew=1.2, label='same switch on sampled answers (v1)' if i == 0 else None)
    axes[0].axhline(1, color='k', lw=0.6, ls='--'); axes[0].set_ylabel('fitted temperature T'); axes[0].set_title('honesty fix: temperature fitted from verdicts')
    axes[1].set_ylabel('share of runs keeping the add-on'); axes[1].set_title('switch: best answers compared'); axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([SSHORT[s] for s in SUITES], rotation=25, ha='right')
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(FIG, 'switch_T.pdf'), bbox_inches='tight'); plt.close(fig)


def fig_tasks_db():
    d = N['tasks_db']
    if '0.8b|feedback' not in d:
        return
    tasks = sorted(d['0.8b|feedback'], key=lambda t: d['0.8b|feedback'][t]['acc'] - d['0.8b|feedback'][t]['base'])
    fig, ax = plt.subplots(figsize=(10, 3.4))
    x = np.arange(len(tasks)); w = 0.4
    for i, (model, mname, col) in enumerate((('0.8b', 'Vega 0.8B', C['pk']), ('4b', 'Vega 4B', C['enc']))):
        if f'{model}|feedback' in d:
            ax.bar(x + (i - 0.5) * w, [100 * (d[f'{model}|feedback'][t]['acc'] - d[f'{model}|feedback'][t]['base']) for t in tasks], w, color=col, label=f'{mname}, Peekaboo')
    short = [t.split('/')[-1] + (' *' if any(a in t for a in ('contains_threat', 'toxicity_severity', 'finqa')) else '') for t in tasks]
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=75, ha='right', fontsize=6); ax.axhline(0, color='k', lw=0.6)
    ax.set_ylabel('gain over released Vega (points)'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'tasks_db.pdf'), bbox_inches='tight'); plt.close(fig)


def fig_benchmarks():
    import collections
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), gridspec_kw={'width_ratios': [1, 1.3]})
    y = np.arange(len(SUITES))[::-1]
    types = ['choice', 'boolean', 'score']; tn = {'choice': 'pick one', 'boolean': 'yes / no', 'score': 'rating scale'}
    counts = {s: collections.Counter(it['question'].get('type') for it in A.items(s)) for s in SUITES}
    left = np.zeros(len(SUITES))
    for t, col in zip(types, ('#4c72b0', '#55a868', '#c44e52')):
        v = np.array([counts[s].get(t, 0) for s in SUITES])
        axes[0].barh(y, v, left=left, color=col, label=tn[t], height=0.7); left += v
    axes[0].set_yticks(y); axes[0].set_yticklabels([SNAME[s] for s in SUITES]); axes[0].set_xlabel('test rows'); axes[0].set_title('test rows by question type')
    axes[0].legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.22))
    bins = [(2, 2, '2'), (3, 3, '3'), (4, 4, '4'), (5, 5, '5'), (6, 8, '6 to 8'), (9, 10 ** 9, '9 or more')]
    shades = plt.cm.Blues(np.linspace(0.3, 0.95, len(bins)))
    left = np.zeros(len(SUITES))
    for (lo, hi, lab), col in zip(bins, shades):
        v = np.array([np.mean([(lo <= len(it['labels']) <= hi) for it in A.items(s)]) if A.items(s) else 0 for s in SUITES])
        axes[1].barh(y, v, left=left, color=col, label=lab, height=0.7, edgecolor='white', lw=0.5)
        for yi, vi, li in zip(y, v, left):
            if vi >= 0.12:
                axes[1].text(li + vi / 2, yi, f'{100 * vi:.0f}%', ha='center', va='center', fontsize=7, color='white' if col[2] < 0.8 else 'black')
        left += v
    axes[1].set_yticks(y); axes[1].set_yticklabels([]); axes[1].set_xlim(0, 1); axes[1].set_xlabel('share of test rows'); axes[1].set_title('number of options per question')
    axes[1].tick_params(axis='y', length=0); axes[1].legend(frameon=False, ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.22), title='options', title_fontsize=8, handlelength=1.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'benchmarks.pdf'), bbox_inches='tight'); plt.close(fig)


# ───────────────────────────── tables ─────────────────────────────
def table(lines, name):
    open(os.path.join(HERE, name), 'w').write('\n'.join(lines) + '\n')


def num(x):
    return f'{x:,}'.replace(',', '{,}')


def tab_bench():
    pools = {s: len(A.items(s, 'pool')) for s in SUITES}
    rows = [('43 public tasks', 'Vega', num(len(A.items('public'))), num(pools['public']), "43 public datasets as typed questions, up to 60 test rows each; feedback: VegaML's 50 fit rows per task, drawn class by class; macro-averaged over tasks"),
            ('JevBench', 'Vega', num(len(A.items('jevbench'))), '(5-fold)', 'rules, policies and judgment calls in 18 families, paraphrase groups; no training split, every method cross-fits'),
            ('DecisionBench', 'Vega', num(len(A.items('decisionbench'))), num(pools['decisionbench']), "43 tasks in 28 domains; the README's 25-row slice per task is the test set, other rows of the same tasks the pool"),
            ('typed-decisions', 'Laya', num(len(A.items('typed'))), num(pools['typed']), "400 test and 1,200 training cases, five decisions each; the pool is the public training split, also the training data of Vega's shipped adapter"),
            ('emotion', '(ours)', num(len(A.items('emotion'))), num(pools['emotion']), 'six emotions of a tweet; pool: training rows drawn at random (natural class mix)'),
            ('AG News', '(ours)', num(len(A.items('agnews'))), num(pools['agnews']), 'four news sections; pool: training rows drawn at random')]
    L = ['\\begin{table}[t]\\centering\\small', '\\caption{The six benchmarks. ``Feedback rows\'\' are the rows whose verdicts a method may learn from (80\\%) and calibrate on (20\\%); no test row is ever used for learning.}',
         '\\label{tab:bench}', '\\setlength{\\tabcolsep}{4pt}', '\\begin{tabular}{llrrp{7.4cm}}', '\\toprule', 'benchmark & reported by & test rows & feedback rows & rows and pool \\\\', '\\midrule']
    for r in rows:
        L.append(' & '.join(r) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    table(L, 'tables_bench.tex')


def tab_main():
    rows_v = [('released', 'records', 'released model', 'none'), ('ttt', 'records', "TTT head, 0.8.0 rule", 'answer key'), ('ttt', 'records_guard', "TTT head, 0.12.3 rule", 'answer key'),
              ('sup_ts', 'records', 'answer-key adapter, same recipe', 'answer key'), ('sup_ts', 'records_es', 'answer-key adapter, early-stopped', 'answer key'), ('sup_1pass', 'records', 'answer-key adapter, one pass', 'answer key'),
              ('feedback', 'records', '\\textbf{Peekaboo}, eight passes', 'verdicts'), ('feedback_1pass', 'records', '\\textbf{Peekaboo}, one pass', 'verdicts'),
              ('logged_argmax', 'records', 'logged best answers, rated once', 'verdicts'), ('logged_ips', 'records', 'logged sampled answers, rated once', 'verdicts'), ('feedback_kl0', 'records', 'Peekaboo, no KL term', 'verdicts')]
    rows_l = [('released', 'records', 'released model', 'none'), ('laya_td', 'records', "laya-typed-decisions (fine-tune on typed-decisions)", 'answer key'),
              ('sup_ts', 'records', 'answer-key head, same recipe', 'answer key'), ('sup_ts', 'records_es', 'answer-key head, early-stopped', 'answer key'), ('sup_1pass', 'records', 'answer-key head, one pass', 'answer key'),
              ('feedback', 'records', '\\textbf{Peekaboo}, head only, eight passes', 'verdicts'), ('feedback_1pass', 'records', '\\textbf{Peekaboo}, head only, one pass', 'verdicts'),
              ('sup_enc', 'records', 'answer-key fine-tune, encoder unfrozen', 'answer key'), ('sup_enc', 'records_es', 'answer-key fine-tune, encoder unfrozen, early-stopped', 'answer key'),
              ('feedback_enc', 'records', '\\textbf{Peekaboo}, encoder unfrozen, eight passes', 'verdicts'), ('feedback_enc_1pass', 'records', '\\textbf{Peekaboo}, encoder unfrozen, one pass', 'verdicts'),
              ]
    for what, caption, label in (('accuracy', 'Accuracy, mean over seeds ($\\pm$ s.d.; three seeds except where Section~\\ref{sec:limitations} lists fewer)', 'tab:acc'),
                                 ('gain', 'Gain over the released model in accuracy points; $\\dagger$ marks differences that survive Holm adjustment within the method\'s family. The 95\\% intervals are in Figure~\\ref{fig:gains} and Table~\\ref{tab:h2h}', 'tab:gain')):
        L = ['\\begin{table}[tbp]\\centering\\scriptsize', f'\\caption{{{caption}. JevBench has no training split: every method learns by 5-fold cross-fitting there.}}', f'\\label{{{label}}}',
             '\\setlength{\\tabcolsep}{2.5pt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{llr' + 'r' * len(SUITES) + '}', '\\toprule',
             'model & method & learns from & ' + ' & '.join(SSHORT[s] for s in SUITES) + ' \\\\', '\\midrule']
        for model, mname in MODELS:
            first = True
            for meth, key, name, src in (rows_l if model == 'laya' else rows_v):
                cells = []
                for s in SUITES:
                    if what == 'accuracy':
                        v = sc(model, meth, key, s, 'accuracy'); cells.append(f3(v) if v else '$\\cdot$')
                    else:
                        g = cmp(model, meth, key, s); cells.append('--' if meth == 'released' else (fg(g, ci=False) if g else '$\\cdot$'))
                if all(c == '$\\cdot$' for c in cells):
                    continue
                L.append(f'{mname if first else ""} & {name} & {src} & ' + ' & '.join(cells) + ' \\\\'); first = False
            if model != 'laya':
                L.append('\\addlinespace')
        L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
        table(L, f'tables_{"acc" if what == "accuracy" else "gain"}.tex')


def tab_h2h():
    L = ['\\begin{table}[tbp]\\centering\\scriptsize', '\\caption{Head to head, in accuracy points with 95\\% intervals (seeds and row clusters resampled). Positive means the first method is better. '
         '$\\dagger$: survives Holm adjustment within the row\'s family; NI: non-inferior at a 2-point margin (lower bound above $-2$). TTT is scored under the vegaml 0.12.3 rule; for Laya ``their method\'\' is the typed-decisions fine-tune, in-domain only on typed-decisions. The Laya head-only one-pass arms barely move from the released model (Section~\\ref{sec:laya}), so no NI marks are given for them.}',
         '\\label{tab:h2h}', '\\setlength{\\tabcolsep}{3pt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{ll' + 'r' * len(SUITES) + '}', '\\toprule',
         'model & comparison & ' + ' & '.join(SSHORT[s] for s in SUITES) + ' \\\\']
    rows = [('feedback', 'records', 'ttt', 'records_guard', 'Peekaboo (8 passes) vs. their method'), ('feedback', 'records', 'sup_ts', 'records', 'Peekaboo (8) vs. answer key, same recipe (8)'),
            ('feedback', 'records', 'sup_ts', 'records_es', 'Peekaboo (8) vs. answer key, early-stopped'), ('feedback_1pass', 'records', 'sup_1pass', 'records', 'Peekaboo (1 pass) vs. answer key (1 pass)'),
            ('feedback_1pass', 'records', 'feedback', 'records', 'Peekaboo, 1 pass vs. 8 passes'), ('logged_ips', 'records', 'feedback_1pass', 'records', 'logged sampled answers vs. Peekaboo (1 pass)'),
            ('logged_ips', 'records', 'feedback', 'records', 'logged sampled answers vs. Peekaboo (8)'), ('logged_ips', 'records', 'sup_ts', 'records_es', 'logged sampled answers vs. answer key, early-stopped'),
            ('logged_argmax', 'records', 'sup_ts', 'records_es', 'logged best answers vs. answer key, early-stopped'),
            ('feedback', 'records', 'feedback_kl0', 'records', 'KL term vs. none'), ('feedback', 'records_switch', 'feedback', 'records', 'with the switch vs. without')]
    rows_l = [('feedback', 'records', 'laya_td', 'records', 'Peekaboo, head (8) vs. their fine-tune'), ('feedback', 'records', 'sup_ts', 'records_es', 'Peekaboo, head (8) vs. answer-key head, early-stopped'),
              ('feedback_1pass', 'records', 'sup_1pass', 'records', 'Peekaboo, head (1) vs. answer-key head (1)'),
              ('feedback_enc', 'records', 'laya_td', 'records', 'Peekaboo, encoder (8) vs. their fine-tune'), ('feedback_enc', 'records', 'sup_enc', 'records', 'Peekaboo, encoder (8) vs. answer-key encoder, same recipe'),
              ('feedback_enc', 'records', 'sup_enc', 'records_es', 'Peekaboo, encoder (8) vs. answer-key encoder, early-stopped'), ('feedback_enc_1pass', 'records', 'feedback_enc', 'records', 'Peekaboo, encoder: 1 pass vs. 8 passes'),
              ('feedback_enc', 'records', 'feedback', 'records', 'encoder unfrozen vs. head only (8 passes)')]
    for model, mname in MODELS:
        L += ['\\midrule', f'\\multicolumn{{{2 + len(SUITES)}}}{{l}}{{\\textit{{{mname}}}}} \\\\']
        for a, ka, b, kb, name in (rows_l if model == 'laya' else rows):
            cells = []
            for s in SUITES:
                g = N['compare'].get(f'{model}|{a}|{ka}|{s}|vs|{b}|{kb}')
                cells.append((fg(g) + (' NI' if g.get('non_inferior_2') and 'answer' in name and 'head' not in name else '')) if g else '--')
            if any(c != '--' for c in cells):
                L.append(f' & {name} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_h2h.tex')


def tab_realistic():
    arms = [('feedback', 'eight passes, clean verdicts'), ('feedback_1pass', 'one pass, clean verdicts'), ('logged_argmax', "logged: the released model's best answers, rated once"),
            ('logged_ips', "logged: the released model's sampled answers, rated once (clipped IPS)"), ('feedback_noise20', '20\\% of verdicts flipped (eight passes)'),
            ('feedback_half', 'half of the answers rated (eight passes)'), ('feedback_complaints', 'every wrong answer reported, 20\\% of right ones (eight passes)'), ('sup_1pass', 'answer key, one pass'), ('sup_ts', 'answer key, eight passes')]
    suites = ['decisionbench', 'typed', 'emotion']
    L = ['\\begin{table}[tbp]\\centering\\small', '\\caption{Feedback as a deployment delivers it: gain over the released model in points (95\\% intervals), on the three benchmarks with the realistic-feedback arms.}', '\\label{tab:realistic}',
         '\\setlength{\\tabcolsep}{3pt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{l' + 'r' * (3 * len(suites)) + '}', '\\toprule',
         'feedback & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{SNAME[s]}}}' for s in suites) + ' \\\\', ' & ' + ' & '.join('Vega 0.8B & Vega 4B & Laya (head)' for _ in suites) + ' \\\\', '\\midrule']
    for meth, name in arms:
        cells = [fg(cmp(m, meth, 'records', s), ci=True, mark=False) for s in suites for m, _ in MODELS]
        if any(c != '--' for c in cells):
            L.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_realistic.tex')


def tab_calibration():
    L = ['\\begin{table}[tbp]\\centering\\scriptsize', '\\caption{Calibration, proper scores and selective risk (lower is better; mean over seeds). ``+ T\'\' rows apply one temperature to the released model, fitted on its own calibration rows from verdicts on sampled answers (mean over 10 draws) or from labels; JevBench has no calibration pool for them.}',
         '\\label{tab:cal}', '\\setlength{\\tabcolsep}{2.5pt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{ll' + 'rrrr' * len(SUITES) + '}', '\\toprule',
         'model & method & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{SSHORT[s]}}}' for s in SUITES) + ' \\\\', ' & & ' + ' & '.join('ECE & NLL & Brier & AURC' for _ in SUITES) + ' \\\\']
    rows = [('released', 'records', 'released model'), ('baseT', 'feedback', 'released + T from verdicts'), ('baseT', 'labels', 'released + T from labels'), ('feedback', 'records_raw', 'Peekaboo before the honesty fix'),
            ('feedback', 'records', 'Peekaboo'), ('feedback_1pass', 'records', 'Peekaboo, one pass'), ('sup_ts', 'records', 'answer key + label temperature'), ('feedback_enc', 'records', 'Peekaboo, encoder unfrozen'), ('sup_enc', 'records', 'answer-key encoder + label temperature')]
    for model, mname in MODELS:
        L += ['\\midrule', f'\\multicolumn{{{2 + 4 * len(SUITES)}}}{{l}}{{\\textit{{{mname}}}}} \\\\']
        for meth, key, name in rows:
            cells = []
            for s in SUITES:
                if meth == 'baseT':
                    v = (N['base_T'].get(f'{model}|{s}') or {}).get(key)
                    cells += [f'{v[w]:.3f}' if v and v.get(w) is not None else '--' for w in ('ece', 'nll', 'brier', 'aurc')]
                else:
                    cells += [f'{np.mean(sc(model, meth, key, s, w)):.3f}' if sc(model, meth, key, s, w) else '--' for w in ('ece', 'nll', 'brier', 'aurc')]
            if any(c != '--' for c in cells):
                L.append(f' & {name} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_cal.tex')


def tab_anatomy():
    L = ['\\begin{table}[tbp]\\centering\\scriptsize', '\\caption{What the gain is made of. Left: gain over the released model of a per-task label prior with no input features (log-offsets on the released probabilities), trained from verdicts with Peekaboo\'s loop and from labels; and Peekaboo minus that prior. '
         'Right: how often eight on-policy passes reveal the gold answer: reward-1 events per training row (each one names the gold answer), the share of rows the base policy would reveal in 8 draws, the share of two-option rows, and the first pass\'s reward against the released model\'s best-answer accuracy on the same rows.}',
         '\\label{tab:anatomy}', '\\setlength{\\tabcolsep}{2.5pt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrr|rrrrr}', '\\toprule',
         'model & benchmark & prior (verdicts) & prior (labels) & Peekaboo $-$ prior & params & reveals / row & $P$(revealed, 8 draws) & 2-option rows & pass-1 reward & base best-answer acc. \\\\']
    for model, mname in MODELS:
        L += ['\\midrule']
        first = True
        for s in SUITES:
            p = N['prior'].get(f'{model}|{s}'); r = N['reveal'].get(f'{model}|feedback|{s}')
            if not p and not r:
                continue
            c = [fg(p['feedback']['vs_released'], ci=False, mark=False) if p else '--', fg(p['labels']['vs_released'], ci=False, mark=False) if p else '--',
                 fg(p['feedback']['peekaboo_vs'], ci=True, mark=False) if p and p['feedback'].get('peekaboo_vs') else '--', str(p['feedback']['params']) if p else '--',
                 f"{r['reveals_per_row']:.1f}" if r and r.get('reveals_per_row') else '--', f"{r['p_reveal_8_base_policy']:.2f}" if r and r.get('p_reveal_8_base_policy') is not None else '--',
                 f"{100 * r['binary_share']:.0f}\\%" if r and r.get('binary_share') is not None else '--', f"{r['pass1_reward']:.2f}" if r and r.get('pass1_reward') else '--',
                 f"{r['base_argmax_acc_train']:.2f}" if r and r.get('base_argmax_acc_train') is not None else '--']
            L.append(f'{mname if first else ""} & {SNAME[s]} & ' + ' & '.join(c) + ' \\\\'); first = False
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_anatomy.tex')
    L = ['\\begin{table}[tbp]\\centering\\small', '\\caption{Which rows change. Share of the released model\'s wrong rows that a method fixes, by the released model\'s rank of the gold option, and share of its right rows the method breaks (mean over benchmarks with a training split and over seeds).}',
         '\\label{tab:fixbreak}', '\\begin{tabular}{llrrrrr}', '\\toprule', 'model & method & fixed, all & gold 2nd & gold 3rd & gold 4th+ & broken \\\\', '\\midrule']
    for model, mname in MODELS:
        for meth, name in (('feedback', 'Peekaboo (8 passes)'), ('feedback_1pass', 'Peekaboo (1 pass)'), ('sup_ts', 'answer key (8 passes)'), ('feedback_enc', 'Peekaboo, encoder (8)'), ('sup_enc', 'answer-key encoder (8)')):
            vals = [N['fixbreak'].get(f'{model}|{meth}|{s}') for s in SUITES if s != 'jevbench']
            vals = [v for v in vals if v]
            if not vals:
                continue
            m = lambda k: f"{100 * np.nanmean([v[k] if v.get(k) is not None else np.nan for v in vals]):.0f}\\%"
            L.append(f'{mname} & {name} & {m("fix_rate")} & {m("fix_rate_rank2")} & {m("fix_rate_rank3")} & {m("fix_rate_rank4plus")} & {m("break_rate")} \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    table(L, 'tables_fixbreak.tex')
    L = ['\\begin{table}[tbp]\\centering\\small', '\\caption{VegaML\'s TTT head answers a row only where a head was kept for its task and the option set matches (``covered\'\'); elsewhere its output is the released engine\'s. Eight-pass Peekaboo minus TTT, and the answer-key adapter minus TTT, on covered and uncovered rows, under the vegaml 0.12.3 rule used in the main comparison; rows covered under the 0.8.0 rule for reference.}',
         '\\label{tab:ttt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrrr}', '\\toprule', 'model & benchmark & covered (0.12.3) & covered (0.8.0) & Peekaboo $-$ TTT, covered & uncovered & answer key $-$ TTT, covered \\\\', '\\midrule']
    for model, mname in MODELS[:2]:
        for s in SUITES:
            t = N['ttt'].get(f'{model}|{s}'); t0 = N['ttt'].get(f'{model}|{s}|v080') or {}
            if t:
                L.append(f'{mname} & {SNAME[s]} & {t["rows_covered"]} / {t["rows"]} & {t0.get("rows_covered", "--")} & {fg(t.get("peekaboo_vs_ttt_covered"), mark=False)} & {fg(t.get("peekaboo_vs_ttt_uncovered"), mark=False)} & {fg(t.get("adapter_vs_ttt_covered"), mark=False)} \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_ttt.tex')
    L = ['\\begin{table}[tbp]\\centering\\small', '\\caption{Benchmark artifacts. Gain over the released model with every task, and without the five tasks whose gold answer an adapter can learn without reading the input (public: truthfulqa\\_mc1, pubmedqa; DecisionBench: contains\\_threat, toxicity\\_severity, finqa-numerical-reasoning).}',
         '\\label{tab:artifacts}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrr}', '\\toprule', 'model & method & public, all 43 & public, 41 tasks & DecisionBench, all 43 & DecisionBench, 40 tasks \\\\', '\\midrule']
    for model, mname in MODELS:
        for meth, key, name in (('feedback', 'records', 'Peekaboo (8)'), ('feedback_1pass', 'records', 'Peekaboo (1)'), ('sup_ts', 'records', 'answer key'), ('ttt', 'records_guard', 'TTT (0.12.3)'), ('feedback_enc', 'records', 'Peekaboo, encoder'), ('laya_td', 'records', 'their fine-tune')):
            c = [fg(cmp(model, meth, key, 'public'), mark=False), fg(cmp(model, meth, key, 'public', suffix='|noartifact'), mark=False),
                 fg(cmp(model, meth, key, 'decisionbench'), mark=False), fg(cmp(model, meth, key, 'decisionbench', suffix='|noartifact'), mark=False)]
            if any(x != '--' for x in c):
                L.append(f'{mname} & {name} & ' + ' & '.join(c) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_artifacts.tex')


def tab_noship():
    L = ['\\begin{table}[tbp]\\centering\\small', '\\caption{typed-decisions on Vega with the shipped \\texttt{typed\\_decisions} adapter, which was fitted with labels on this feedback pool, switched off. Accuracy (mean over seeds) and gain over the corresponding released model.}',
         '\\label{tab:noship}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrr}', '\\toprule', 'model & method & accuracy, adapters on & gain & accuracy, adapters off & gain \\\\', '\\midrule']
    for model, mname in MODELS[:2]:
        for on, off, name in (('released', 'noship_released', 'released engine'), ('sup_ts', 'sup_ts_noship', 'answer-key adapter (8)'), ('feedback', 'feedback_noship', 'Peekaboo (8)'), ('feedback_1pass', 'feedback_1pass_noship', 'Peekaboo (1)')):
            a_on = sc(model, on, 'records', 'typed', 'accuracy'); a_off = sc(model, off, 'records', 'typed', 'accuracy')
            g_on = cmp(model, on, 'records', 'typed') if on != 'released' else None
            g_off = N['compare'].get(f'{model}|{off}|records|typed|vs|noship_released|records') if off != 'noship_released' else None
            if a_on or a_off:
                L.append(f'{mname} & {name} & {f3(a_on)} & {fg(g_on, mark=False) if g_on else "--"} & {f3(a_off)} & {fg(g_off, mark=False) if g_off else "--"} \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_noship.tex')


def tab_tasks():
    cols = [('0.8b', 'released', 'records', 'base'), ('0.8b', 'ttt', 'records_guard', 'TTT'), ('0.8b', 'sup_ts', 'records_es', 'key'), ('0.8b', 'feedback', 'records', 'Pk8'), ('0.8b', 'feedback_1pass', 'records', 'Pk1'),
            ('4b', 'released', 'records', 'base'), ('4b', 'ttt', 'records_guard', 'TTT'), ('4b', 'sup_ts', 'records_es', 'key'), ('4b', 'feedback', 'records', 'Pk8'), ('4b', 'feedback_1pass', 'records', 'Pk1'),
            ('laya', 'released', 'records', 'base'), ('laya', 'laya_td', 'records', 'f-t'), ('laya', 'feedback', 'records', 'Pk8'), ('laya', 'sup_enc', 'records_es', 'key-enc'), ('laya', 'feedback_enc', 'records', 'Pk-enc')]
    tg = A.task_groups('public')
    acc = {}
    for m, meth, key, lab in cols:
        R = A.runs(m, 'public', meth, key)
        if R:
            c = np.mean([A.correct(r) for r in R], 0); acc[(m, meth, key)] = {t: float(c[g].mean()) for t, g in tg.items()}
    cols = [c for c in cols if (c[0], c[1], c[2]) in acc]
    L = ['\\begin{table}[tbp]\\centering\\scriptsize', '\\caption{Accuracy on each of the 43 public tasks (mean over seeds). key: the early-stopped answer-key adapter / head; Pk8, Pk1: Peekaboo with eight passes, one pass; f-t: laya-typed-decisions; -enc: encoder unfrozen. $^*$ marks the two tasks with a construction artifact. Bold: the best method for that model on that task.}',
         '\\label{tab:tasks}', '\\setlength{\\tabcolsep}{2pt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{l' + 'r' * len(cols) + '}', '\\toprule',
         'task & ' + ' & '.join(f'\\multicolumn{{{sum(1 for c in cols if c[0] == m)}}}{{c}}{{{mn}}}' for m, mn in MODELS if any(c[0] == m for c in cols)) + ' \\\\', ' & ' + ' & '.join(c[3] for c in cols) + ' \\\\', '\\midrule']
    for t in sorted(tg):
        row = []
        for m, _ in MODELS:
            vals = [acc[(c[0], c[1], c[2])].get(t) for c in cols if c[0] == m]
            best = max(v for v in vals if v is not None) if any(v is not None for v in vals) else None
            row += [('' if v is None else (f'\\textbf{{{v:.2f}}}' if abs(v - best) < 5e-4 else f'{v:.2f}')) for v in vals]
        L.append(t.replace('_', '\\_') + ('$^*$' if t in A.ARTIFACT['public'] else '') + ' & ' + ' & '.join(row) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_tasks.tex')
    d = N['tasks_db']
    if '0.8b|feedback' in d:
        L = ['\\begin{table}[tbp]\\centering\\scriptsize', '\\caption{DecisionBench per task (25 test rows each): accuracy of the released model and gain in points of Peekaboo (8 passes), the answer-key adapter and TTT (0.8.0 rule). $^*$: tasks with a construction artifact.}',
             '\\label{tab:tasks_db}', '\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{lrrrrrrrr}', '\\toprule', 'task & \\multicolumn{4}{c}{Vega 0.8B} & \\multicolumn{4}{c}{Vega 4B} \\\\', ' & base & Pk & key & TTT & base & Pk & key & TTT \\\\', '\\midrule']
        for t in sorted(d['0.8b|feedback']):
            c = []
            for m in ('0.8b', '4b'):
                b = d.get(f'{m}|feedback', {}).get(t, {}).get('base')
                c.append(f'{b:.2f}' if b is not None else '')
                for meth in ('feedback', 'sup_ts', 'ttt'):
                    v = d.get(f'{m}|{meth}', {}).get(t)
                    c.append(f'{100 * (v["acc"] - v["base"]):+.0f}' if v else '')
            star = '$^*$' if any(a in t for a in ('contains_threat', 'toxicity_severity', 'finqa')) else ''
            L.append(t.split('/')[-1].replace('_', '\\_') + star + ' & ' + ' & '.join(c) + ' \\\\')
        L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
        table(L, 'tables_tasks_db.tex')


def tex_numbers():
    tr = {'0.8b': 'Small', '4b': 'Big', 'laya': 'Laya', 'released': 'Base', 'ttt': 'Ttt', 'sup_ts': 'Sup', 'sup_1pass': 'SupOne', 'sup_enc': 'SupEnc', 'feedback': 'Pk', 'feedback_1pass': 'PkOne',
          'feedback_enc': 'PkEnc', 'feedback_enc_1pass': 'PkEncOne', 'pooled': 'Pooled', 'feedback_kl0': 'Klzero', 'laya_td': 'Td', 'logged_argmax': 'LogArg', 'logged_ips': 'LogIps',
          'feedback_noise20': 'Noise', 'feedback_half': 'Half', 'feedback_complaints': 'Compl', 'noship_released': 'NoshipBase', 'feedback_noship': 'PkNoship', 'sup_ts_noship': 'SupNoship',
          'feedback_1pass_noship': 'PkOneNoship', 'public': 'Pub', 'jevbench': 'Jev', 'decisionbench': 'Db', 'typed': 'Typed', 'emotion': 'Emo', 'agnews': 'Ag',
          'records': '', 'records_switch': 'Sw', 'records_raw': 'Raw', 'records_es': 'Es', 'records_guard': 'Guard', 'noartifact': 'Clean'}
    nm = lambda *p: ''.join(tr.get(x, x.title()) for x in p)
    L = ['% generated by paper/make_figures.py; every number in the text comes from here']
    seen = set()

    def add(name, val):
        if name not in seen:
            seen.add(name); L.append(f'\\newcommand{{\\{name}}}{{{val}}}')
    for k, v in N['compare'].items():
        if not v:
            continue
        parts = k.split('|')
        model, meth, key, s = parts[:4]
        if parts[5] == 'released':
            base = nm(model, meth, key, s) + ('Clean' if parts[-1] == 'noartifact' else '')
        else:
            base = nm(model, meth, key, s) + 'Vs' + nm(parts[5], parts[6])
        add('gain' + base, f'{v["gain"]:+.1f}'); add('ci' + base, f'[{v["lo"]:+.1f}, {v["hi"]:+.1f}]'); add('p' + base, f'{v["p"]:.3f}')
        add('abs' + base, f'{abs(v["gain"]):.1f}')
    for k, v in N['scores'].items():
        model, meth, key, s = k.split('|')
        for w in ('accuracy', 'ece', 'nll', 'aurc', 'brier'):
            if v.get(w) and v[w][0] is not None:
                add(('acc' if w == 'accuracy' else w) + nm(model, meth, key, s), f'{np.mean(v[w]):.3f}')
    for k, v in N['base_T'].items():
        model, s = k.split('|')
        for how in ('feedback', 'labels'):
            add('eceBaseT' + ('Fb' if how == 'feedback' else 'Lab') + nm(model, s), f'{v[how]["ece"]:.3f}'); add('TBaseT' + ('Fb' if how == 'feedback' else 'Lab') + nm(model, s), f'{v[how]["T"]:.1f}')
    for k, v in N['prior'].items():
        model, s = k.split('|')
        for how in ('feedback', 'labels'):
            add('gainPrior' + ('Fb' if how == 'feedback' else 'Lab') + nm(model, s), f'{v[how]["vs_released"]["gain"]:+.1f}')
            if v[how].get('peekaboo_vs'):
                add('gainPkVsPrior' + ('Fb' if how == 'feedback' else 'Lab') + nm(model, s), f'{v[how]["peekaboo_vs"]["gain"]:+.1f}')
                add('ciPkVsPrior' + ('Fb' if how == 'feedback' else 'Lab') + nm(model, s), f'[{v[how]["peekaboo_vs"]["lo"]:+.1f}, {v[how]["peekaboo_vs"]["hi"]:+.1f}]')
    for k, v in N['reveal'].items():
        model, meth, s = k.split('|')
        for w, name in (('reveals_per_row', 'reveals'), ('p_reveal_8_base_policy', 'preveal'), ('pass1_reward', 'passone'), ('base_argmax_acc_train', 'baseargmax'), ('binary_share', 'binary'), ('max_pass_reward', 'maxpass')):
            if v.get(w) is not None:
                add(name + nm(model, meth, s), f'{v[w]:.2f}')
    for k, v in N['ttt'].items():
        parts = k.split('|'); model, s = parts[:2]; old = 'Vzero' if len(parts) == 3 else ''
        add('tttCovered' + old + nm(model, s), str(v['rows_covered']))
        if old:
            continue
        for w in ('peekaboo_vs_ttt_covered', 'peekaboo_vs_ttt_uncovered', 'adapter_vs_ttt_covered'):
            if v.get(w):
                add('gain' + ''.join(x.title() for x in w.split('_')) + nm(model, s), f'{v[w]["gain"]:+.1f}'); add('ci' + ''.join(x.title() for x in w.split('_')) + nm(model, s), f'[{v[w]["lo"]:+.1f}, {v[w]["hi"]:+.1f}]')
    for k, v in N['switch'].items():
        model, meth, s = k.split('|')
        add('swOn' + nm(model, meth, s), f'{100 * v["on"]:.0f}')
        if v.get('sampled_on') is not None:
            add('swOnSampled' + nm(model, meth, s), f'{100 * v["sampled_on"]:.0f}')
    for k, v in N['fixbreak'].items():
        model, meth, s = k.split('|')
        for w, name in (('fix_rate', 'Fix'), ('break_rate', 'Break'), ('fix_rate_rank2', 'FixRankTwo'), ('fix_rate_rank3', 'FixRankThree'), ('fix_rate_rank4plus', 'FixRankFour')):
            if v.get(w) is not None:
                add(name + nm(model, meth, s), f'{100 * v[w]:.0f}')
    for model in ('0.8b', '4b', 'laya'):
        for meth in ('feedback', 'feedback_1pass', 'sup_ts', 'feedback_enc', 'sup_enc'):
            vals = [N['fixbreak'].get(f'{model}|{meth}|{s}') for s in SUITES if s != 'jevbench']
            vals = [x for x in vals if x]
            if vals:
                for w, name in (('fix_rate', 'Fix'), ('break_rate', 'Break'), ('fix_rate_rank2', 'FixRankTwo'), ('fix_rate_rank4plus', 'FixRankFour')):
                    add(name + 'Mean' + nm(model, meth), f'{100 * np.nanmean([x[w] if x.get(w) is not None else np.nan for x in vals]):.0f}')
    if N.get('t_match'):
        add('TmatchSame', str(N['t_match']['same'])); add('TmatchNear', str(N['t_match']['within_one_step'])); add('TmatchRuns', str(N['t_match']['runs']))
    if N.get('T_range'):
        add('Tmin', f'{N["T_range"]["min"]:.1f}'); add('Tmax', f'{N["T_range"]["max"]:.1f}'); add('Tmedian', f'{N["T_range"]["median"]:.1f}')
    for k, v in N['balanced'].items():
        model, meth, key, s = k.split('|')
        add('bal' + nm(model, meth, key, s), f'{np.mean(v):.3f}')
    # a few numbers from the first version's grid (output/analysis.json), prefixed vone: Laya head-only eight-pass arms
    v1 = os.path.join(PR, 'output', 'analysis.json')
    if os.path.exists(v1):
        N1 = json.load(open(v1))
        for k, v in N1['compare'].items():
            parts = k.split('|')
            if v and parts[0] == 'laya' and parts[1] in ('feedback', 'sup_ts', 'feedback_kl0', 'pooled') and parts[2] == 'records' and parts[5] == 'released' and len(parts) == 6:
                add('gainVone' + nm(*parts[:4]), f'{v["gain"]:+.1f}'); add('ciVone' + nm(*parts[:4]), f'[{v["lo"]:+.1f}, {v["hi"]:+.1f}]')
            if v and parts[0] == 'laya' and parts[1] == 'feedback' and parts[2] == 'records' and len(parts) == 7 and parts[5] in ('sup_ts', 'laya_td') and parts[6] == 'records':
                add('gainVone' + nm(*parts[:4]) + 'Vs' + nm(parts[5]), f'{v["gain"]:+.1f}'); add('ciVone' + nm(*parts[:4]) + 'Vs' + nm(parts[5]), f'[{v["lo"]:+.1f}, {v["hi"]:+.1f}]')
        for k, v in N1['prior'].items():
            model, s = k.split('|')
            if model == 'laya':
                add('gainVonePriorFb' + nm(model, s), f'{v["feedback"]["vs_released"]["gain"]:+.1f}')
        for k, v in N1['scores'].items():
            model, meth, key, s = k.split('|')
            if model == 'laya' and meth in ('feedback', 'sup_ts') and key == 'records' and v.get('accuracy'):
                add('accVone' + nm(model, meth, key, s), f'{np.mean(v["accuracy"]):.3f}')
    open(os.path.join(HERE, 'numbers.tex'), 'w').write('\n'.join(L) + '\n')
    return len(L)


if __name__ == '__main__':
    fig_benchmarks(); fig_gains(); fig_laya(); fig_passes(); fig_realistic(); fig_anatomy(); fig_calibration(); fig_reliability(); fig_switch_T(); fig_tasks_db()
    tab_bench(); tab_main(); tab_h2h(); tab_realistic(); tab_calibration(); tab_anatomy(); tab_noship(); tab_tasks()
    n = tex_numbers()
    print('wrote figures, tables and', n, 'macros from', OUT)
