"""results/*.json -> figs/*.png + results/tables.md"""
import glob, json, os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lpol.kuhn import Kuhn

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, 'results'), os.path.join(HERE, 'figs')
os.makedirs(FIG, exist_ok=True)
COL = {'table': '#3a8a4a', 'mlp': '#3a6ea5', 'laya': '#c0483a', 'laya0': '#e0a090'}
NAME = {'table': 'tabular (from scratch)', 'mlp': 'small MLP (from scratch)', 'laya': 'Laya head + RL', 'laya0': 'Laya zero-shot'}
runs = [json.load(open(p)) for p in glob.glob(os.path.join(RES, '*.json'))]
G = defaultdict(list)
for r in runs:
    a = r['args']
    G[(a['env'], a['brain'], a['pool']) if a['env'] == 'kuhn' else (a['env'], a['brain'], a['pool'], a.get('tag', ''))].append(r)
md = []
k3, k4 = Kuhn(3), Kuhn(4)


def avg_strategy(hist, key='strategy'):
    """Time-averaged strategy at each eval point (fictitious-play style average)."""
    out, acc = [], None
    for i, h in enumerate(hist):
        s = h[key]
        acc = dict(s) if acc is None else {k: acc[k] + (s[k] - acc[k]) / (i + 1) for k in s}
        out.append(dict(acc))
    return out


def rank_map_4(s3):
    """Hand-made transfer for policies without language: lowest->J, highest->K, the rest->Q."""
    m = {'J': 'J', 'Q': 'Q', 'K': 'Q', 'A': 'K'}
    return {c + h: s3[m[c] + h] for c in k4.cards for h in ['', 'c', 'b', 'cb']}


def at(xs, ys, x):
    """mean ± sd across seeds of the eval closest to (not past) x episodes."""
    j = max(i for i, v in enumerate(xs) if v <= x)
    v = [y[j] for y in ys if len(y) > j]
    return f"{np.mean(v):+.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:+.3f}"


def band(ax, xs, ys, c, label, ls='-'):
    L = min(map(len, ys)); Y = np.array([y[:L] for y in ys], float); x = xs[:L]
    m = Y.mean(0); ax.plot(x, m, ls, color=c, label=label)
    if len(ys) > 1:
        ax.fill_between(x, m - Y.std(0), m + Y.std(0), color=c, alpha=.15)
    return m


# ─────────────────────────────── Kuhn poker ───────────────────────────────
kuhn = {k: v for k, v in G.items() if k[0] == 'kuhn'}
if kuhn:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    md += ['## Kuhn poker (3 cards): exploitability, lower is better (Nash = 0, uniform random = 0.458)\n',
           '| Policy | Opponents | Seeds | Hands played | Start | At 9,600 hands |  Final | Best during training | Time-averaged strategy (at 9,600 / final) | 4-card deck, never trained on (final / time-avg) |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for (env, brain, pool), rs in sorted(kuhn.items()):
        bs = rs[0]['args']['batch']
        xs = np.array([h['update'] * bs for h in rs[0]['hist']])
        cur = [[h['exploitability'] for h in r['hist']] for r in rs]
        avg = [[k3.exploitability(s) for s in avg_strategy(r['hist'])] for r in rs]
        lab = f"{NAME[brain]}, {'pool' if pool else 'naive self-play'}"
        ls = '-' if pool else '--'
        if brain != 'laya0':
            band(ax[0], xs, cur, COL[brain], lab, ls)
            band(ax[1], xs, avg, COL[brain], lab, ls)
        if brain.startswith('laya'):
            e4 = [[h['exploitability_4card'] for h in r['hist']] for r in rs]
            a4 = [[k4.exploitability(s) for s in avg_strategy(r['hist'], 'strategy_4card')] for r in rs]
            if brain == 'laya':
                band(ax[2], xs, a4, COL[brain], lab + ' (time-avg)', ls)
            t4 = f"{np.mean([e[-1] for e in e4]):.3f} / {np.mean([e[-1] for e in a4]):.3f}"
        else:
            m4 = [k4.exploitability(rank_map_4(avg_strategy(r['hist'])[-1])) for r in rs]
            t4 = f"rank-mapped by hand: {np.mean(m4):.3f}"
            if pool:
                ax[2].axhline(np.mean(m4), color=COL[brain], ls=':', label=f'{NAME[brain]} + hand-made rank mapping')
        md.append(f"| {NAME[brain]} | {'pool of past selves' if pool else 'naive self-play'} | {len(rs)} | {xs[-1]} | "
                  f"{np.mean([c[0] for c in cur]):.3f} | {at(xs, cur, 9600)} | {np.mean([c[-1] for c in cur]):.3f} ± {np.std([c[-1] for c in cur]):.3f} | "
                  f"{np.mean([min(c) for c in cur]):.3f} | {at(xs, avg, 9600)} / {np.mean([a[-1] for a in avg]):.3f} | {t4} |")
    if ('kuhn', 'laya0', 1) in kuhn:
        z = kuhn[('kuhn', 'laya0', 1)][0]['hist'][0]
        for a_ in ax[:2]:
            a_.axhline(z['exploitability'], color=COL['laya0'], ls=':', label='Laya zero-shot')
        ax[2].axhline(z['exploitability_4card'], color=COL['laya0'], ls=':', label='Laya zero-shot')
    for a_, t in zip(ax, ['current strategy', 'time-averaged strategy', '4-card deck (never trained on)']):
        a_.axhline(0.458 if '4' not in t else 0.4375, color='grey', lw=.8, ls='-.', label='uniform random')
        a_.set_title(t); a_.set_xlabel('hands played'); a_.set_ylabel('exploitability'); a_.grid(alpha=.3); a_.set_ylim(0, 1)
    ax[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'kuhn.png'), dpi=150); plt.close(fig)
    md.append('')

# ─────────────────────────────── grid hide-and-seek ───────────────────────────────
grid = {k[:3] + ((k[3],) if k[3].startswith('_v2_') else ()): v for k, v in G.items()
        if k[0] == 'grid' and (k[3].startswith('_v2') or k[1] == 'laya0')}
grid_v1 = {k[:3]: v for k, v in G.items() if k[0] == 'grid' and k[3] == '' and k[1] != 'laya0'}
laya_eps = [r['hist'][-1]['update'] * r['args']['batch'] for k, v in grid.items() if k[1] == 'laya' and len(k) == 3 for r in v]
BUDGET = min(laya_eps) if laya_eps else None
if grid:
    keys = [('hider_train', 'hider vs chasing seeker, 7×7 training maps'), ('seeker_train', 'seeker vs fleeing hider, 7×7'),
            ('hider_test', 'hider, 9×9 denser maps (unseen)'), ('seeker_test', 'seeker, 9×9 denser maps (unseen)')]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    md += ['## Text hide-and-seek: score per episode against scripted opponents (−1 to +1, higher is better)\n',
           'Scripted reference: a fleeing hider scores +0.17 against a chasing seeker; standing still scores −0.61.\n',
           '| Policy | Opponents | Seeds | Episodes | ' + ' | '.join(t.split(',')[0] for _, t in keys) + ' | distinct situations encoded |',
           '|---|---|---|---|' + '---|' * len(keys) + '---|']
    for key, rs in sorted(grid.items()):
        env, brain, pool = key[:3]
        extra = f" (lr {rs[0]['args']['lr']:g})" if brain == 'laya' else ''
        bs = rs[0]['args']['batch']
        xs = np.array([h['update'] * bs for h in rs[0]['hist']])
        cells = []
        for i, (k, t) in enumerate(keys):
            ys = [[h[k] for h in r['hist']] for r in rs]
            if brain == 'laya0':
                ax[i].axhline(np.mean([y[0] for y in ys]), color=COL[brain], ls=':', label='Laya zero-shot')
            else:
                band(ax[i], xs, ys, COL[brain] if len(key) == 3 else '#7a2a20', f"{NAME[brain]}{extra}, {'pool' if pool else 'naive'}", '-' if pool else '--')
            f = [y[-1] for y in ys]
            cells.append(f"{np.mean([y[0] for y in ys]):+.2f} → {np.mean(f):+.2f}" + (f" ± {np.std(f):.2f}" if len(f) > 1 else '')
                         + (f" (at {BUDGET} eps: {at(xs, ys, BUDGET)})" if BUDGET and brain == 'mlp' else ''))
        enc = rs[0]['hist'][-1].get('encoded_states', '—')
        md.append(f"| {NAME[brain]}{extra} | {'pool' if pool else 'naive self-play'} | {len(rs)} | {xs[-1]} | " + ' | '.join(cells) + f' | {enc} |')
    for (env, brain, pool), rs in sorted(grid_v1.items()):
        md.append(f"| {NAME[brain]} (v1: {rs[0]['args']['updates'] * rs[0]['args']['batch']} self-play episodes only) | {'pool' if pool else 'naive self-play'} | {len(rs)} | | "
                  + ' | '.join(f"{np.mean([r['hist'][-1][k] for r in rs]):+.2f}" for k, _ in keys) + ' | |')
    for i, (k, t) in enumerate(keys):
        ax[i].axhline(0.173 if 'hider' in k else -0.173, color='grey', lw=.8, ls='-.', label='scripted flee/chase')
        ax[i].set_title(t, fontsize=9); ax[i].set_xlabel('training episodes'); ax[i].grid(alpha=.3)
    ax[0].set_ylabel('score per episode'); ax[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'grid.png'), dpi=150)
    for a_ in ax:                     # the budget Laya could afford
        a_.set_xlim(0, 4000); a_.relim(); a_.autoscale(axis='y')
    fig.savefig(os.path.join(FIG, 'grid_zoom.png'), dpi=150); plt.close(fig)
    md.append('')

open(os.path.join(RES, 'tables.md'), 'w').write('\n'.join(md))
print('\n'.join(md))
