"""Learning curve: accuracy gain and ECE vs feedback rows per task (1,720-row test set, 3 seeds)."""
import glob, json, os, re, numpy as np, matplotlib, matplotlib.ticker
import paths as P
matplotlib.use('Agg'); import matplotlib.pyplot as plt
def load(tag):
    out = {}
    for n in (5, 15, 30, 60):
        fs = glob.glob(os.path.join(P.RESULTS, f'db_*_rows_s*_big_n{n}_{tag}.json'))
        acc = [json.load(open(f))['final']['test_recal']['acc'] for f in fs] if tag != 'sup' else [json.load(open(f))['final']['test']['acc'] for f in fs]
        ece = [json.load(open(f))['final']['test_recal']['ece'] for f in fs] if tag != 'sup' else [json.load(open(f))['final']['test']['ece'] for f in fs]
        out[n] = (np.mean(acc), np.std(acc), np.mean(ece))
    return out
Z = glob.glob(os.path.join(P.RESULTS, 'db_zero_rows_s*_big.json'))
z = np.mean([json.load(open(f))['final']['test']['acc'] for f in Z])
ze = np.mean([json.load(open(f))['final']['test']['ece'] for f in Z])
S = {'Ours: feedback only (KL 0.1 + feedback calibration)': load('kl0.1'), 'Feedback only, KL 1': load('kl1'), 'Supervised, gold labels (no calibration step)': load('sup')}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for (name, d), c in zip(S.items(), ['#1f77b4', '#9ecae1', '#d62728']):
    xs = sorted(d)
    ax[0].errorbar(xs, [100 * (d[x][0] - z) for x in xs], [100 * d[x][1] for x in xs], marker='o', color=c, label=name, capsize=3)
    ax[1].plot(xs, [d[x][2] for x in xs], marker='o', color=c)
ax[0].axhline(0, color='gray', lw=0.8); ax[1].axhline(ze, color='gray', lw=0.8, ls='--', label='released ' + P.LABEL)
ax[0].set(xscale='log', xticks=[5, 15, 30, 60], xticklabels=['5', '15', '30', '60'], xlabel='feedback rows per task (43 tasks)', ylabel=f'accuracy gain over released {P.LABEL} (points)')
ax[1].set(xscale='log', xticks=[5, 15, 30, 60], xticklabels=['5', '15', '30', '60'], xlabel='feedback rows per task', ylabel='expected calibration error (lower is better)')
for a_ in ax: a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].legend(fontsize=7.5, loc='upper left'); ax[1].legend(fontsize=7.5)
plt.tight_layout(); os.makedirs(P.FIGURES, exist_ok=True); out = os.path.join(P.FIGURES, f'vega_feedback_curve{P.SFX}.png'); plt.savefig(out, dpi=150); print('saved', out)
