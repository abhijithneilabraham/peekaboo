"""Build the head-to-head tables from results/compare/*.json (all on Vega's exact 1,075-row DecisionBench slice).

  1. main table: mean ± std over seeds, every metric the README reports for DecisionBench
  2. paired tests: per-row accuracy difference vs the released model and vs vegaml's TTT, bootstrap 95% CI
  3. where TTT cannot help: rows whose option set its per-task heads do not cover
  4. per-domain accuracy
"""
import glob, json, os, re
import numpy as np

import paths as P

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for f in sorted(f for f in glob.glob(os.path.join(P.COMPARE, '*.json')) if not f.endswith('latency.json')):
    d = json.load(open(f)); key = re.sub(r'_s\d+', '', os.path.basename(f)[:-5])
    R.setdefault(key, []).append(d)
ORDER = [('zero', 'Released ' + P.LABEL, 'none'), ('ttt_auto', "vegaml fit() TTT head, 'auto'", 'gold labels'),
         ('sup_ts', 'Adapter, supervised + temp. scaling', 'gold labels'),
         ('linear_fb', 'No-engine control (bilinear), feedback', 'right/wrong only'),
         ('feedback_kl1', 'Ours, KL 1', 'right/wrong only'), ('feedback', 'Ours: feedback RL + feedback calibration', 'right/wrong only')]
rng = np.random.default_rng(0)


def ms(key, f):
    x = [d['test'][f] for d in R[key]]
    return f'{np.mean(x):.3f}' + (f' ± {np.std(x):.3f}' if len(x) > 1 else '')


def correct(key):
    return np.mean([d['test']['rows']['correct'] for d in R[key]], 0)


def ci(diff):
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
    return f'{diff.mean() * 100:+.1f} [{np.percentile(bs, 2.5) * 100:+.1f}, {np.percentile(bs, 97.5) * 100:+.1f}]'


print('| method | learns from | seeds | accuracy | skill | NLL | ECE | KL to gold |\n|---|---|---|---|---|---|---|---|')
for k, name, src in ORDER:
    if k in R:
        print(f"| {name} | {src} | {len(R[k])} | {ms(k, 'acc')} | {ms(k, 'skill')} | {ms(k, 'nll')} | {ms(k, 'ece')} | {ms(k, 'kl_soft')} |")

print('\nPaired accuracy gain, points [95% bootstrap CI], same 1,075 rows:')
for k, name, _ in ORDER[1:]:
    if k in R:
        line = f'  {name:42s} vs released: {ci(correct(k) - correct("zero"))}'
        if 'ttt_auto' in R and k != 'ttt_auto':
            line += f'   vs TTT: {ci(correct(k) - correct("ttt_auto"))}'
        print(line)

if 'ttt_auto' in R:
    items = json.load(open(os.path.join(HERE, 'decisionbench_items.json')))
    bh = np.array(R['ttt_auto'][0]['info']['by_head'], bool)
    print(f"\nTTT heads kept {R['ttt_auto'][0]['info']['heads_kept']}/43 tasks and answer {bh.sum()}/1075 rows; the rest fall back to the engine.")
    print('| rows | n | ' + ' | '.join(n for k, n, _ in ORDER if k in R) + ' |')
    print('|---|---|' + '---|' * sum(k in R for k, _, _ in ORDER))
    for nm, m in [('TTT head answers', bh), ('TTT falls back to engine', ~bh)]:
        print(f'| {nm} | {m.sum()} | ' + ' | '.join(f'{correct(k)[m].mean():.3f}' for k, _, _ in ORDER if k in R) + ' |')

doms = sorted(R['zero'][0]['test']['by_domain']) if 'zero' in R else []
if doms:
    print('\n| domain | ' + ' | '.join(n for k, n, _ in ORDER if k in R) + ' |')
    print('|---|' + '---|' * sum(k in R for k, _, _ in ORDER))
    for dm in doms:
        print(f'| {dm} | ' + ' | '.join(f"{np.mean([d['test']['by_domain'][dm] for d in R[k]]):.3f}" for k, _, _ in ORDER if k in R) + ' |')
