"""DecisionBench feedback runs, mean over seeds. 'recal' = after the feedback-only temperature.
Paired test: per test row, method vs the released model on the same rows; bootstrap 95% CI of the accuracy gain."""
import json, glob, re, sys, numpy as np
pat = sys.argv[1] if len(sys.argv) > 1 else 'results/db_*.json'
G = {}
for f in glob.glob(pat):
    d = json.load(open(f)); key = re.sub(r'_s\d+', '', f.split('/')[-1][:-5])
    fin = d.get('final', {}); z = d['hist'][0]['test']
    t = fin.get('test_recal') or fin.get('test') or d['hist'][-1]['test']
    raw = fin.get('test') or d['hist'][-1]['test']
    gain = None
    if 'rows' in t and 'rows' in z:
        gain = np.array(t['rows']['correct']) - np.array(z['rows']['correct'])
    G.setdefault(key, []).append(dict(acc=t['acc'], nll=t['nll'], ece=t['ece'], kl=t['kl_soft'], raw_nll=raw['nll'], zacc=z['acc'], znll=z['nll'],
                                      zece=z['ece'], T=fin.get('T'), gain=gain, n=d['n_test'], ntr=d['n_train']))
rng = np.random.default_rng(0)
for k, v in sorted(G.items()):
    m = lambda f: np.mean([x[f] for x in v])
    ci = ''
    gs = [x['gain'] for x in v if x['gain'] is not None]
    if gs:
        g = np.mean(gs, 0); bs = [g[rng.integers(0, len(g), len(g))].mean() for _ in range(2000)]
        ci = f' gain {g.mean()*100:+.1f}pt [{np.percentile(bs,2.5)*100:+.1f},{np.percentile(bs,97.5)*100:+.1f}]'
    T = f" T={m('T'):.2f}" if v[0]['T'] else ''
    print(f"{k:34s} n={len(v)} train={v[0]['ntr']:4d} test={v[0]['n']}  acc {m('acc'):.4f} (zero {m('zacc'):.4f}){ci}  nll {m('nll'):.3f} (raw {m('raw_nll'):.2f}, zero {m('znll'):.3f})  ece {m('ece'):.3f} (zero {m('zece'):.3f}){T}")
