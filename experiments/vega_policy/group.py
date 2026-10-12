"""Mean over seeds of final / last-5-eval exploitability, grouped by run name minus the seed."""
import json, glob, re, sys, numpy as np
pat = sys.argv[1] if len(sys.argv) > 1 else 'results/kuhn_*.json'
G = {}
for f in glob.glob(pat):
    name = f.split('/')[-1][:-5]; key = re.sub(r'_s\d+', '', name)
    h = json.load(open(f))['hist']
    G.setdefault(key, []).append((np.mean([x['exploitability'] for x in h[-5:]]), np.mean([x.get('exploitability_4card', np.nan) for x in h[-5:]]), min(x['exploitability'] for x in h)))
for k, v in sorted(G.items(), key=lambda kv: np.mean([x[0] for x in kv[1]])):
    v = np.array(v); print(f'{k:42s} n={len(v)}  last5 {v[:,0].mean():.3f}±{v[:,0].std():.3f}  4card {v[:,1].mean():.3f}  best-ever {v[:,2].mean():.3f}')
