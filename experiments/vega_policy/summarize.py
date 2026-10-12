import json, glob, sys, numpy as np
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'results/kuhn_*.json')):
    h = json.load(open(f))['hist']; e = [x['exploitability'] for x in h]; e4 = [x['exploitability_4card'] for x in h]
    print(f.split('/')[-1][:-5].ljust(38), 'final %.3f  last5 %.3f  min %.3f | 4card final %.3f last5 %.3f' % (e[-1], np.mean(e[-5:]), min(e), e4[-1], np.mean(e4[-5:])))
