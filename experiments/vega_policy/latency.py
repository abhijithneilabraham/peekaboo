"""Engine-only decision latency on CPU, batch 1, on cached features: released engine vs engine + one feedback adapter."""
import json, time, numpy as np, torch
import bench_eval as B
from bench_rl import Policy
torch.set_num_threads(4)
v, items, bank = B.load('decisionbench_items.json')
rows = items[:300]
def bench(pol=None):
    ts = []
    with torch.no_grad():
        for it in rows:
            t = time.perf_counter()
            if pol is None: B.run(v, [it], bank)
            else: pol.logp(bank, [it])
            ts.append(1000 * (time.perf_counter() - t))
    return float(np.percentile(ts[20:], 50)), float(np.percentile(ts[20:], 95))
base = bench()
pol = Policy(v, 1e-3, rank=8)
with torch.no_grad():   # give the adapter non-zero weights so its compute is real
    for n, p in pol.eng.named_parameters():
        if 'lora_B' in n and n.endswith(f'.{pol.j}'): p.normal_(0, 1e-3)
ad = bench(pol)
n_ad = sum(p.numel() for p in pol.params)
out = {'released_ms_p50_p95': base, 'with_adapter_ms_p50_p95': ad, 'adapter_params': n_ad, 'engine_params': sum(p.numel() for n, p in pol.eng.named_parameters() if 'lora_' not in n and 'gate_nets' not in n)}
print(json.dumps(out)); json.dump(out, open('results/compare/latency.json', 'w'))
