"""Why does Laya's grid policy barely move? Track action distributions, entropy and grad norms
over a few PPO updates on fixed probe states."""
import json, sys
import numpy as np
import torch

import train as T
from lpol import grid_hs as G

lr = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
rng = np.random.default_rng(0); torch.manual_seed(0)
brain = T.LayaBrain({'hider': G.QUESTION['hider'], 'seeker': G.QUESTION['seeker']}, lr=lr)
probe = {'hider': [], 'seeker': []}
for i in range(40):
    e = G.GridHS(rng=np.random.default_rng(1000 + i)).reset()
    for _ in range(int(rng.integers(0, 10))):
        e.step({'hider': G.flee(e), 'seeker': G.chase(e)})
    for who in probe:
        probe[who].append(e.describe(who))


def report(tag):
    out = {'tag': tag}
    with torch.no_grad():
        for who, S in probe.items():
            p = torch.softmax(brain.logits(S, who), -1).numpy()
            out[who + '_mean_p'] = dict(zip(G.ACTS, np.round(p.mean(0), 3).tolist()))
            out[who + '_entropy'] = round(float(-(p * np.log(p + 1e-12)).sum(1).mean()), 3)
            out[who + '_argmax_share'] = round(float(np.bincount(p.argmax(1), minlength=5).max() / len(p)), 2)
    print(json.dumps(out), flush=True)
    return out


report('start')
V = {}
w0 = [p.detach().clone() for p in brain.params()]
for u in range(6):
    trans, _, _ = T.grid_batch(brain, 16, rng, (7, 0.18), opponent='script' if u % 2 else 'self')
    adv_src = np.array([t[4] for t in trans])
    T.ppo_update(brain, trans, V)
    g = sum(float(p.grad.norm() ** 2) for p in brain.params() if p.grad is not None) ** .5
    dw = sum(float((p.detach() - q).norm() ** 2) for p, q in zip(brain.params(), w0)) ** .5
    print(json.dumps({'update': u, 'n_trans': len(trans), 'return_sd': round(float(adv_src.std()), 3),
                      'grad_norm_last_epoch': round(g, 4), 'weight_change': round(dw, 3)}), flush=True)
report('after 6 updates')
