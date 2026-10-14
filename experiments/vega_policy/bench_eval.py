"""Replay Vega's engine on cached DecisionBench features and score variants of the readout.

No backbone is loaded: Vega is built around a stub perception, so only the 14M engine and its shipped adapters
and calibration sit in memory. The baseline replay must reproduce the official probabilities stored by
bench_extract.py before any variant is trusted.

    python bench_eval.py
"""
import json, math, os, sys
import numpy as np
import torch

import paths

HERE = os.path.dirname(os.path.abspath(__file__))


def load(items_file='decisionbench_items.json'):
    from vegaml import hub
    from vegaml._api import Vega
    mdir = hub.fetch(paths.SIZE)
    cfg = json.load(open(os.path.join(mdir, 'vega_config.json')))

    class Stub:
        feature_layers = cfg['feature_layers']
    v = Vega(mdir, device='cpu', perception=Stub())
    items = json.load(open(os.path.join(HERE, items_file)))
    bank = torch.load(paths.bank('decisionbench'))
    items = [it for it in items if it['id'] in bank and 'feats' in bank[it['id']]]
    return v, items, bank


def run(v, items, bank, feat_fn=None, steps=None, halt=None, adapters=None):
    """Probabilities per row (original option order), mirroring Vega._view on cached features."""
    from vegaml._common import QTYPES, thermal_features, thermal_tau, softmax_np
    eng = v.engine
    old = (eng.adapter_mode, eng.max_steps)
    if adapters:
        eng.adapter_mode = adapters
    if steps:
        eng.max_steps = steps
    out_p = []
    with torch.no_grad():
        for it in items:
            r = bank[it['id']]
            feats = r['feats'][None]
            if feat_fn:
                feats = feat_fn(it, feats)
            k = r['opts'].shape[0]
            opts, mask = r['opts'][None], torch.ones(1, k, dtype=torch.bool)
            qt = torch.tensor([QTYPES[it['question']['type']]])
            chunks = None if r['chunks'] is None else r['chunks'][None]
            cmask = None if r['cmask'] is None else r['cmask'][None]
            out = eng(feats, opts, mask, qt, halt=v.halt if halt is None else halt, return_traj=False, chunks=chunks, chunk_mask=cmask)
            lg = out['logits'].float().numpy()
            phys = thermal_features(out, mask, eng.max_steps).numpy()
            gates = out['gates'].float().numpy() if out.get('gates') is not None else None
            route = v._route(gates)
            rt = -1 if route is None else int(route[0])
            P = softmax_np(lg, thermal_tau(v._cal(rt)['theta'], qt.numpy(), phys))[0, :k]
            out_p.append(P)
    eng.adapter_mode, eng.max_steps = old
    return out_p


def ece(conf, corr, bins=10):
    conf, corr = np.asarray(conf), np.asarray(corr)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return e


def score(items, P):
    gold = [it['labels'].index(it['gold']) for it in items]
    acc = np.mean([int(np.argmax(p) == g) for p, g in zip(P, gold)])
    nll = np.mean([-math.log(max(p[g], 1e-12)) for p, g in zip(P, gold)])
    e = ece([p.max() for p in P], [int(np.argmax(p) == g) for p, g in zip(P, gold)])
    soft = [(np.asarray(it['soft_gold']), p) for it, p in zip(items, P) if it['soft_gold'] and max(it['soft_gold']) < 0.999]
    kl = np.mean([(q * (np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(p, 1e-12)))).sum() for q, p in soft]) if soft else float('nan')
    return {'acc': round(float(acc), 4), 'nll': round(float(nll), 4), 'ece': round(float(e), 4), 'kl_soft': round(float(kl), 4), 'n_soft': len(soft), 'n': len(items)}


def task_means(items, bank):
    by = {}
    for it in items:
        by.setdefault(it['task'], []).append(bank[it['id']]['feats'])
    return {t: torch.stack(v).mean(0) for t, v in by.items()}


def main():
    v, items, bank = load()
    print(len(items), 'rows cached')
    P0 = run(v, items, bank)
    chk = [(it, p) for it, p in zip(items, P0) if bank[it['id']].get('official')]
    if chk:
        diffs = [max(abs(bank[it['id']]['official'][str(lab)] - p[j]) for j, lab in
                     enumerate(['False', 'True'] if it['question']['type'] == 'boolean' else it['labels'])) for it, p in chk]
        print(f'replay vs official decide(): max |dp| = {max(diffs):.2e} over {len(chk)} rows')
    res = {'baseline': score(items, P0)}
    print('baseline', res['baseline'], flush=True)
    mu = task_means(items, bank)
    for c in [float(x) for x in (sys.argv[1:] or ['1.5', '2', '3'])]:
        P = run(v, items, bank, feat_fn=lambda it, f, c=c: mu[it['task']] + c * (f - mu[it['task']]))
        res[f'task_contrast_{c:g}'] = score(items, P); print(f'task_contrast {c:g}', res[f'task_contrast_{c:g}'], flush=True)
    for name, kw in [('adapters_off', dict(adapters='off')), ('no_halt', dict(halt=False)), ('steps24_nohalt', dict(steps=24, halt=False))]:
        res[name] = score(items, run(v, items, bank, **kw)); print(name, res[name], flush=True)
    os.makedirs(paths.RESULTS, exist_ok=True)
    json.dump(res, open(os.path.join(paths.RESULTS, 'decisionbench_variants.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
