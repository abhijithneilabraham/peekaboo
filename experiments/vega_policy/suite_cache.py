"""Stage 1 (the only stage that runs the language model): read every row of a suite once through
Vega.decide(mode="engine"), exactly as VegaML's benchmark notebook reads it, and keep two things:

  * the official record the notebook scores (probabilities, latency): the "released Vega" column
  * the backbone features decide() used for that row, captured from inside the call, so every method
    afterwards (ours and the baselines) runs the engine on exactly the features Vega itself saw

Suites and rows, all loaded with the notebook's own code (harness/):
  public         the 43-task menu: 60 evaluation rows per task, plus the 50 "fit" rows per task from the
                 training split that the notebook gives Vega's TTT head (our feedback pool)
  jevbench       the public tiers (231 items); no training rows exist, so methods cross-fit (adapt.py)
  decisionbench  the README slice (25 per task, 1,075 rows) plus a pool of other rows of the same tasks
                 (100 per task minus the slice)
  typed          LocalLLaMA/typed-decisions, Laya's headline benchmark: 2,000 test decisions, the 6,000 training
                 decisions as the pool

    python suite_cache.py --suite public            # VEGA_SIZE / VEGA_OUT / VEGA_DEVICE from the environment
"""
import argparse, json, os, sys, time
import numpy as np
import torch

import paths
paths.harness()

SEED = 20261011


# ───────────────────────────── rows ─────────────────────────────
def build_items(suite, quick=False):
    """The suite's rows as a list of dicts with: id, state, gold, question, labels, qtype, task, split."""
    import bench_suites as BS
    import bench_tasks as BT
    from bench_load import load_task
    meta = {}
    if suite == 'public':
        out, keys = [], [t['key'] for t in BT.TASKS]
        if quick:
            keys = ['boolq', 'emotion', 'arc_challenge', 'banking77']
        for key in keys:
            t = BT.BY_KEY[key]
            try:
                r = load_task(t, n_eval=8 if quick else 60, n_fit=12 if quick else 50, seed=SEED)
            except Exception as exc:
                r = {'skipped': '%s: %s' % (type(exc).__name__, str(exc)[:200])}
            if r.get('skipped'):
                meta[key] = {'skipped': r['skipped']}; print('  public', key, 'SKIPPED', r['skipped'], flush=True); continue
            qtype = r['meta']['qtype']
            for split, rows in (('test', r['items']), ('pool', r['fit'])):
                for it in rows:
                    q = it.get('question') or r['question']
                    out.append({'id': f'{key}:{it["id"]}', 'state': it['state'], 'gold': it['gold'], 'question': q,
                                'labels': list(r['labels']), 'qtype': qtype, 'task': key, 'split': split})
            meta[key] = {'n_test': len(r['items']), 'n_pool': len(r['fit']), 'labels': len(r['labels']), 'qtype': qtype}
            print(f'  public {key:24s} test={len(r["items"]):3d} pool={len(r["fit"]):3d} labels={len(r["labels"])}', flush=True)
        return out, meta
    if suite == 'jevbench':
        items, meta = BS.fetch_jevbench()
        if meta.get('errors'):
            raise RuntimeError('JevBench tiers unreachable: %s' % meta['errors'])
        if quick:
            items = items[:24]
        for it in items:
            it.update(split='test', task=it.get('family') or 'jevbench')
        return items, meta
    if suite == 'decisionbench':
        test, meta = BS.fetch_decisionbench(per_task=2 if quick else 25, seed=SEED)
        pool, _ = BS.fetch_decisionbench(per_task=6 if quick else 100, seed=SEED)
        ids = {it['id'] for it in test}
        full = [dict(it, split='test' if it['id'] in ids else 'pool') for it in pool]    # 100 per task, a superset
        missing = ids - {it['id'] for it in full}
        if missing:
            raise RuntimeError(f'{len(missing)} slice rows are not in the 100-per-task sample')
        meta = {k: v for k, v in meta.items() if k != 'binary_note'}
        return full, meta
    if suite in ('emotion', 'agnews'):               # large public datasets with real training splits: dair-ai/emotion is the
        key = {'emotion': 'emotion', 'agnews': 'ag_news'}[suite]   # dataset of Vega's fine-tuning recipe, AG News one Laya reports
        natural = os.environ.get('POOL_DRAW', 'natural') == 'natural'
        r = load_task(BT.BY_KEY[key], n_eval=40 if quick else 2000, n_fit=80 if quick else 4000, seed=SEED)   # VegaML's own loader and question
        if r.get('skipped'):
            raise RuntimeError(f'{suite}: {r["skipped"]}')
        fit = r['fit']                                # v1: VegaML's class-by-class (round robin) draw, a class-balanced pool
        if natural and not quick:
            # v2: the feedback pool keeps the training split's natural class mix (a deployment's traffic). Asking the
            # loader for the whole split returns every usable training row; a random 4,000 of them (never a row whose
            # id a test row carries) is the pool. The test rows are the same as v1's.
            r2 = load_task(BT.BY_KEY[key], n_eval=0, n_fit=10 ** 6, seed=SEED)
            test_ids = {it['id'] for it in r['items']}
            allrows = [it for it in r2['fit'] if it['id'] not in test_ids]
            rng = np.random.default_rng(SEED)
            fit = [allrows[i] for i in sorted(rng.choice(len(allrows), size=min(4000, len(allrows)), replace=False))]
        out = []
        for split, rows in (('test', r['items']), ('pool', fit)):
            for it in rows:
                out.append({'id': f'{suite}:{it["id"]}', 'state': it['state'], 'gold': it['gold'], 'question': it.get('question') or r['question'],
                            'labels': list(r['labels']), 'qtype': r['meta']['qtype'], 'task': key, 'split': split})
        return out, {'source': r['meta'].get('hf'), 'n_test': len(r['items']), 'n_pool': len(fit), 'labels': r['labels'],
                     'pool_draw': 'natural class mix, random sample' if natural else "VegaML's stratified round robin"}
    if suite == 'typed':                              # Laya's headline benchmark, LocalLLaMA/typed-decisions (all workflows)
        from datasets import load_dataset
        out, meta = [], {'source': 'LocalLLaMA/typed-decisions, config all', 'decisions': {}}
        for split, tag in (('test', 'test'), ('train', 'pool')):
            ds = load_dataset('LocalLLaMA/typed-decisions', 'all', split=split)
            rows = list(ds)[:12 if quick else None]
            for r in rows:
                state = r['state']
                try:
                    state = json.loads(state)
                except Exception:
                    pass
                qs, gold = json.loads(r['questions']), json.loads(r['gold'])
                for qid, qd in qs.items():
                    if qid not in gold:
                        continue
                    g, t = gold[qid], qd['type']
                    if t == 'choice':
                        labels = [str(k) for k in qd['criteria']]; lab = str(g['label'])
                        q = {'type': 'choice', 'instructions': qd['instructions'], 'criteria': qd['criteria']}
                    elif t == 'noul':                 # Laya's yes/no type is Vega's boolean: options [false, true]
                        labels = ['false', 'true']; lab = str(g['label']).lower()
                        q = {'type': 'boolean', 'instructions': qd['instructions'],
                             'criteria': {k: v for k, v in (qd.get('criteria') or {}).items()} or {'false': 'no', 'true': 'yes'}}
                    elif t == 'score':
                        labels = [str(i) for i in range(len(qd['criteria']))]; lab = str(int(g['label']))
                        q = {'type': 'score', 'instructions': qd['instructions'], 'criteria': qd['criteria']}
                    else:
                        continue
                    if lab not in labels:
                        continue
                    pr = g.get('probabilities') or {}
                    soft = [float(pr.get(k, 0.0)) for k in labels]
                    out.append({'id': f'{r["id"]}:{qid}', 'state': state, 'gold': lab, 'question': q, 'labels': labels,
                                'qtype': t, 'task': f'{r["workflow"]}/{qid}', 'workflow': r['workflow'], 'split': tag,
                                'soft_gold': soft if sum(soft) > 0 else None})
            meta['decisions'][split] = sum(it['split'] == tag for it in out)
        return out, meta
    raise SystemExit('unknown suite ' + suite)


def load_items(suite, quick=False):
    path = paths.items(f'items_{suite}.json')
    if not os.path.exists(path):
        items, meta = build_items(suite, quick)
        if not any(it['split'] == 'test' for it in items):
            raise RuntimeError(f'{suite}: no rows loaded')
        json.dump({'items': items, 'meta': meta}, open(path + '.tmp', 'w')); os.replace(path + '.tmp', path)
        if suite == 'decisionbench':              # the row lists bench_rl.py reads (Section 5.5 experiments)
            json.dump([it for it in items if it['split'] == 'test'], open(paths.items('decisionbench_items.json'), 'w'))
            json.dump(items, open(paths.items('decisionbench_items_100.json'), 'w'))
    return json.load(open(path))['items']


# ───────────────────────────── features + official records ─────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', required=True); ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    items = load_items(a.suite, a.quick)
    bank_path, rec_path = paths.bank(a.suite), os.path.join(paths.RESULTS, f'released_{a.suite}.json')
    os.makedirs(paths.RESULTS, exist_ok=True)
    bank = torch.load(bank_path) if os.path.exists(bank_path) else {}
    recs = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
    todo = [it for it in items if it['id'] not in bank or it['id'] not in recs]
    print(f'{paths.LABEL} {a.suite}: {len(items)} rows, {len(todo)} to read', flush=True)
    marker = os.path.join(paths.RESULTS, f'cache_{a.suite}.done')
    if not todo:
        open(marker, 'w').write(f'{len(items)} rows\n'); return

    import bench_readouts as RO
    dev = paths.device()
    vr = RO.VegaRunner(paths.SIZE, dev, token=os.environ.get('HF_TOKEN') or None)
    eng = vr.vega.engine
    print(f'loaded {paths.LABEL} on {dev}: backbone {vr.cfg.get("backbone_name")}, adapters {vr.adapter_names}', flush=True)
    cap, orig = {}, eng.perc.pooled

    def hooked(*args, **kw):                      # what decide() feeds the engine, kept for replay
        r = orig(*args, **kw); cap['r'] = r; return r
    eng.perc.pooled = hooked

    def save():
        torch.save(bank, bank_path + '.tmp'); os.replace(bank_path + '.tmp', bank_path)
        json.dump(recs, open(rec_path + '.tmp', 'w')); os.replace(rec_path + '.tmp', rec_path)

    t0 = time.time()
    for n, it in enumerate(todo, 1):
        cap.clear()
        try:
            rec = vr.engine_record(it, it['question'], it['labels'])
        except Exception as exc:                  # scored as wrong, as the notebook does
            rec = {'id': it['id'], 'gold': it['gold'], 'pred': None, 'probs': None, 'error': '%s: %s' % (type(exc).__name__, str(exc)[:200])}
        rec['labels'] = it['labels']
        recs[it['id']] = rec
        r = cap.get('r')
        if r is not None and 'error' not in rec:
            if len(r) == 4:
                feats, opt_flat, chunks, cmask = r
            else:
                (feats, opt_flat), chunks, cmask = r, None, None
            bank[it['id']] = {'feats': feats[0].float().cpu().clone(), 'opts': opt_flat.float().cpu().clone(),
                              'chunks': None if chunks is None else chunks[0].float().cpu().clone(),
                              'cmask': None if cmask is None else cmask[0].cpu().clone()}
        else:
            bank[it['id']] = {'error': rec.get('error', 'no features captured')}
        if n % 500 == 0 or n == len(todo):
            save()
            el = time.time() - t0
            print(f'  {n}/{len(todo)}  {el:.0f}s  ({el / n:.2f} s/row)', flush=True)
    errs = sum('error' in r for r in recs.values())
    print(f'done: {len(recs)} records, {errs} rows decide() could not answer', flush=True)
    if all(it['id'] in recs and it['id'] in bank for it in items):
        open(marker, 'w').write(f'{len(items)} rows, {errs} unanswerable\n')


if __name__ == '__main__':
    main()
