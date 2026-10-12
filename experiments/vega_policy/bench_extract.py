"""Cache Vega's frozen backbone features for the DecisionBench slice Vega reports on, plus the official answers.

The slice is rebuilt exactly as notebooks/vega_benchmark_2xT4.ipynb builds it (25 rows per task, seed 20261011),
and each row is read the way Vega.decide reads it (views=1, shipped adapters on 'auto', halt on). Saved per row:
pooled features, option features, memory chunks, and the official probabilities, so every later experiment
replays the engine on the cache and can check itself against the official numbers.

    python bench_extract.py [limit]
"""
import json, os, random, re, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'bank_decisionbench.pt')

_POS = {"yes", "true", "relevant", "safe", "permitted", "permit", "allowed", "allow", "approve",
        "approved", "accept", "accepted", "retain", "valid", "compliant", "sufficient"}
_NEG = {"no", "false", "irrelevant", "unsafe", "prohibited", "prohibit", "denied", "deny",
        "reject", "rejected", "discard", "invalid", "noncompliant", "insufficient", "block", "blocked"}


def _first_word(label):
    m = re.search(r"[A-Za-z]+", str(label))
    return m.group(0).lower() if m else ""


def _polarity(labels):
    if len(labels) != 2:
        return None
    low = [str(x).strip().lower() for x in labels]
    if low[0] == low[1]:
        return None
    for i in (0, 1):
        other = low[1 - i]
        if other in ("not " + low[i], "non-" + low[i], "no " + low[i], low[i] + " not"):
            return i
    heads = [_first_word(x) for x in labels]
    for i in (0, 1):
        if heads[i] in _POS and heads[1 - i] in _NEG:
            return i
    return None


def fetch_decisionbench(per_task=25, seed=20261011, max_state_chars=12000):
    """Verbatim logic of the notebook's fetch_decisionbench (binary_as_boolean=True)."""
    from datasets import load_dataset
    ds = load_dataset("Hanno-Labs/decision-bench", split="eval")
    by_task = {}
    for i, task in enumerate(ds["task_id"]):
        by_task.setdefault(task, []).append(i)
    rng = random.Random(seed)
    picked = []
    for task in sorted(by_task):
        idx = by_task[task]
        rng.shuffle(idx)
        picked.extend(idx[:per_task])
    items = []
    for i in sorted(picked):
        row = ds[i]
        try:
            cands = json.loads(row["candidates_json"]); state = json.loads(row["state_json"])
        except Exception:
            continue
        if not isinstance(cands, list) or len(cands) < 2:
            continue
        ids = [str(c.get("id")) for c in cands]
        labs = [str(c.get("label") or c.get("id")) for c in cands]
        descs = [c.get("description") for c in cands]
        if str(row["gold_candidate_id"]) not in ids:
            continue
        gold_i = ids.index(str(row["gold_candidate_id"]))
        prim = row["primitive"]
        soft = list(row.get("gold_probabilities") or [])
        text = json.dumps(state, ensure_ascii=False)[:max_state_chars]
        if prim.startswith("binary") and _polarity(labs) is not None:
            pos = _polarity(labs)
            question = {"type": "boolean", "instructions": row["instruction"],
                        "criteria": {"true": labs[pos], "false": labs[1 - pos]}}
            labels, gold = ["false", "true"], ("true" if gold_i == pos else "false")
            soft = [soft[1 - pos], soft[pos]] if len(soft) == 2 else []
        else:
            crit = {}
            for lab, desc, cid in zip(labs, descs, ids):
                crit[lab if lab not in crit else "%s (%s)" % (lab, cid)] = desc
            question = {"type": "choice", "instructions": row["instruction"], "criteria": crit}
            labels = list(crit); gold = labels[gold_i]
        items.append({"id": row["row_id"], "state": text, "gold": gold, "question": question, "labels": labels,
                      "task": row["task_id"], "primitive": prim, "domain": row.get("domain"),
                      "soft_gold": [float(x) for x in soft] if len(soft) == len(labels) else None})
    return items


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    per_task = int(os.environ.get('DB_PER_TASK', '25'))       # 25 = the slice Vega reports; larger supersets it
    items_path = os.path.join(HERE, 'decisionbench_items.json' if per_task == 25 else f'decisionbench_items_{per_task}.json')
    if not os.path.exists(items_path):
        json.dump(fetch_decisionbench(per_task=per_task), open(items_path, 'w'))
    items = json.load(open(items_path))[:limit]
    bank = torch.load(OUT) if os.path.exists(OUT) else {}
    todo = [it for it in items if it['id'] not in bank]
    print(len(items), 'rows,', len(todo), 'to encode', flush=True)
    if not todo:
        return
    from vegaml import hub
    from vegaml._api import Vega
    from vegaml._common import build_prompt, serialize_state
    from vpol.vega import QWEN
    v = Vega(hub.fetch(), backbone=QWEN, device='cpu')
    cfg, C = v.cfg, (v.engine.memory_chunks if v.engine.memory is not None else 0)
    t0 = time.time()
    for n, it in enumerate(todo, 1):
        q = v._internal(it['question'])
        try:
            ids, spans = build_prompt(v.perc.tok, serialize_state(it['state']), q, cfg['max_len'], template=0,
                                      option_order=list(range(len(it['labels']))), opt_max_tokens=cfg['opt_max_tokens'],
                                      query_first_max_state=cfg.get('query_first_max_state', 0))
        except Exception as e:
            bank[it['id']] = {'error': str(e)[:200]}; continue
        x = [{'ids': np.asarray(ids, dtype=np.int32), 'spans': spans}]
        with torch.no_grad():
            if C:
                feats, opt_flat, chunks, cmask = v.perc.pooled(x, chunks=C, prefix_min=cfg.get('prefix_min_tokens', 4096),
                                                               chunk_min_tokens=cfg.get('memory_min_tokens', 2048),
                                                               chunk_tokens=cfg.get('memory_chunk_tokens', 128))
            else:
                (feats, opt_flat), chunks, cmask = v.perc.pooled(x, prefix_min=cfg.get('prefix_min_tokens', 4096)), None, None
        ans = v.decide(it['state'], {'q': it['question']})['answers']['q'] if len(bank) < 20 else None   # official reading, to check replays
        bank[it['id']] = {'feats': feats[0].clone(), 'opts': opt_flat.clone(), 'n_tokens': len(ids),
                          'chunks': None if chunks is None else chunks[0].clone(), 'cmask': None if cmask is None else cmask[0].clone(),
                          'official': ans and {str(k): float(p) for k, p in ans['probs'].items()}}
        if n % 100 == 0 or n == len(todo):
            torch.save(bank, OUT + '.tmp'); os.replace(OUT + '.tmp', OUT)
            print(f'{n}/{len(todo)}  {time.time() - t0:.0f}s  ({(time.time() - t0) / n:.1f}s/row, {len(ids)} tok)', flush=True)


if __name__ == '__main__':
    main()
