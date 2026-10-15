# Vendored verbatim from NandhaKishorM/vegaml@00d3d99, notebooks/vega_benchmark_2xT4.ipynb (Apache-2.0).
# Do not edit: this is VegaML's own benchmark harness, used so every number is loaded and scored as theirs is.
"""Turn a registry entry into typed questions plus disjoint evaluation and fitting rows.

Rules that matter for the numbers:

* **Evaluation rows and fitting rows never overlap.** Where upstream ships one split only, the split
  is permuted once with the run seed and cut in half: fitting rows from the front, evaluation rows
  from the back. A head fitted on rows it is then scored on would make the TTT column meaningless.
* **Fitting rows are sampled class by class**, round robin, so a 50-label budget on a 77-class task
  spreads over as many classes as it can reach. It still cannot cover them all, and that is the
  finding rather than a flaw in the sampling.
* **Multiple-choice rows are kept at one option count per task** (the modal count), because one
  fitted head has to span a single set of option ids. Rows off that count are dropped and counted.
"""
import random

from bench_tasks import CHOICE_IDS, stable_hash


def _load(task, split, cache_dir=None):
    """Load one split, falling back to the Hub's converted parquet branch.

    `datasets` 4 refuses to run a dataset's loading script, and several of the canonical evaluation
    sets here still ship one. For most of them the Hub holds an auto-converted parquet branch at
    `refs/convert/parquet`, which loads with the same config names and keeps the ClassLabel names,
    so trying that before giving up recovers the task rather than skipping it.
    """
    from datasets import load_dataset
    kw = {"cache_dir": cache_dir} if cache_dir else {}
    args = (task["hf"], task["config"]) if task["config"] else (task["hf"],)
    try:
        return load_dataset(*args, split=split, **kw)
    except Exception as first:
        if "script" not in str(first).lower():
            raise
        try:
            return load_dataset(*args, split=split, revision="refs/convert/parquet", **kw)
        except Exception:
            raise first


def _label_names(task, ds):
    """The option ids, in a fixed order: the registry's override, the ClassLabel names, or the
    sorted unique values of a plain string column."""
    if task.get("labels"):
        return [str(x) for x in task["labels"]]
    col = task["label_col"]
    feat = ds.features.get(col)
    names = list(getattr(feat, "names", []) or [])
    if names:
        return [str(n) for n in names]
    vals = sorted({str(v) for v in ds[col]})
    return vals


def _gold_fixed(task, row, names, class_names=()):
    """The gold option id for a fixed-option task, whatever shape upstream stores it in."""
    raw = row[task["label_col"]]
    gm = task.get("gold_map")
    if gm == "nonempty_list":
        return "true" if raw else "false"
    if isinstance(gm, dict):
        hit = gm.get(raw, gm.get(str(raw)))
        if hit is None and isinstance(raw, int) and not isinstance(raw, bool):
            for pool in (class_names, names):
                if 0 <= raw < len(pool):
                    hit = gm.get(pool[raw], gm.get(str(pool[raw])))
                if hit is not None:
                    break
        return hit
    if task["qtype"] == "score":
        return str(int(raw))
    if isinstance(raw, bool):
        return "true" if raw else "false"
    if isinstance(raw, int):
        return names[raw] if 0 <= raw < len(names) else None
    return str(raw)


def _question_fixed(task, names):
    if task["qtype"] == "boolean":
        return {"type": "boolean", "instructions": task["instructions"],
                "criteria": task.get("criteria") or {"false": "no", "true": "yes"}}
    if task["qtype"] == "score":
        return {"type": "score", "instructions": task["instructions"],
                "criteria": list(task.get("criteria") or [])}
    crit = task.get("criteria") or {}
    return {"type": "choice", "instructions": task["instructions"],
            "criteria": {n: crit.get(n) for n in names}}


def _state_fixed(task, row):
    if task.get("state_fn"):
        return task["state_fn"](row, {})
    return {c: row[c] for c in task["state_cols"] if row.get(c) is not None}


def _row_id(task, row, i):
    for key in ("id", "row_id", "uid", "pubid", "conv_id", "question_id", "unique_id", "message_id"):
        if row.get(key) not in (None, ""):
            return "%s#%s" % (task["key"], row[key])
    return "%s#%d" % (task["key"], i)


def _roundrobin(by_label, n, rng):
    """Up to n rows, one label at a time, so every class gets a turn before any gets a second."""
    keys = sorted(by_label)
    rng.shuffle(keys)
    for v in by_label.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < n:
        took = False
        for k in keys:
            if i < len(by_label[k]):
                out.append(by_label[k][i])
                took = True
                if len(out) >= n:
                    break
        if not took:
            break
        i += 1
    return out


def load_task(task, n_eval=60, n_fit=50, seed=20261011, cache_dir=None, max_state_chars=12000):
    """-> dict(items, fit, question(s), meta) or dict(skipped=reason)."""
    rng = random.Random(seed ^ (stable_hash(task["key"]) & 0xFFFF))
    single = task["eval_split"] == task["train_split"]
    try:
        ds_eval = _load(task, task["eval_split"], cache_dir)
        ds_fit = ds_eval if single else _load(task, task["train_split"], cache_dir)
    except Exception as exc:                        # a moved, gated or script-only dataset
        return {"skipped": "load failed: %s: %s" % (type(exc).__name__, str(exc)[:200])}

    ctx = {"seed": seed}
    if task["key"] == "cruxeval_output":            # distractor pool, drawn once per run
        ctx["crux_pool"] = list(ds_eval["output"])[:2000]

    def rows_of(ds, want, front=True, wide=False):
        """Candidate row indices: the split permuted once with the run seed, halved when upstream
        ships a single split. `wide` scans far more candidates, which is what a 151-class fit set
        needs before round robin can reach every class."""
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        if single:
            half = len(idx) // 2
            idx = idx[:half] if front else idx[half:]
        cap = min(len(idx), max(want * (120 if wide else 6), want))
        return idx[:cap]

    meta = {"licence": task["licence"], "group": task["group"], "qtype": task["qtype"],
            "hf": task["hf"], "config": task["config"], "note": task.get("note"),
            "eval_split": task["eval_split"], "fit_split": task["train_split"],
            "single_split_halved": single, "rows_upstream_eval": len(ds_eval)}

    if task["kind"] == "fixed":
        src = ds_eval if single else ds_fit
        feat = src.features.get(task["label_col"])
        class_names = [str(n) for n in (getattr(feat, "names", []) or [])]
        names = _label_names(task, src)
        question = _question_fixed(task, names)
        labels = ["false", "true"] if task["qtype"] == "boolean" else (
            [str(i) for i in range(len(task.get("criteria") or []))] if task["qtype"] == "score"
            else list(question["criteria"]))

        def collect(ds, cand, want, stratify):
            by, flat = {}, []
            for i in cand:
                row = ds[i]
                gold = _gold_fixed(task, row, names, class_names)
                if gold is None or gold not in labels:
                    continue
                state = _state_fixed(task, row)
                st = {k: (v[:max_state_chars] if isinstance(v, str) else v) for k, v in state.items()}
                item = {"id": _row_id(task, row, i), "state": st, "gold": gold}
                flat.append(item)
                by.setdefault(gold, []).append(item)
            return _roundrobin(by, want, rng) if stratify else flat[:want]

        cand_fit = rows_of(ds_fit, n_fit, front=True, wide=True)
        cand_eval = rows_of(ds_eval, n_eval, front=False)
        fit = collect(ds_fit, cand_fit, n_fit, True)
        items = collect(ds_eval, cand_eval, n_eval, False)
        fit_ids = {f["id"] for f in fit}
        items = [it for it in items if it["id"] not in fit_ids][:n_eval]
        if len(items) < min(10, n_eval):
            return {"skipped": "only %d usable evaluation rows" % len(items)}
        meta.update(n_labels=len(labels), labels_preview=labels[:12],
                    upstream_label_names=class_names[:12] or None)
        return {"kind": "fixed", "question": question, "labels": labels,
                "items": items, "fit": fit, "meta": meta}

    # ---------------------------------------------------------------- per-row options
    def build(ds, cand, want):
        out = []
        for i in cand:
            row = ds[i]
            try:
                opts = [str(o) for o in task["options_fn"](row, ctx)]
                gold = task["gold_fn"](row, ctx)
                state = task["state_fn"](row, ctx)
            except Exception:
                continue
            if not opts or gold is None or gold not in CHOICE_IDS[:len(opts)]:
                continue
            state = {k: (v[:max_state_chars] if isinstance(v, str) else v) for k, v in state.items()}
            out.append({"id": _row_id(task, row, i), "state": state, "gold": gold,
                        "options": opts})
            if len(out) >= want:
                break
        return out

    cand_fit = rows_of(ds_fit, n_fit, front=True, wide=True)
    cand_eval = rows_of(ds_eval, n_eval, front=False)
    fit_all = build(ds_fit, cand_fit, n_fit * 3)
    items_all = build(ds_eval, cand_eval, n_eval * 3)
    counts = {}
    for it in items_all:
        counts[len(it["options"])] = counts.get(len(it["options"]), 0) + 1
    if not counts:
        return {"skipped": "no usable rows after option extraction"}
    k = max(counts, key=counts.get)
    dropped = sum(v for kk, v in counts.items() if kk != k)
    items = [it for it in items_all if len(it["options"]) == k][:n_eval]
    fit_ids = set()
    fit = []
    for it in fit_all:
        if len(it["options"]) == k and it["id"] not in fit_ids:
            fit.append(it)
            fit_ids.add(it["id"])
        if len(fit) >= n_fit:
            break
    items = [it for it in items if it["id"] not in fit_ids]
    if len(items) < min(10, n_eval):
        return {"skipped": "only %d rows at the modal option count %d" % (len(items), k)}
    labels = CHOICE_IDS[:k]
    for it in items + fit:
        it["question"] = {"type": "choice", "instructions": task["instructions"],
                          "criteria": {lab: opt for lab, opt in zip(labels, it["options"])}}
    meta.update(n_labels=k, option_count=k, rows_dropped_off_modal_k=dropped,
                labels_preview=labels)
    return {"kind": "perrow", "labels": labels, "items": items, "fit": fit, "meta": meta}
