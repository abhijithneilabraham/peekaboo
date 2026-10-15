# Vendored verbatim from NandhaKishorM/vegaml@00d3d99, notebooks/vega_benchmark_2xT4.ipynb (Apache-2.0).
# Do not edit: this is VegaML's own benchmark harness, used so every number is loaded and scored as theirs is.
"""Two third-party typed-decision suites, loaded at run time and never redistributed.

* **JevBench** public tiers (MIT, `fstandhartinger/jevbench`): already written as typed decisions,
  with paraphrase groups, so it also measures whether the same rule in different wordings gets the
  same answer. The published leaderboard uses these plus held-out items, so our numbers on the
  public tiers are comparable in kind, not identical in value.
* **DecisionBench** (`Hanno-Labs/decision-bench`): 43 tasks over three primitives, with gold
  probability distributions as well as gold labels.

Neither is used to fit anything. They are other people's evaluation sets held under their own
licences, and the one thing that would spoil them is training on them, so nothing here does: no
head is fitted on these rows, and the saved output holds row ids and metrics only.
"""
import json
import random
import urllib.request

JEVBENCH_RAW = ("https://raw.githubusercontent.com/fstandhartinger/jevbench/main/"
                "datasets/public/%s.jsonl")
JEV_TIERS = ("original", "easy", "hard")


# ----------------------------------------------------------------------------- JevBench
def fetch_jevbench(tiers=JEV_TIERS, timeout=60):
    """-> (items, meta). An unreachable tier is reported, not silently dropped."""
    items, errors = [], {}
    for tier in tiers:
        try:
            with urllib.request.urlopen(JEVBENCH_RAW % tier, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
        except Exception as exc:
            errors[tier] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
            continue
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except Exception:
                continue
            conv = _jev_item(raw, tier)
            if conv:
                items.append(conv)
    return items, {"source": "fstandhartinger/jevbench public tiers", "licence": "mit",
                   "tiers": list(tiers), "errors": errors, "n": len(items)}


def _jev_item(raw, tier):
    q = raw.get("question") or {}
    t = {"noul": "boolean", "choice": "choice", "score": "score"}.get(q.get("type"))
    if not t or "state" not in raw:
        return None
    question = {"type": t, "instructions": q.get("instructions") or "", "criteria": q.get("criteria")}
    expected = raw.get("expected")
    if t == "boolean":
        gold = {"yes": "true", "true": "true", "no": "false", "false": "false"}.get(
            str(expected).strip().lower())
        labels = ["false", "true"]
        if not question["criteria"]:
            question["criteria"] = {"false": "no", "true": "yes"}
    elif t == "score":
        crit = list(question["criteria"] or [])
        labels = [str(i) for i in range(len(crit))]
        s = str(expected).strip()
        gold = s if s in labels else (str(crit.index(expected)) if expected in crit else None)
    else:
        crit = question["criteria"] or {}
        labels = [str(k) for k in crit]
        gold = str(expected) if str(expected) in labels else None
    if gold is None:
        return None
    return {"id": "jevbench:%s:%s" % (tier, raw.get("id")), "state": raw["state"], "gold": gold,
            "question": question, "labels": labels, "tier": tier,
            "family": raw.get("family"), "group": raw.get("group"), "qtype": t}


def paraphrase_consistency(items, records):
    """The share of paraphrase groups answered the same way throughout.

    A group is one rule written several ways. Scoring only accuracy hides a model that is right on
    one wording and wrong on its twin, which in a pipeline is the same thing as being unreliable.
    """
    by_id = {r["id"]: r for r in records}
    groups = {}
    for it in items:
        if it.get("group"):
            groups.setdefault(it["group"], []).append(it)
    whole, consistent, pairs_same = 0, 0, []
    for members in groups.values():
        preds = [by_id.get(m["id"], {}).get("pred") for m in members if m["id"] in by_id]
        if len(preds) < 2:
            continue
        whole += 1
        consistent += int(len(set(preds)) == 1)
        pairs_same.append(len(set(preds)) == 1)
    if not whole:
        return {}
    return {"paraphrase_groups": whole,
            "paraphrase_consistency": round(consistent / whole, 4)}


# ----------------------------------------------------------------------------- DecisionBench
# The affirmative and negative sides of a two-candidate row, matched on the first word of the
# label. Measured against the real label pairs in the suite: Yes/No and Relevant/Not relevant
# account for most of them, "Yes - the evidence is sufficient" style labels are caught by the first
# word, and genuinely non-polar pairs such as Retain verbatim / Summarize fall through on purpose
# and are asked as a two-option choice instead.
_POS = {"yes", "true", "relevant", "safe", "permitted", "permit", "allowed", "allow", "approve",
        "approved", "accept", "accepted", "retain", "valid", "compliant", "sufficient"}
_NEG = {"no", "false", "irrelevant", "unsafe", "prohibited", "prohibit", "denied", "deny",
        "reject", "rejected", "discard", "invalid", "noncompliant", "insufficient", "block",
        "blocked"}


def fetch_decisionbench(per_task=30, seed=20261011, cache_dir=None, max_state_chars=12000,
                        binary_as_boolean=True):
    """-> (items, meta). `per_task` rows per task_id, sampled with a fixed seed."""
    from datasets import load_dataset

    kw = {"cache_dir": cache_dir} if cache_dir else {}
    ds = load_dataset("Hanno-Labs/decision-bench", split="eval", **kw)
    by_task = {}
    for i, task in enumerate(ds["task_id"]):
        by_task.setdefault(task, []).append(i)
    rng = random.Random(seed)
    picked = []
    for task in sorted(by_task):
        idx = by_task[task]
        rng.shuffle(idx)
        picked.extend(idx[:per_task])
    items, skipped, as_bool = [], 0, 0
    primitives = {}
    for i in sorted(picked):
        row = ds[i]
        try:
            cands = json.loads(row["candidates_json"])
            state = json.loads(row["state_json"])
        except Exception:
            skipped += 1
            continue
        if not isinstance(cands, list) or len(cands) < 2:
            skipped += 1
            continue
        ids = [str(c.get("id")) for c in cands]
        labs = [str(c.get("label") or c.get("id")) for c in cands]
        descs = [c.get("description") for c in cands]
        gold_i = ids.index(str(row["gold_candidate_id"])) if str(row["gold_candidate_id"]) in ids \
            else None
        if gold_i is None:
            skipped += 1
            continue
        prim = row["primitive"]
        primitives[prim] = primitives.get(prim, 0) + 1
        soft = list(row.get("gold_probabilities") or [])
        text = json.dumps(state, ensure_ascii=False)[:max_state_chars]
        if prim.startswith("binary") and binary_as_boolean and _polarity(labs) is not None:
            pos = _polarity(labs)
            question = {"type": "boolean", "instructions": row["instruction"],
                        "criteria": {"true": labs[pos], "false": labs[1 - pos]}}
            labels = ["false", "true"]
            gold = "true" if gold_i == pos else "false"
            soft = [soft[1 - pos], soft[pos]] if len(soft) == 2 else []
            as_bool += 1
        else:
            crit = {}
            for lab, desc, cid in zip(labs, descs, ids):
                key = lab if lab not in crit else "%s (%s)" % (lab, cid)
                crit[key] = desc
            question = {"type": "choice", "instructions": row["instruction"], "criteria": crit}
            labels = list(crit)
            gold = labels[gold_i]
        items.append({"id": "decisionbench:%s" % row["row_id"], "state": text, "gold": gold,
                      "question": question, "labels": labels, "qtype": question["type"],
                      "task": row["task_id"], "primitive": prim, "family": row.get("family"),
                      "domain": row.get("domain"), "candidate_count": len(cands),
                      "reasoning_required": bool(row.get("reasoning_required")),
                      "soft_gold": [float(x) for x in soft] if len(soft) == len(labels) else None})
    return items, {"source": "Hanno-Labs/decision-bench", "licence": "other-see-card",
                   "ordinal_rows_asked_as_choice": True,
                   "ordinal_note": ("ordinal_scoring rows are asked as a choice, not as a score. "
                                    "Their candidate lists are shuffled per row and the stored "
                                    "ordinal_value follows that position rather than the severity "
                                    "of the label, so one row runs Negligible to Critical, the "
                                    "next runs Critical to Negligible, and a third interleaves "
                                    "them. A score-typed question asserts that level i is below "
                                    "level i+1, which these rows do not carry, so asking them as "
                                    "a score would score the harness and not the model. The score "
                                    "type is exercised on JevBench's ordinal family and on the "
                                    "ordinal tasks in the public set instead."),
                   "split": "eval", "per_task": per_task, "tasks": len(by_task),
                   "rows_upstream": len(ds), "n": len(items), "rows_skipped": skipped,
                   "binary_rows_asked_as_boolean": as_bool,
                   "primitives_seen": primitives,
                   "binary_note": ("a two-candidate row is asked as a yes/no only when one of its "
                                   "labels is clearly the affirmative side. Pairs that are not "
                                   "polar, such as Retain verbatim against Summarize, are asked "
                                   "as a two-option choice, because guessing the polarity would "
                                   "invert the gold answer for that row.")}


def _first_word(label):
    import re
    m = re.search(r"[A-Za-z]+", str(label))
    return m.group(0).lower() if m else ""


def _polarity(labels):
    """Which of two candidate labels is the affirmative one, or None when neither clearly is.

    None is the safe answer: the row is then asked as a two-option choice, which decides nothing
    about polarity. Guessing wrong would invert the gold for that row.
    """
    if len(labels) != 2:
        return None
    low = [str(x).strip().lower() for x in labels]
    if low[0] == low[1]:
        return None
    for i in (0, 1):                              # "Relevant" against "Not relevant"
        other = low[1 - i]
        if other in ("not " + low[i], "non-" + low[i], "no " + low[i], low[i] + " not"):
            return i
    heads = [_first_word(x) for x in labels]
    for i in (0, 1):
        if heads[i] in _POS and heads[1 - i] in _NEG:
            return i
    return None


def soft_gold_kl(items, records):
    """Mean KL(gold || predicted) over rows whose gold is a distribution rather than a single label.

    DecisionBench ships gold probabilities, so on the rows where the gold is genuinely uncertain a
    model that commits hard is wrong in a way accuracy cannot see.
    """
    import math

    by_id = {r["id"]: r for r in records}
    kls, n_soft = [], 0
    for it in items:
        soft = it.get("soft_gold")
        rec = by_id.get(it["id"])
        if not soft or not rec or not rec.get("probs"):
            continue
        if max(soft) >= 0.999:            # one-hot gold: KL would just restate the NLL
            continue
        n_soft += 1
        kl = 0.0
        for lab, g in zip(it["labels"], soft):
            if g <= 0:
                continue
            p = max(float(rec["probs"].get(lab, 0.0)), 1e-12)
            kl += g * math.log(g / p)
        kls.append(kl)
    if not kls:
        return {"soft_gold_rows": n_soft}
    return {"soft_gold_rows": n_soft,
            "soft_gold_kl_mean": round(float(sum(kls) / len(kls)), 4)}
