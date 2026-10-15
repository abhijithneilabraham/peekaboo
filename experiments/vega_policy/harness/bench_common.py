# Vendored verbatim from NandhaKishorM/vegaml@00d3d99, notebooks/vega_benchmark_2xT4.ipynb (Apache-2.0).
# Do not edit: this is VegaML's own benchmark harness, used so every number is loaded and scored as theirs is.
"""Metrics, typed-question helpers and the prompt-budget audit for the Vega benchmark run.

Everything here is plain Python and numpy so it can be unit tested off the GPU. Three rules hold
throughout:

* A readout that cannot answer scores as wrong, and the share of answered rows is reported as
  `coverage`. Dropping unanswerable rows would flatter whichever readout fails loudest.
* Probabilities are kept, not just the argmax, so calibration is measurable. A readout that only
  emits a label gets `ece = None` rather than a fabricated zero.
* No source text leaves this harness. The saved JSON holds row ids, metrics and probabilities. The
  evaluation sets carry other people's licences and are not ours to redistribute.
"""
import json
import math
import os
import sys
import time

import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------------- typed questions
def label_ids(q):
    """The option ids a readout must span, for any of the three answer types."""
    t = q["type"]
    if t == "boolean":
        return ["false", "true"]
    if t == "score":
        return [str(i) for i in range(len(q.get("criteria") or []))]
    return [str(k) for k in q["criteria"]]


def option_texts(q):
    """What each option looks like in a prompt: the id plus its criterion when one is given."""
    t, crit = q["type"], q.get("criteria")
    if t == "choice":
        return [str(k) if not v else "%s - %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d - %s" % (i, c) for i, c in enumerate(crit or [])]
    crit = crit or {}
    return ["false - " + (crit.get("false") or "no, the statement does not hold"),
            "true - " + (crit.get("true") or "yes, the statement holds")]


def serialize_state(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def chance_level(q):
    """The accuracy a uniform guess reaches, which is what a many-class score has to be read against."""
    return 1.0 / max(len(label_ids(q)), 2)


# ----------------------------------------------------------------------------- budget audit
def budget_audit(tok, questions, opt_max_tokens=32, ins_max_tokens=256):
    """How much of each question survives Vega's option and instruction caps.

    `build_prompt` truncates an option at `opt_max_tokens` and an instruction at `ins_max_tokens`
    rather than refusing, so a task with long options is quietly asked a different question than the
    one on paper. Worse, two options that share a long prefix can truncate to the same tokens, and
    then the task is unanswerable by construction. Both are measured here and reported per task, so
    a low score on CaseHOLD or MMLU-Pro is read as a budget limit rather than as reasoning.
    """
    n_opt = n_over = n_collide = 0
    ins_over = 0
    opt_lens, ins_lens = [], []
    for q in questions:
        ins = tok(str(q["instructions"]), add_special_tokens=False)["input_ids"]
        ins_lens.append(len(ins))
        ins_over += int(len(ins) > ins_max_tokens)
        seen = {}
        for text in option_texts(q):
            ids = tok(text, add_special_tokens=False)["input_ids"]
            opt_lens.append(len(ids))
            n_opt += 1
            n_over += int(len(ids) > opt_max_tokens)
            key = tuple(ids[:opt_max_tokens])
            seen[key] = seen.get(key, 0) + 1
        n_collide += sum(v - 1 for v in seen.values() if v > 1)
    pct = lambda a, b: round(a / b, 4) if b else 0.0
    return {"questions": len(questions), "options": n_opt,
            "options_over_cap": n_over, "options_over_cap_frac": pct(n_over, n_opt),
            "option_tokens_p50": int(np.percentile(opt_lens, 50)) if opt_lens else 0,
            "option_tokens_p95": int(np.percentile(opt_lens, 95)) if opt_lens else 0,
            "option_tokens_max": int(max(opt_lens)) if opt_lens else 0,
            "truncation_collisions": n_collide,
            "truncation_collisions_frac": pct(n_collide, n_opt),
            "instructions_over_cap": ins_over,
            "instruction_tokens_p95": int(np.percentile(ins_lens, 95)) if ins_lens else 0}


# ----------------------------------------------------------------------------- metrics
def _vec(probs, labels):
    """A probability vector over `labels`, renormalised, with missing options read as zero."""
    p = np.array([float(probs.get(k, 0.0)) for k in labels], dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    return p / s if s > 0 else np.full(len(labels), 1.0 / max(len(labels), 1))


def ece(conf, correct, bins=15):
    """Expected calibration error on the top-label confidence, equal-width bins."""
    conf, correct = np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if not sel.any():
            continue
        out += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(out)


def aurc(conf, correct):
    """Area under the risk-coverage curve: the error rate you keep as you drop the least confident
    answers first. Lower is better, and unlike accuracy it rewards knowing which answers to doubt."""
    conf, correct = np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return None
    order = np.argsort(-conf)
    err = 1.0 - correct[order]
    risks = np.cumsum(err) / np.arange(1, err.size + 1)
    return float(risks.mean())


def macro_f1(gold, pred, labels):
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else None


def score_metrics(records, labels=None, qtype="choice", chance=None):
    """Every metric the suite reports, from a list of per-row records.

    A record is {"gold": id, "pred": id or None, "probs": {id: p} or None, "latency_ms": float,
    "abstain": bool or None, "set_size": int or None}. `pred = None` means the readout could not
    answer, which counts as wrong and lowers coverage.

    `labels` is the option set when every row shares one. Rows whose option set is their own (one
    DecisionBench row offers "Negligible/Low/Moderate", the next offers something else) carry
    "labels" and "chance" on the record, and then the distribution metrics are computed row by row
    and the label-set metrics (macro F1) are left out rather than computed over a union that no
    single row was asked.
    """
    n = len(records)
    if not n:
        return {"n": 0}
    answered = [r for r in records if r.get("pred") is not None]
    correct_all = [1.0 if r.get("pred") == r["gold"] else 0.0 for r in records]
    acc = float(np.mean(correct_all))
    row_labels = [r.get("labels") or labels or [] for r in records]
    chances = [r.get("chance", 1.0 / max(len(ls), 2)) for r, ls in zip(records, row_labels)]
    ch = float(np.mean(chances)) if chance is None else float(chance)
    out = {"n": n, "answered": len(answered), "coverage": round(len(answered) / n, 4),
           "accuracy": round(acc, 4), "chance": round(ch, 4),
           "skill": round(max(0.0, (acc - ch) / (1.0 - ch)) if ch < 1 else 0.0, 4)}
    if labels:
        out["macro_f1"] = macro_f1([r["gold"] for r in records],
                                   [r.get("pred") for r in records], labels)
    lat = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    if lat:
        out.update(latency_ms_p50=round(float(np.percentile(lat, 50)), 1),
                   latency_ms_p95=round(float(np.percentile(lat, 95)), 1),
                   latency_ms_mean=round(float(np.mean(lat)), 1))
    withp = [(r, ls) for r, ls in zip(records, row_labels) if r.get("probs") and ls]
    if withp:
        conf, corr, nlls, briers = [], [], [], []
        for r, ls in withp:
            p = _vec(r["probs"], ls)
            top = int(p.argmax())
            conf.append(float(p[top]))
            corr.append(1.0 if ls[top] == r["gold"] else 0.0)
            if r["gold"] in ls:
                gi = ls.index(r["gold"])
                nlls.append(-math.log(max(float(p[gi]), EPS)))
                onehot = np.zeros(len(ls))
                onehot[gi] = 1.0
                briers.append(float(((p - onehot) ** 2).sum()))
        out.update(ece=round(ece(conf, corr), 4),
                   nll=round(float(np.mean(nlls)), 4) if nlls else None,
                   brier=round(float(np.mean(briers)), 4) if briers else None,
                   aurc=round(aurc(conf, corr), 4),
                   mean_confidence=round(float(np.mean(conf)), 4),
                   rows_with_probabilities=len(withp))
    if qtype == "score" and labels:
        idx = {lab: i for i, lab in enumerate(labels)}
        pairs = [(idx.get(r["gold"]), idx.get(r.get("pred"))) for r in records]
        pairs = [(g, p) for g, p in pairs if g is not None and p is not None]
        if pairs:
            out["ordinal_mae"] = round(float(np.mean([abs(g - p) for g, p in pairs])), 4)
    ab = [r["abstain"] for r in records if r.get("abstain") is not None]
    if ab:
        out["abstain_rate"] = round(float(np.mean(ab)), 4)
        kept = [r for r in records if not r.get("abstain")]
        if kept:
            out["accuracy_when_not_abstaining"] = round(
                float(np.mean([1.0 if r.get("pred") == r["gold"] else 0.0 for r in kept])), 4)
    sets = [r["set_size"] for r in records if r.get("set_size")]
    if sets:
        out["conformal_set_size_mean"] = round(float(np.mean(sets)), 3)
        hit = [1.0 if r.get("set_hit") else 0.0 for r in records if r.get("set_size")]
        out["conformal_coverage"] = round(float(np.mean(hit)), 4)
    err = [r for r in records if r.get("error")]
    if err:
        out["rows_with_errors"] = len(err)
        out["first_error"] = str(err[0]["error"])[:200]
    trunc = [r for r in records if r.get("state_truncated")]
    if trunc:
        out["rows_state_truncated"] = len(trunc)
    return out


def trivial_baselines(golds, labels):
    """What majority class and a uniform guess score on the same rows. Any readout below these on a
    task has told you nothing, and on a 77-class task the difference is easy to miss by eye."""
    if not golds:
        return {}
    counts = {lab: golds.count(lab) for lab in set(golds)}
    top = max(counts, key=counts.get)
    k = max(len(labels), 2)
    return {"majority_class": top, "majority_accuracy": round(counts[top] / len(golds), 4),
            "uniform_accuracy": round(1.0 / k, 4),
            "uniform_nll": round(float(math.log(k)), 4),
            "label_entropy": round(float(-sum((c / len(golds)) * math.log(c / len(golds))
                                              for c in counts.values())), 4)}


# ----------------------------------------------------------------------------- misc
class Budget:
    """A wall-clock guard. Kaggle stops a committed notebook at 12 hours, and a run that dies at
    hour 11 with nothing written is a wasted day, so every phase checks this and stops early."""

    def __init__(self, seconds):
        self.t0, self.limit = time.time(), float(seconds)

    def left(self):
        return self.limit - (time.time() - self.t0)

    def spent(self):
        return time.time() - self.t0

    def ok(self, need=0.0):
        return self.left() > need


# ----------------------------------------------------------------------------- host memory
def memory_gb():
    """This process's resident size and how much room the container has left, in GB.

    A Kaggle kernel that outgrows the machine is killed outright: the notebook log shows
    "Kernel died while waiting for execute reply" and no Python traceback, because the process
    never got to raise anything. The only way to see that coming is to record these numbers
    before and after every model load, so this is called at every phase boundary.
    """
    out = {"rss_gb": None, "available_gb": None, "total_gb": None,
           "cgroup_used_gb": None, "cgroup_limit_gb": None, "cgroup_cache_gb": None,
           "cgroup_committed_gb": None}
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    out["rss_gb"] = round(int(line.split()[1]) / 1048576.0, 2)
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    out["available_gb"] = round(int(line.split()[1]) / 1048576.0, 2)
                elif line.startswith("MemTotal:"):
                    out["total_gb"] = round(int(line.split()[1]) / 1048576.0, 2)
    except OSError:
        pass
    if out["rss_gb"] is None:                      # not Linux: good enough for a dry run
        try:
            import resource
            raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss is bytes on darwin and kilobytes on linux, and it is a peak rather than
            # the current size, so this branch is only a sanity figure for the dry run
            out["rss_gb"] = round(raw / (1073741824.0 if sys.platform == "darwin"
                                         else 1048576.0), 2)
        except Exception:
            pass
    out.update(_cgroup_memory())
    return out


def _cgroup_memory():
    """The container's own accounting, which is what the kill decision uses.

    /proc/meminfo reports the host, and on a shared machine the host can look roomy while this
    container is one allocation from being killed. cgroup v2 first, then v1.

    `used` counts the page cache this container caused, and reading an 11 GB checkpoint charges
    all of it. That cache is clean and reclaimable, so `limit - used` is not the room left: on
    the resumed run it read 3.4 GB while the host still had 25 GB available, and a load that
    would have fitted was refused. `committed` takes the cache back out and is the number to
    compare an allocation against.
    """
    v2 = ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max",
          "/sys/fs/cgroup/memory.stat", ("file", "slab_reclaimable"))
    v1 = ("/sys/fs/cgroup/memory/memory.usage_in_bytes",
          "/sys/fs/cgroup/memory/memory.limit_in_bytes",
          "/sys/fs/cgroup/memory/memory.stat", ("total_cache",))
    for cur, mx, stat, cache_keys in (v2, v1):
        try:
            used = int(open(cur).read().strip()) / 1073741824.0
        except (OSError, ValueError):
            continue
        limit = None
        try:
            raw = open(mx).read().strip()
            if raw != "max":
                val = int(raw)
                if val < 2 ** 50:                  # v1 writes a sentinel near 2**63 for "no limit"
                    limit = round(val / 1073741824.0, 2)
        except (OSError, ValueError):
            pass
        cache = None
        try:
            fields = {}
            for line in open(stat):
                parts = line.split()
                if len(parts) == 2:
                    fields[parts[0]] = int(parts[1])
            if any(k in fields for k in cache_keys):
                cache = sum(fields.get(k, 0) for k in cache_keys) / 1073741824.0
        except (OSError, ValueError):
            pass
        return {"cgroup_used_gb": round(used, 2), "cgroup_limit_gb": limit,
                "cgroup_cache_gb": None if cache is None else round(cache, 2),
                "cgroup_committed_gb": None if cache is None
                else round(max(used - cache, 0.0), 2)}
    return {}


def headroom_gb():
    """GB that can still be allocated before the kernel is killed, or None when unknowable.

    The smaller of what the container has left and what the host has available: either one
    running out ends the process. The container's own figure counts out its reclaimable page
    cache, since the kernel drops that rather than killing the process, and only falls back to
    the raw usage when the cgroup does not break it down.
    """
    m = memory_gb()
    room = []
    if m["cgroup_limit_gb"] is not None:
        used = m["cgroup_committed_gb"]
        if used is None:
            used = m["cgroup_used_gb"]
        if used is not None:
            room.append(m["cgroup_limit_gb"] - used)
    if m["available_gb"] is not None:
        room.append(m["available_gb"])
    return round(min(room), 2) if room else None


def memory_line(label=""):
    m = memory_gb()
    head = headroom_gb()
    return ("%srss %s GB, container %s of %s GB (%s cache, %s committed), host available %s GB, "
            "headroom %s GB"
            % (label + ": " if label else "", m["rss_gb"], m["cgroup_used_gb"],
               m["cgroup_limit_gb"], m["cgroup_cache_gb"], m["cgroup_committed_gb"],
               m["available_gb"], head))


def drop_file_cache(*paths):
    """Ask the kernel to forget the page cache for files under these paths.

    Reading the two backbones charges about 11 GB of clean page cache to the container, and the
    weights are on the GPU afterwards, so nothing needs those pages again. Dropping them lowers
    the number the kill decision reads. Best effort: pages still mapped stay, and this is a hint
    to the kernel rather than a command, so the caller reports the before and after rather than
    assuming an amount.
    """
    freed = 0
    if not hasattr(os, "posix_fadvise"):
        return 0
    for root in paths:
        if not root or not os.path.exists(root):
            continue
        walk = [(root, [], [os.path.basename(root)])] if os.path.isfile(root) else os.walk(root)
        base = os.path.dirname(root) if os.path.isfile(root) else None
        for folder, _, files in walk:
            for name in files:
                full = os.path.join(base if base is not None else folder, name)
                if not full.endswith((".safetensors", ".bin", ".pt", ".gguf")):
                    continue
                try:
                    fd = os.open(full, os.O_RDONLY)
                except OSError:
                    continue
                try:
                    size = os.fstat(fd).st_size
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    freed += size
                except OSError:
                    pass
                finally:
                    os.close(fd)
    return round(freed / 1073741824.0, 2)


def jsonable(obj):
    """numpy scalars and arrays are not JSON serialisable, and silently stringifying them later
    makes the output unusable for plots. Convert once, here."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)
