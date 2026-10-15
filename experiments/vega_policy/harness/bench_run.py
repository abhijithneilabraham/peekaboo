# Vendored verbatim from NandhaKishorM/vegaml@00d3d99, notebooks/vega_benchmark_2xT4.ipynb (Apache-2.0).
# Do not edit: this is VegaML's own benchmark harness, used so every number is loaded and scored as theirs is.
"""Orchestration: run the readouts over a suite, collect metrics, save after every task.

The result file is written incrementally. Kaggle stops a committed notebook at twelve hours and the
4B is the slow part, so a run that is cut short still leaves every task it finished, with a
`completed` flag saying so. Nothing here deletes or rewrites earlier phases, so the notebook can
call the phases cell by cell and watch them land.
"""
import gc
import json
import os
import time

import bench_common as bc
import bench_readouts as RO


class Results:
    def __init__(self, cfg, path, resume_from=None):
        self.path = path
        self.data = {"schema": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "config": cfg, "environment": {}, "suites": {}, "results": {}, "phases": {},
                     "notes": [], "completed": False}
        if resume_from:
            self.resume(resume_from)

    def resume(self, src):
        """Carry a previous run's finished readouts forward so they are not paid for twice.

        A kernel that dies loses the session but not the file: the output JSON is written after
        every task. Attach the previous version's output as an input dataset, point
        `CFG["resume_from"]` at it, and every readout already in it is skipped. Nothing is
        recomputed and nothing already measured is overwritten, so the two runs' numbers sit in
        one file exactly as they were measured.
        """
        if not os.path.exists(src):
            self.note("resume_from %s does not exist, starting from nothing" % src)
            return
        with open(src) as fh:
            old = json.load(fh)
        self.data["results"] = old.get("results") or {}
        self.data["suites"] = old.get("suites") or {}
        self.data["phases"] = {("prior_" + k): v for k, v in (old.get("phases") or {}).items()}
        self.data["notes"] = ["prior run: " + n for n in (old.get("notes") or [])]
        self.data["resumed_from"] = {
            "path": src, "created": old.get("created"), "completed": old.get("completed"),
            "wall_clock_minutes": old.get("wall_clock_minutes"),
            "environment": old.get("environment"), "config": old.get("config")}
        self.note("resumed from %s: %d readouts already measured, %d tasks"
                  % (src, self.n_metrics(), sum(len(v) for v in self.data["results"].values())))
        redo = list(self.data["config"].get("redo") or [])
        if redo:
            self.drop(redo)
        self.save()

    def drop(self, readouts):
        """Forget named readouts so a resumed run measures them again.

        A carried-forward number that turned out to be measured wrongly has to go, or the resume
        guard will protect it forever. Naming it in `CFG["redo"]` removes it from every task and
        every slice, and the removal is recorded in the notes rather than done silently.
        """
        gone = 0
        for suite in self.data["results"].values():
            for rs in suite.values():
                for name in list(rs):
                    if name in readouts:
                        del rs[name]
                        gone += 1
        self.note("redo: dropped %d carried-forward readouts %s" % (gone, sorted(set(readouts))))

    def n_metrics(self):
        return sum(len([k for k in rs if k != "_meta"])
                   for suite in self.data["results"].values() for rs in suite.values())

    def has(self, suite, key, *readouts):
        """True when every named readout for this task is already in the file.

        The resume guards ask this before spending GPU time, so they name the readout a phase
        writes last: if that one is present the whole phase finished.
        """
        rs = (self.data["results"].get(suite) or {}).get(key) or {}
        return bool(readouts) and all(r in rs for r in readouts)

    def note(self, text):
        self.data["notes"].append(text)
        print("note:", text, flush=True)

    def put(self, suite, key, readout, metrics):
        self.data["results"].setdefault(suite, {}).setdefault(key, {})[readout] = metrics
        self.save()

    def meta(self, suite, key, meta):
        self.data["results"].setdefault(suite, {}).setdefault(key, {})["_meta"] = meta
        self.save()

    def phase(self, name, **kw):
        self.data["phases"][name] = dict(kw, at=time.strftime("%H:%M:%S", time.gmtime()),
                                         memory=bc.memory_gb())
        self.save()
        print("phase %-24s %s" % (name, bc.memory_line()), flush=True)

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(bc.jsonable(self.data), fh, indent=1)
        os.replace(tmp, self.path)


def _progress(label, i, n, t0):
    """One line per block of rows, carrying the host memory left.

    The Kaggle log is the only thing left after a kernel is killed, so the headroom goes on the
    progress line: a run that creeps towards zero over an hour and then dies says what happened,
    and a run that dies with room to spare says it was something else.
    """
    rate = (time.time() - t0) / max(i, 1)
    head = bc.headroom_gb()
    print("    %s %d/%d  %.0f ms/row  eta %.0f s%s"
          % (label, i, n, rate * 1000, rate * (n - i),
             "" if head is None else "  free %.1f GB" % head), flush=True)


def run_records(label, items, fn, budget, every=25):
    """Apply one readout to every item, printing progress and stopping if the clock runs out."""
    out, t0 = [], time.time()
    for i, it in enumerate(items, 1):
        if not budget.ok(10):
            print("    %s stopped early at %d/%d: out of time" % (label, i, len(items)), flush=True)
            break
        try:
            out.append(fn(it))
        except Exception as exc:
            out.append({"id": it["id"], "gold": it["gold"], "pred": None, "probs": None,
                        "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])})
        if i % every == 0 or i == len(items):
            _progress(label, i, len(items), t0)
    return out


def question_of(item, shared):
    return shared if shared is not None else item["question"]


def labels_of(item, shared):
    return shared if shared is not None else item["labels"]


# ----------------------------------------------------------------------------- Vega phases
def vega_task(R, suite, key, items, fit_items, shared_q, shared_labels, qtype, vr, cfg, budget,
              do_heads=True, do_flip=False, views=()):
    """Every Vega readout for one task, under the model's own name."""
    tag = "vega-%s" % vr.size
    chance = None
    if shared_labels:
        chance = 1.0 / max(len(shared_labels), 2)

    def add(name, records, extra=None):
        m = bc.score_metrics(records, shared_labels, qtype, chance)
        if extra:
            m.update(extra)
        R.put(suite, key, "%s/%s" % (tag, name), m)
        return m

    eng = run_records("%s engine" % tag, items,
                      lambda it: vr.engine_record(it, question_of(it, shared_q),
                                                  labels_of(it, shared_labels)), budget)
    for rec, it in zip(eng, items):
        if not shared_labels:
            rec["labels"] = it["labels"]
    add("engine", eng)

    if cfg.get("run_no_adapter", True) and vr.adapter_names and budget.ok(60):
        vr.set_adapters("off")
        try:
            recs = run_records("%s engine_no_adapter" % tag, items,
                               lambda it: vr.engine_record(it, question_of(it, shared_q),
                                                           labels_of(it, shared_labels)), budget)
        finally:
            vr.set_adapters("auto")
        for rec, it in zip(recs, items):
            if not shared_labels:
                rec["labels"] = it["labels"]
        add("engine_no_adapter", recs)

    for v in views:
        if not budget.ok(120):
            break
        recs = run_records("%s engine_views%d" % (tag, v), items,
                           lambda it: vr.engine_record(it, question_of(it, shared_q),
                                                       labels_of(it, shared_labels), views=v),
                           budget)
        for rec, it in zip(recs, items):
            if not shared_labels:
                rec["labels"] = it["labels"]
        add("engine_views%d" % v, recs)

    if do_flip and budget.ok(60):
        def flipped(it):
            q = dict(question_of(it, shared_q))
            if q["type"] == "choice":
                q["criteria"] = {k: q["criteria"][k] for k in reversed(list(q["criteria"]))}
            elif q["type"] == "score":
                return vr.engine_record(it, q, labels_of(it, shared_labels))
            return vr.engine_record(it, q, labels_of(it, shared_labels))
        recs = run_records("%s engine_reversed_options" % tag, items, flipped, budget)
        for rec, it in zip(recs, items):
            if not shared_labels:
                rec["labels"] = it["labels"]
        by = {r["id"]: r for r in recs}
        flips = [1.0 for r in eng if r["id"] in by and by[r["id"]]["pred"] != r["pred"]]
        n_cmp = len([r for r in eng if r["id"] in by])
        add("engine_reversed_options", recs,
            {"option_order_flip_rate": round(len(flips) / n_cmp, 4) if n_cmp else None,
             "option_order_compared": n_cmp})

    if not (do_heads and fit_items and cfg.get("run_ttt", True) and budget.ok(120)):
        return
    # One read per fitting row, one per evaluation row, then every head budget scored off the same
    # features. The learning curve is therefore nearly free once the features exist.
    t0 = time.time()
    Xf = vr.features_for(fit_items, shared_q)
    yf = [it["gold"] for it in fit_items]
    fit_seconds = time.time() - t0
    labels_for_head = shared_labels or fit_items[0]["labels"]
    ladder = RO.fit_head_ladder(Xf, yf, labels_for_head, cfg["shots"], folds=cfg.get("folds", 5))
    R.put(suite, key, "%s/ttt_fit_report" % tag,
          {"fit_rows": len(fit_items), "fit_seconds": round(fit_seconds, 1),
           "shots": {str(k): {kk: vv for kk, vv in v.items() if kk != "head"}
                     for k, v in ladder.items()}})
    Xe = vr.features_for(items, shared_q)
    best_k, best_info = None, None
    for k in sorted(ladder):
        info = ladder[k]
        if not info.get("fitted"):
            continue
        head = info["head"]
        recs = [RO.head_record(head, x, it, labels_for_head,
                               allowed=labels_of(it, shared_labels)) for x, it in zip(Xe, items)]
        for rec, it in zip(recs, items):
            if not shared_labels:
                rec["labels"] = it["labels"]
        add("ttt_%d" % k, recs, {"cv_accuracy": info.get("cv_accuracy"),
                                 "classes_in_fit": info.get("classes_seen"),
                                 "beats_chance": info.get("beats_chance")})
        best_k, best_info = k, info
    if best_info is not None:
        head = best_info["head"]
        auto = []
        for x, it, e in zip(Xe, items, eng):
            hr = RO.head_record(head, x, it, labels_for_head,
                                allowed=labels_of(it, shared_labels))
            auto.append(RO.auto_record(e, hr, best_info))
        for rec, it in zip(auto, items):
            if not shared_labels:
                rec["labels"] = it["labels"]
        add("auto", auto, {"head_used": bool(best_info.get("beats_chance")), "head_shots": best_k})


# ----------------------------------------------------------------------------- base phases
def base_task(R, suite, key, items, shared_q, shared_labels, qtype, base, cfg, budget):
    tag = "base-%s" % cfg["base_tag"]
    chance = 1.0 / max(len(shared_labels), 2) if shared_labels else None

    def add(name, records, extra=None):
        m = bc.score_metrics(records, shared_labels, qtype, chance)
        if extra:
            m.update(extra)
        R.put(suite, key, "%s/%s" % (tag, name), m)

    if cfg.get("run_base_logprob", True) and budget.ok(60) \
            and not R.has(suite, key, "%s/logprob" % tag):
        pairs = run_records("%s logprob" % tag, items,
                            lambda it: base.logprob_record(it, question_of(it, shared_q),
                                                            labels_of(it, shared_labels)), budget)
        mean_recs = [p[0] if isinstance(p, tuple) else p for p in pairs]
        sum_recs = [p[1] if isinstance(p, tuple) else p for p in pairs]
        for recs in (mean_recs, sum_recs):
            for rec, it in zip(recs, items):
                if not shared_labels:
                    rec["labels"] = it["labels"]
        add("logprob", mean_recs)
        add("logprob_unnormalised", sum_recs)

    if cfg.get("run_base_generate", True) and budget.ok(60) \
            and not R.has(suite, key, "%s/generate" % tag):
        recs = run_records("%s generate" % tag, items,
                           lambda it: base.generate_record(it, question_of(it, shared_q),
                                                            labels_of(it, shared_labels),
                                                            max_new_tokens=cfg.get("gen_tokens", 12)),
                           budget)
        for rec, it in zip(recs, items):
            if not shared_labels:
                rec["labels"] = it["labels"]
        parsed = [r for r in recs if r.get("parsed")]
        # parse_rate alone cannot tell "the weights answered something unusable" from "the prompt
        # left a reasoning block open and the budget went on thinking", and the first full run was
        # the second of those. These three say which.
        lens = [r.get("raw_len") for r in recs if r.get("raw_len") is not None]
        add("generate", recs,
            {"parse_rate": round(len(parsed) / len(recs), 4) if recs else None,
             "raw_len_mean": round(float(sum(lens)) / len(lens), 1) if lens else None,
             "thinking_open_frac": round(sum(1 for r in recs if r.get("thinking_open")) / len(recs),
                                         4) if recs else None})


# ----------------------------------------------------------------------------- Jev phase
def jev_task(R, suite, key, items, shared_q, shared_labels, qtype, jev, cfg, budget):
    recs = jev.run(items, lambda it: question_of(it, shared_q),
                   lambda it: labels_of(it, shared_labels),
                   workers=cfg.get("jev_workers", 6), budget=budget)
    for rec, it in zip(recs, items):
        if not shared_labels:
            rec["labels"] = it["labels"]
    chance = 1.0 / max(len(shared_labels), 2) if shared_labels else None
    m = bc.score_metrics(recs, shared_labels, qtype, chance)
    R.put(suite, key, "jev/%s" % jev.model, m)
    return recs


def free_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def free_vega(VEGA, R, cache_paths=()):
    """Free the loaded Vega engines, for when nothing after this point reads them.

    The base phase runs last and the summary tables work off the saved metrics, so the two
    engines can go before a second copy of a backbone is loaded. That hands back a whole card and
    takes their weights out of the host accounting, which is the thing the first full run ran out
    of.
    """
    for size, vr in list(VEGA.items()):
        try:
            vr.free()
        except Exception as exc:
            R.note("could not free vega-%s: %s: %s" % (size, type(exc).__name__, str(exc)[:120]))
    free_cuda()
    hinted = bc.drop_file_cache(*[p for p in cache_paths if p])
    R.data["environment"]["memory_after_freeing_vega"] = bc.memory_gb()
    R.save()
    print("freed the Vega engines%s   %s"
          % ("" if not hinted else ", hinted %.1f GB of file cache" % hinted,
             bc.memory_line()), flush=True)


def load_base(repo, device, cfg, R, tag):
    """A second copy of the backbone, loaded only if the host can still hold it.

    Returns None rather than risking the kernel. Host memory, not GPU memory, is what ends a run
    here: a CUDA allocation that does not fit raises an exception this harness catches and
    records, while a host allocation that does not fit has the process killed, and a killed
    kernel takes the remaining phases with it.
    """
    need = float(cfg.get("base_min_free_gb", 6.0))
    head = bc.headroom_gb()
    if head is not None and head < need:
        R.note("did not load the base backbone for %s: %.1f GB of host memory free, %.1f needed"
               % (tag, head, need))
        return None
    print("loading base backbone %s on %s   %s" % (repo, device, bc.memory_line()), flush=True)
    t0 = time.time()
    try:
        base = RO.BaseLM(repo, device, max_len=cfg["base_max_len"], batch=cfg["base_batch"],
                         token=os.environ.get("HF_TOKEN"),
                         thinking=cfg.get("base_thinking", False))
    except Exception as exc:
        R.note("base backbone %s failed to load: %s: %s" % (tag, type(exc).__name__,
                                                            str(exc)[:200]))
        free_cuda()
        return None
    dt = time.time() - t0
    R.data["environment"].setdefault("base_models", {})[tag] = {
        "repo": repo, "device": str(device), "max_len": cfg["base_max_len"],
        "load_seconds": round(dt, 1), "load_mode": base.load_mode,
        "load_failures": getattr(base, "load_failures", []),
        "chat_template": bool(getattr(base.tok, "chat_template", None)),
        "thinking": base.thinking,
        "template_takes_thinking": base.template_takes_thinking,
        "memory_before": base.mem_before, "memory_after": base.mem_after}
    R.save()
    print("loaded in %.1f s via %s   %s" % (dt, base.load_mode, bc.memory_line()), flush=True)
    return base


def free_base(base, cache_paths=()):
    if base is not None:
        base.free()
    free_cuda()
    bc.drop_file_cache(*[p for p in cache_paths if p])
    print("freed the base backbone   %s" % bc.memory_line(), flush=True)


def sliced(items, records, keyfn, qtype="mixed", min_rows=12):
    """Metrics for the whole set and for each slice of it (by primitive, by domain, by tier).

    Slices below `min_rows` are reported with their row count and no metrics, because an accuracy
    over eight rows has a confidence interval wider than any difference it could show.
    """
    by_id = {r["id"]: r for r in records}
    groups = {"all": []}
    for it in items:
        rec = by_id.get(it["id"])
        if rec is None:
            continue
        groups["all"].append(rec)
        name = str(keyfn(it))
        if name != "all":
            groups.setdefault(name, []).append(rec)
    out = {}
    for name, recs in groups.items():
        if len(recs) < min_rows and name != "all":
            out[name] = {"n": len(recs), "too_few_rows": True}
            continue
        out[name] = bc.score_metrics(recs, None, qtype)
    return out


def summarise(data, suite="public", skip=("_meta",)):
    """Macro averages per readout across a suite's tasks: the table to read first.

    Macro, not micro: a 151-class task and a 2-class task count the same, which is the only way a
    mean over tasks of different widths says anything. `skill` is the chance-corrected accuracy, so
    it is the column to compare across tasks rather than raw accuracy.
    """
    import numpy as np

    rows = {}
    for key, readouts in (data.get("results", {}).get(suite) or {}).items():
        for name, m in readouts.items():
            if name in skip or not isinstance(m, dict) or "accuracy" not in m:
                continue
            rows.setdefault(name, []).append((key, m))
    out = {}
    for name, pairs in sorted(rows.items()):
        pick = lambda f: [m[f] for _, m in pairs if m.get(f) is not None]
        mean = lambda f: round(float(np.mean(pick(f))), 4) if pick(f) else None
        out[name] = {"tasks": len(pairs), "accuracy_macro": mean("accuracy"),
                     "skill_macro": mean("skill"), "coverage_macro": mean("coverage"),
                     "ece_macro": mean("ece"), "nll_macro": mean("nll"),
                     "brier_macro": mean("brier"), "aurc_macro": mean("aurc"),
                     "latency_ms_p50_median": round(float(np.median(pick("latency_ms_p50"))), 1)
                     if pick("latency_ms_p50") else None,
                     "macro_f1": mean("macro_f1")}
    return out


def head_to_head(data, suite, a, b):
    """Task by task, which of two readouts won. A mean of means can hide a readout that wins big on
    two tasks and loses everywhere else, and a win count cannot."""
    wins = {"a": 0, "b": 0, "tie": 0, "tasks": [], "a_name": a, "b_name": b}
    for key, readouts in (data.get("results", {}).get(suite) or {}).items():
        ma, mb = readouts.get(a), readouts.get(b)
        if not (isinstance(ma, dict) and isinstance(mb, dict)):
            continue
        if ma.get("accuracy") is None or mb.get("accuracy") is None:
            continue
        d = round(ma["accuracy"] - mb["accuracy"], 4)
        wins["tasks"].append({"task": key, "a": ma["accuracy"], "b": mb["accuracy"], "delta": d})
        wins["a" if d > 0 else ("b" if d < 0 else "tie")] += 1
    wins["tasks"].sort(key=lambda r: r["delta"])
    return wins
