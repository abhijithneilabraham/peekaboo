# Vendored verbatim from NandhaKishorM/vegaml@00d3d99, notebooks/vega_benchmark_2xT4.ipynb (Apache-2.0).
# Do not edit: this is VegaML's own benchmark harness, used so every number is loaded and scored as theirs is.
"""The readouts under test, from the frozen backbone alone up to a fitted head.

The ladder, all on the same frozen weights and the same rows:

| readout              | what answers                                     | needs labels |
|----------------------|--------------------------------------------------|--------------|
| `base_generate`      | the backbone, asked in words, greedy decode      | no           |
| `base_logprob`       | the backbone, scoring each option as text        | no           |
| `engine_no_adapter`  | the physics engine, adapters off                 | no           |
| `engine`             | the engine with its gated adapters               | no           |
| `engine_views2/3`    | the engine, averaged over surface forms          | no           |
| `ttt_k`              | a linear head on the same pooled features        | k examples   |
| `auto`               | head where it beats chance, engine elsewhere     | k examples   |

The backbone is a base checkpoint, not an instruction tune, so `base_generate` is not a straw man
of a chat model: it is the same weights the engine reads, asked the only other way they can be
asked. Its coverage number (how often the output can be parsed into an option at all) is as
interesting as its accuracy.
"""
import json
import re
import time

import numpy as np

import bench_common as bc


# ----------------------------------------------------------------------------- shared
def surface_options(q):
    """Short natural wordings of the options, for the readouts that have to put them in a prompt.

    Vega's own prompt uses `option_texts`, which carries the criterion as well. A base model asked
    to pick among "false - no, the statement does not hold" style strings is being handicapped by
    the harness rather than measured, so the baselines get the plain wording.
    """
    t, crit = q["type"], q.get("criteria")
    if t == "boolean":
        return ["no", "yes"]
    if t == "score":
        return [str(c) for c in (crit or [])]
    return [str(v) if v else str(k) for k, v in (crit or {}).items()]


def _state_text(state):
    return bc.serialize_state(state) if not isinstance(state, str) else state


def normalise_answer(a, qtype, labels):
    """One Vega answer in the harness's own vocabulary: probabilities keyed by option id.

    Needed because the engine reports a boolean as `p(true)` with `probs` keyed "False"/"True",
    a score with integer keys, and a choice with the criteria's own keys. Scoring code that reads
    those three shapes inline is where silent off-by-one label bugs live.
    """
    if qtype == "boolean":
        p = float(a.get("boolean", 0.0))
        probs = {"false": 1.0 - p, "true": p}
    else:
        probs = {str(k): float(v) for k, v in (a.get("probs") or {}).items()}
    probs = {k: probs.get(k, 0.0) for k in labels}
    pred = max(probs, key=probs.get) if probs else None
    pset = a.get("prediction_set")
    ids = None
    if pset is not None:
        ids = [str(x).lower() if qtype == "boolean" else str(x) for x in pset]
    return probs, pred, ids


# ----------------------------------------------------------------------------- Vega
class VegaRunner:
    """One loaded Vega: the engine readouts, the pooled features, and the fitted heads."""

    def __init__(self, size, device, adapters="auto", max_len=None, token=None):
        import vegaml

        self.size = size
        self.device = device
        self.vega = vegaml.load(size, device=device, adapters=adapters, mode="engine",
                                max_len=max_len, token=token)
        self.tok = self.vega.engine.perc.tok
        self.cfg = self.vega.engine.cfg
        self.adapter_names = [a["name"] for a in (self.vega.engine.adapters or [])]

    def set_adapters(self, mode):
        self.vega.engine.set_adapters(mode)

    def free(self):
        """Drop the weights, keeping the metadata later cells read.

        `cfg`, `tok` and `adapter_names` stay, because the base phase asks this object for the
        backbone's name after the engine itself is gone. The cycle collector has to run before
        emptying the CUDA cache or nothing is actually returned.
        """
        import gc

        self.vega = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def engine_record(self, item, question, labels, views=1, alpha="0.1"):
        t0 = time.perf_counter()
        out = self.vega.decide(item["state"], {"q": question}, mode="engine", alpha=alpha,
                               views=views)
        dt = (time.perf_counter() - t0) * 1000
        a = (out.get("answers") or {}).get("q") or {}
        probs, pred, ids = normalise_answer(a, question["type"], labels)
        rec = {"id": item["id"], "gold": item["gold"], "pred": pred, "probs": probs,
               "latency_ms": dt, "abstain": bool(a.get("abstain")),
               "unbound": bool(a.get("unbound")), "n_tokens": out.get("n_tokens"),
               "settle_steps": a.get("settle_steps"), "gate": a.get("gate"),
               "adapter": a.get("adapter")}
        if ids is not None:
            rec["set_size"] = len(ids)
            rec["set_hit"] = item["gold"] in ids
        return rec

    def feature(self, item, question):
        return self.vega.features(item["state"], question)

    def features_for(self, items, question=None):
        """Pooled features for a list of items, one read each. `question=None` means per-row."""
        return [self.feature(it, question or it["question"]) for it in items]


def fit_head_ladder(X, y, labels, shots, folds=5):
    """One head per budget in `shots`, each fitted on the first k examples of the same feature set.

    Fitting on prefixes costs nothing extra: the features are computed once. What it buys is a
    learning curve, which is the only honest way to answer "is a head worth it", since the answer
    is a function of how many labels you have.
    """
    from vegaml import TTTHead

    out = {}
    for k in shots:
        if k > len(X):
            continue
        Xk, yk = X[:k], y[:k]
        if len(set(yk)) < 2:
            out[k] = {"fitted": False, "reason": "one class only in the first %d examples" % k}
            continue
        head = TTTHead(labels)
        classes = len(set(yk))
        # Folds are stratified, so a class needs one example per fold to survive the split. Five
        # folds over fifty examples of six classes is fine; five folds over six examples is not,
        # and asking for them returns None, which "auto" then reads as an unmeasured head and
        # leaves to the engine. Scaling the fold count to what the data can support is the
        # difference between a measured head and an unknown one at the small budgets.
        folds_k = max(2, min(folds, len(yk) // max(classes, 1)))
        try:
            head.fit(Xk, yk)
            cv = head.cross_val_accuracy(Xk, yk, folds=folds_k)
        except Exception as exc:
            out[k] = {"fitted": False, "reason": "%s: %s" % (type(exc).__name__, str(exc)[:160])}
            continue
        chance = 1.0 / max(len(labels), 2)
        out[k] = {"fitted": True, "head": head, "cv_accuracy": cv, "chance": round(chance, 4),
                  "cv_n": getattr(head, "cross_val_n", 0), "classes_seen": classes,
                  "folds": folds_k, "examples": len(Xk),
                  "beats_chance": None if cv is None else bool(cv > chance)}
        if cv is None:
            out[k]["cv_note"] = ("%d examples over %d classes is too few to cross-validate, so "
                                 "this head's accuracy is unknown rather than good or bad, and "
                                 "auto leaves the question to the engine" % (len(yk), classes))
    return out


def head_record(head, feat, item, labels, allowed=None):
    t0 = time.perf_counter()
    p = head.predict(feat, allowed=allowed or labels)
    dt = (time.perf_counter() - t0) * 1000
    tot = sum(p.values()) or 1.0
    probs = {k: v / tot for k, v in p.items()}
    return {"id": item["id"], "gold": item["gold"], "pred": max(probs, key=probs.get),
            "probs": probs, "latency_ms": dt}


def auto_record(engine_rec, head_rec, head_info):
    """What mode="auto" would answer: the head when its cross-validated accuracy beat chance on
    this task, the engine otherwise. Reported as its own column so the routing rule is visible."""
    use_head = bool(head_info and head_info.get("beats_chance"))
    rec = dict(head_rec if use_head else engine_rec)
    rec["readout"] = "ttt" if use_head else "engine"
    return rec


# ----------------------------------------------------------------------------- base backbone
class BaseLM:
    """The frozen backbone on its own: asked in words, or scoring each option as a continuation.

    Loaded as a second copy of the weights because Vega's perception registers forward hooks that
    stop the forward pass at the deepest layer it reads. Those hooks make the shared instance
    unusable for generation or for logits, which is why this class does not borrow it.
    """

    def __init__(self, repo, device, dtype=None, max_len=3072, batch=16, token=None,
                 thinking=False):
        import gc

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(repo, token=token)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.mem_before = bc.memory_gb()
        # The 4B is about 8 GB of fp16 weights, and `from_pretrained(...).to(device)` materialises
        # all of it in host memory first. A second copy of that, on top of two resident Vega
        # instances, is enough to have the kernel killed outright, which arrives as a dead kernel
        # with no traceback. device_map streams the shards straight to the card instead, so the
        # host only ever holds one shard. The plain path stays as a fallback for an image without
        # accelerate, and which path ran is recorded.
        self.model, self.load_mode, failures = None, None, []
        for mode in ("device_map", "plain"):
            kw = dict(dtype=dtype, attn_implementation="sdpa", token=token)
            if mode == "device_map":
                kw["device_map"] = {"": str(self.device)}
            try:
                self.model = AutoModelForCausalLM.from_pretrained(repo, **kw)
                if mode == "plain":
                    self.model.to(self.device)
                self.load_mode = mode
                break
            except Exception as exc:
                failures.append("%s: %s: %s" % (mode, type(exc).__name__, str(exc)[:160]))
                self.model = None
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        if self.model is None:
            raise RuntimeError("could not load %s; tried %s" % (repo, " | ".join(failures)))
        self.load_failures = failures
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mem_after = bc.memory_gb()
        self.max_len = max_len
        self.batch = batch
        self.repo = repo
        self.thinking = thinking
        self.template_takes_thinking = "enable_thinking" in str(
            getattr(self.tok, "chat_template", "") or "")

    # ---------------------------------------------------------------- prompting
    def _fit_state(self, state, reserve):
        """The state, cut to whatever the prompt budget leaves. Cut rather than refused, because a
        baseline that declines to read long rows would look better than it is; the cut is recorded."""
        text = _state_text(state)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        room = max(64, self.max_len - reserve)
        if len(ids) <= room:
            return text, False
        return self.tok.decode(ids[:room]), True

    def _prompt(self, state, question, letters):
        """The prompt, whether the state had to be cut, and whether a reasoning block is left open.

        Qwen3.5's chat template opens a `<think>` block, and the two sizes of the same family
        default the other way round: with `enable_thinking` unset the 0.8B template closes the
        block inside the generation prompt and the 4B leaves it open. Left open, a short generation
        budget is spent reasoning and the baseline scores near zero for a formatting reason rather
        than a modelling one, so thinking is switched off explicitly and the same prompt shape
        reaches both sizes. The flag comes back too, so a run with thinking on stays readable
        instead of quietly wrong.
        """
        opts = surface_options(question)
        lines = ["%s) %s" % (l, o) for l, o in zip(letters, opts)]
        head = ("You are a decision model. Read the record and answer the question with exactly one "
                "option id.\n\n")
        tail = ("\n\nQuestion: %s\nOptions:\n%s\n\nAnswer with one id from %s and nothing else.\n"
                "Answer:" % (question["instructions"], "\n".join(lines),
                             ", ".join(letters)))
        reserve = len(self.tok(head + tail, add_special_tokens=False)["input_ids"]) + 16
        body, cut = self._fit_state(state, reserve)
        text = head + "Record:\n" + body + tail
        if getattr(self.tok, "chat_template", None):
            kw = {"tokenize": False, "add_generation_prompt": True}
            if self.template_takes_thinking:
                kw["enable_thinking"] = bool(self.thinking)
            text = self.tok.apply_chat_template([{"role": "user", "content": text}], **kw)
        think_open = text.count("<think>") > text.count("</think>")
        return text, cut, think_open

    # ---------------------------------------------------------------- readouts
    def generate_record(self, item, question, labels, max_new_tokens=12):
        letters = [chr(ord("A") + i) for i in range(len(labels))]
        text, cut, think_open = self._prompt(item["state"], question, letters)
        ids = self.tok(text, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        dt = (time.perf_counter() - t0) * 1000
        gen = self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        answer = gen
        if think_open:
            # the answer is whatever follows the close tag. Reasoning that never closed is not an
            # answer, and reading an option out of the reasoning would invent a result: 17 of the
            # 43 tasks in the first full run scored above zero that way, from text that was the
            # model thinking aloud rather than answering.
            answer = gen.split("</think>", 1)[1] if "</think>" in gen else ""
        pred = parse_generation(answer, question, labels, letters)
        return {"id": item["id"], "gold": item["gold"], "pred": pred, "probs": None,
                "latency_ms": dt, "raw_len": len(gen), "answer_len": len(answer),
                "thinking_open": think_open, "state_truncated": cut, "parsed": pred is not None}

    def logprob_record(self, item, question, labels):
        """Each option scored as the continuation of the same prompt.

        Two numbers come out of one batch: the mean log probability per option token (what the
        `base_logprob` column reports, because it does not reward short options) and the summed log
        probability (`base_logprob_sum`), which does. Reporting both keeps the normalisation choice
        from quietly deciding the result.
        """
        torch = self.torch
        letters = [chr(ord("A") + i) for i in range(len(labels))]
        prompt, cut, _ = self._prompt(item["state"], question, letters)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        opts = surface_options(question)
        conts = [self.tok(" " + o, add_special_tokens=False)["input_ids"][:48] or
                 self.tok(" ?", add_special_tokens=False)["input_ids"] for o in opts]
        sums, means = [], []
        t0 = time.perf_counter()
        for start in range(0, len(conts), self.batch):
            chunk = conts[start:start + self.batch]
            seqs = [p_ids + c for c in chunk]
            width = max(len(s) for s in seqs)
            pad = self.tok.pad_token_id or 0
            inp = torch.full((len(seqs), width), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), width), dtype=torch.long)
            for i, s in enumerate(seqs):                 # left padding keeps the tail aligned
                inp[i, width - len(s):] = torch.tensor(s)
                att[i, width - len(s):] = 1
            inp, att = inp.to(self.device), att.to(self.device)
            with torch.inference_mode():
                logits = self.model(input_ids=inp, attention_mask=att).logits.float()
            logp = torch.log_softmax(logits, dim=-1)
            for i, c in enumerate(chunk):
                n = len(c)
                # the token at position t is predicted by the logits at t-1
                pos = torch.arange(width - n - 1, width - 1, device=self.device)
                tgt = torch.tensor(c, device=self.device)
                lp = logp[i, pos, tgt]
                sums.append(float(lp.sum()))
                means.append(float(lp.mean()))
        dt = (time.perf_counter() - t0) * 1000

        def to_probs(scores):
            z = np.array(scores, dtype=np.float64)
            z = z - z.max()
            e = np.exp(z)
            return {lab: float(v) for lab, v in zip(labels, e / e.sum())}

        pm, ps = to_probs(means), to_probs(sums)
        return ({"id": item["id"], "gold": item["gold"], "pred": max(pm, key=pm.get), "probs": pm,
                 "latency_ms": dt, "state_truncated": cut},
                {"id": item["id"], "gold": item["gold"], "pred": max(ps, key=ps.get), "probs": ps,
                 "latency_ms": dt, "state_truncated": cut})

    def free(self):
        """Drop the weights and hand the memory back, host side as well as device side.

        `del self.model` alone is not enough: transformers modules sit in reference cycles, so the
        memory is only reclaimed once the cycle collector has run, and emptying the CUDA cache
        before that frees nothing. Calling this twice is safe.
        """
        import gc

        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


_YES = re.compile(r"\b(yes|true|y)\b", re.I)
_NO = re.compile(r"\b(no|false|n)\b", re.I)


def parse_generation(text, question, labels, letters):
    """The option a generated answer names, or None when it names none.

    None is not the same as wrong: it lowers `coverage`, and that is the number that says whether a
    base model asked in words can be put in a pipeline at all.
    """
    t = (text or "").strip()
    if not t:
        return None
    if question["type"] == "boolean":
        head = t[:40]
        y, n = _YES.search(head), _NO.search(head)
        if y and (not n or y.start() < n.start()):
            return "true"
        if n:
            return "false"
    m = re.match(r"\s*\(?([A-Z])\)?[\s.:,)]", t + " ")
    if m and m.group(1) in letters:
        return labels[letters.index(m.group(1))]
    for lab, letter, surf in zip(labels, letters, surface_options(question)):
        if re.search(r"\b%s\b" % re.escape(str(lab)), t, re.I):
            return lab
        if surf and re.search(r"\b%s\b" % re.escape(str(surf)[:40]), t, re.I):
            return lab
    if question["type"] == "score":
        d = re.search(r"\b(\d+)\b", t)
        if d and d.group(1) in labels:
            return d.group(1)
    return None
