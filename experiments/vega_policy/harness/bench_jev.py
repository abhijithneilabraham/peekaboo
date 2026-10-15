# Vendored verbatim from NandhaKishorM/vegaml@00d3d99, notebooks/vega_benchmark_2xT4.ipynb (Apache-2.0).
# Do not edit: this is VegaML's own benchmark harness, used so every number is loaded and scored as theirs is.
"""The hosted reference: Jev's typed-decision endpoint, called with the same questions.

Jev takes the same request shape Vega answers locally, so the comparison needs no re-framing: one
state, a map of typed questions, a probability per declared option. Two differences are handled
here and nowhere else, so the rest of the harness stays uniform:

* Jev calls a yes/no question `noul`; Vega calls it `boolean`.
* Jev answers over the network, so a row can fail. A failed row is recorded as unanswered rather
  than retried into a different number, and it lowers `coverage` like any other refusal.

Responses are cached on disk by request hash. A re-run of the notebook therefore costs nothing and
returns the same answers, which also means the Jev column is reproducible within a session.
"""
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def to_jev_questions(questions):
    """Vega's question map in Jev's vocabulary."""
    out = {}
    for qid, q in questions.items():
        jq = {"instructions": q["instructions"]}
        if q["type"] == "boolean":
            jq["type"] = "noul"
            jq["criteria"] = q.get("criteria") or {"false": "no", "true": "yes"}
        elif q["type"] == "score":
            jq["type"] = "score"
            jq["criteria"] = list(q.get("criteria") or [])
        else:
            jq["type"] = "choice"
            jq["criteria"] = {k: (v or k) for k, v in q["criteria"].items()}
        out[qid] = jq
    return out


def parse_answer(entry, qtype, labels):
    """One Jev answer as (probs, pred). Returns (None, None) when the shape is not understood,
    which is counted rather than guessed at."""
    if not isinstance(entry, dict):
        return None, None
    probs = None
    for key in ("probabilities", "probs", "distribution"):
        if isinstance(entry.get(key), dict):
            probs = {str(k): float(v) for k, v in entry[key].items()}
            break
    if probs is None and qtype == "boolean":
        for key in ("noul", "boolean", "probability", "p_true"):
            if isinstance(entry.get(key), (int, float)):
                p = float(entry[key])
                probs = {"false": 1.0 - p, "true": p}
                break
    if probs is None and isinstance(entry.get("scores"), dict):
        probs = {str(k): float(v) for k, v in entry["scores"].items()}
    if probs is not None:
        low = {k.lower(): v for k, v in probs.items()}
        picked = {}
        for lab in labels:
            if lab in probs:
                picked[lab] = probs[lab]
            elif str(lab).lower() in low:
                picked[lab] = low[str(lab).lower()]
        if picked and sum(picked.values()) > 0:
            tot = sum(picked.values())
            picked = {k: v / tot for k, v in picked.items()}
            return picked, max(picked, key=picked.get)
    # label-only fallback: no distribution, so calibration is not measurable for this row
    for key in ("choice", "score", "label", "answer", "noul"):
        if key in entry:
            val = entry[key]
            if qtype == "boolean" and isinstance(val, (int, float)):
                return None, ("true" if float(val) >= 0.5 else "false")
            sval = str(val)
            if sval in labels:
                return None, sval
            low = {str(l).lower(): l for l in labels}
            if sval.lower() in low:
                return None, low[sval.lower()]
    return None, None


class JevClient:
    def __init__(self, api_key, model="jev-latest", endpoint=ENDPOINT, cache_path=None,
                 timeout=60, retries=4, max_rps=None):
        if not (api_key or "").strip():
            raise ValueError("a Jev API key is required; the notebook reads it from the "
                             "JEV_API Kaggle secret")
        self._key = api_key.strip()
        self.model, self.endpoint, self.timeout, self.retries = model, endpoint, timeout, retries
        self.cache_path = cache_path
        self.cache = {}
        self.lock = threading.Lock()
        self.stats = {"calls": 0, "cache_hits": 0, "http_errors": {}, "usage": {},
                      "decisions": 0, "latency_ms": []}
        self._min_gap = (1.0 / max_rps) if max_rps else 0.0
        self._last = 0.0
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        self.cache[rec["k"]] = rec["v"]
                    except Exception:
                        continue

    # ---------------------------------------------------------------- transport
    def _body(self, state, questions):
        return {"model": self.model, "state": state, "questions": to_jev_questions(questions)}

    def _key_for(self, body):
        blob = json.dumps(body, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _remember(self, key, value):
        with self.lock:
            self.cache[key] = value
            if self.cache_path:
                with open(self.cache_path, "a") as fh:
                    fh.write(json.dumps({"k": key, "v": value}) + "\n")

    def _throttle(self):
        if not self._min_gap:
            return
        with self.lock:
            wait = self._last + self._min_gap - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

    def raw(self, state, questions):
        """The decoded response, from cache when it is there. -> (payload, latency_ms, error)."""
        body = self._body(state, questions)
        key = self._key_for(body)
        if key in self.cache:
            with self.lock:
                self.stats["cache_hits"] += 1
            return self.cache[key], 0.0, None
        data = json.dumps(body).encode()
        last = "no attempt made"
        for attempt in range(self.retries + 1):
            self._throttle()
            req = urllib.request.Request(
                self.endpoint, data=data, method="POST",
                headers={"Authorization": "Bearer " + self._key,
                         "Content-Type": "application/json",
                         "User-Agent": "vega-bench/1.0"})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read())
                dt = (time.perf_counter() - t0) * 1000
                with self.lock:
                    self.stats["calls"] += 1
                    self.stats["latency_ms"].append(dt)
                    for k, v in (payload.get("usage") or {}).items():
                        if isinstance(v, (int, float)):
                            self.stats["usage"][k] = self.stats["usage"].get(k, 0) + v
                self._remember(key, payload)
                return payload, dt, None
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "ignore")[:200]
                last = "HTTP %d: %s" % (exc.code, detail)
                with self.lock:
                    self.stats["http_errors"][str(exc.code)] = \
                        self.stats["http_errors"].get(str(exc.code), 0) + 1
                if exc.code in (400, 401, 403, 404, 422):     # a retry cannot fix the request
                    return None, (time.perf_counter() - t0) * 1000, last
            except Exception as exc:
                last = "%s: %s" % (type(exc).__name__, str(exc)[:160])
            time.sleep(min(20.0, (2 ** attempt) + random.random()))
        return None, 0.0, last

    # ---------------------------------------------------------------- one row
    def record(self, item, question, labels):
        payload, dt, err = self.raw(item["state"], {"q": question})
        rec = {"id": item["id"], "gold": item["gold"], "pred": None, "probs": None,
               "latency_ms": dt or None}
        if err or not isinstance(payload, dict):
            rec["error"] = err or "unparsable response"
            return rec
        answers = payload.get("answers") or {}
        entry = answers.get("q") or (list(answers.values())[0] if answers else None)
        probs, pred = parse_answer(entry, question["type"], labels)
        rec["probs"], rec["pred"] = probs, pred
        if pred is None:
            rec["error"] = "answer shape not understood: %s" % json.dumps(entry)[:160]
        with self.lock:
            self.stats["decisions"] += 1
        rec["model_reported"] = payload.get("model")
        return rec

    def run(self, items, question_for, labels_for, workers=6, budget=None, progress=None):
        """Every item, in a small thread pool. `question_for(item)` and `labels_for(item)` keep
        per-row option sets working. Stops early and returns what it has if `budget` runs out."""
        from concurrent.futures import ThreadPoolExecutor

        out = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for it in items:
                futures[pool.submit(self.record, it, question_for(it), labels_for(it))] = it
            for i, fut in enumerate(futures):
                if budget is not None and not budget.ok(5):
                    break
                try:
                    out.append(fut.result())
                except Exception as exc:
                    out.append({"id": futures[fut]["id"], "gold": futures[fut]["gold"],
                                "pred": None, "probs": None,
                                "error": "%s: %s" % (type(exc).__name__, str(exc)[:120])})
                if progress and i % 50 == 0:
                    progress(i + 1, len(items))
        return out

    def summary(self):
        import numpy as np
        lat = self.stats["latency_ms"]
        return {"model": self.model, "endpoint": self.endpoint, "calls": self.stats["calls"],
                "cache_hits": self.stats["cache_hits"], "decisions": self.stats["decisions"],
                "http_errors": dict(self.stats["http_errors"]),
                "usage_totals": dict(self.stats["usage"]),
                "latency_ms_p50": round(float(np.percentile(lat, 50)), 1) if lat else None,
                "latency_ms_p95": round(float(np.percentile(lat, 95)), 1) if lat else None}
