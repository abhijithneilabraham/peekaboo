"""Side-by-side comparison on Vega's own DecisionBench slice.

Test set: exactly the 1,075 rows Vega's README reports (notebook sampling, 25 per task, seed 20261011).
Adaptation pool: the other 3,205 rows of the same 43 tasks (the same sampler at 100 per task, minus the slice).
Of the pool, 20% per task is held out for calibration; every method sees the same rows.

Methods (what each one is allowed to read from the pool):
  zero        nothing                         the released model, as published
  ttt_auto    gold labels                     vegaml's own adaptation: Vega.fit() TTT head per task, used where its
                                              cross-validated accuracy beats chance ("auto"), engine elsewhere
  sup_ts      gold labels                     a fresh gated adapter trained with cross-entropy, + temperature scaling
                                              fitted on the held-out labelled rows (the standard supervised recipe)
  feedback    right/wrong on its own sample   OURS: a fresh gated adapter trained by REINFORCE with a KL anchor to the
                                              released model, + a temperature fitted from fresh feedback on the
                                              held-out rows (labels never read except through the 0/1 reward)
  linear_fb   right/wrong on its own sample   control without Vega's engine: a bilinear state x option scorer over
                                              the same frozen features, trained with the same feedback and calibration

    python compare.py --method feedback --seed 0
"""
import argparse, json, os, time
import numpy as np
import torch

import bench_eval as B
import paths as P
from bench_rl import Policy, predict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = P.COMPARE; os.makedirs(RES, exist_ok=True)
TGRID = np.exp(np.linspace(np.log(0.25), np.log(32), 61))


def load_split(cal_frac=0.2):
    v, items, bank = B.load('decisionbench_items_100.json')
    official = {it['id'] for it in json.load(open(os.path.join(HERE, 'decisionbench_items.json')))}
    test = [it for it in items if it['id'] in official]
    pool = [it for it in items if it['id'] not in official]
    rng = np.random.default_rng(0)
    train, cal = [], []
    for t in sorted({it['task'] for it in pool}):
        rows = [it for it in pool if it['task'] == t]
        rows = [rows[i] for i in rng.permutation(len(rows))]
        k = int(round(cal_frac * len(rows)))
        cal += rows[:k]; train += rows[k:]
    return v, bank, train, cal, test


def gold(it):
    return it['labels'].index(it['gold'])


# ───────────────────────────── metrics, in the README's terms ─────────────────────────────
def metrics(rows, P):
    s = B.score(rows, P)
    chance = np.mean([1 / max(len(it['labels']), 2) for it in rows])
    s['skill'] = round(max(0.0, (s['acc'] - chance) / (1 - chance)), 4)
    s['by_domain'] = {}
    for d in sorted({it['domain'] for it in rows}):
        idx = [i for i, it in enumerate(rows) if it['domain'] == d]
        s['by_domain'][d] = round(float(np.mean([np.argmax(P[i]) == gold(rows[i]) for i in idx])), 4)
    s['rows'] = {'correct': [int(np.argmax(p) == gold(it)) for p, it in zip(P, rows)],
                 'p_gold': [float(p[gold(it)]) for p, it in zip(P, rows)], 'conf': [float(np.max(p)) for p in P]}
    s['ece15'] = round(ece15(s['rows']['conf'], s['rows']['correct']), 4)     # the README's estimator (15 equal bins)
    return s


def ece15(conf, correct, bins=15):
    """ECE exactly as Vega's benchmark notebook computes it: top-label confidence, 15 equal-width bins."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float); e = np.linspace(0, 1, bins + 1); out = 0.0
    for i in range(bins):
        sel = (conf > e[i]) & (conf <= e[i + 1]) if i else (conf >= e[i]) & (conf <= e[i + 1])
        if sel.any():
            out += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(out)


def temp(P, T):
    out = []
    for p in P:
        l = np.log(np.maximum(p, 1e-12)) / T
        e = np.exp(l - l.max()); out.append(e / e.sum())
    return out


def fit_T_labels(P, rows):
    """Standard temperature scaling: minimise NLL of the gold label on held-out labelled rows."""
    g = [gold(it) for it in rows]
    return float(min(TGRID, key=lambda T: -np.mean([np.log(max(p[gi], 1e-12)) for p, gi in zip(temp(P, T), g)])))


def fit_T_feedback(P, rows, rng):
    """Feedback-only calibration: sample a ~ p, observe r = [a correct]; maximise sum log Bernoulli(r; p_T(a))."""
    logs = []
    for p, it in zip(P, rows):
        a = int(rng.choice(len(p), p=p / p.sum())); logs.append((p, a, float(a == gold(it))))
    def ll(T):
        tot = 0.0
        for p, a, r in logs:
            q = min(max(temp([p], T)[0][a], 1e-6), 1 - 1e-6); tot += r * np.log(q) + (1 - r) * np.log(1 - q)
        return tot
    return float(max(TGRID, key=ll))


# ───────────────────────────── methods ─────────────────────────────
def engine_probs(v, bank, rows):
    return [np.asarray(p, float) for p in B.run(v, rows, bank)]


def ttt_auto(v, bank, train, cal, test):
    """Vega.fit() semantics per task: TTTHead on pooled features, kept if 5-fold CV accuracy > chance; 'auto' uses
    the head for questions whose options it knows and the engine otherwise (generous to the head: a test row with an
    option the head never saw goes to the engine instead of getting probability 0 for that option)."""
    from vegaml.ttt import TTTHead
    eng = engine_probs(v, bank, test)
    P, used, report, by_head = [], 0, {}, []
    heads = {}
    for t in sorted({it['task'] for it in test}):
        rows = [it for it in train + cal if it['task'] == t]          # fit() gets every labelled row it can use
        X = [bank[it['id']]['feats'].numpy().ravel() for it in rows]; y = [it['gold'] for it in rows]
        labels = sorted({l for it in rows for l in it['labels']})
        if len(set(y)) < 2 or len(X) < 6:
            report[t] = 'not fitted'; continue
        h = TTTHead(labels).fit(X, y)
        cv = h.cross_val_accuracy(X, y, folds=5)
        chance = 1.0 / max(len(labels), 2)
        report[t] = {'cv': cv, 'n_labels': len(labels)}
        if cv is not None and cv > chance:
            heads[t] = h
    for it, pe in zip(test, eng):
        h = heads.get(it['task'])
        if h is not None and set(it['labels']) <= set(h.labels):
            q = h.predict(bank[it['id']]['feats'].numpy().ravel(), allowed=it['labels'])
            P.append(np.array([q[l] for l in it['labels']])); used += 1; by_head.append(1)
        else:
            P.append(pe); by_head.append(0)
    return P, {'heads_kept': len(heads), 'rows_answered_by_head': used, 'by_head': by_head, 'tasks': report}


def train_adapter(v, bank, train, method, rng, epochs=8, lr=1e-3, kl=0.1, batch=32):
    pol = Policy(v, lr, rank=8)
    base, curve = 0.5, []
    for ep in range(epochs):
        order = rng.permutation(len(train)); rew = []
        for s in range(0, len(train), batch):
            rows = [train[i] for i in order[s:s + batch]]
            g = torch.tensor([gold(it) for it in rows])
            lp, mask = pol.logp(bank, rows)
            if method == 'sup':
                loss = -lp[torch.arange(len(rows)), g].mean()
            else:
                with torch.no_grad():
                    a = torch.multinomial(lp.exp(), 1).squeeze(1); r = (a == g).float()
                loss = -((r - base) * lp[torch.arange(len(rows)), a]).mean()
                base = 0.9 * base + 0.1 * float(r.mean()); rew.append(float(r.mean()))
            if kl:
                with torch.no_grad():
                    lq, _ = pol.logp(bank, rows, adapter=False)
                loss = loss + kl * (lp.exp() * (lp - lq)).masked_fill(~mask, 0).sum(-1).mean()
            pol.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.params, 1.0); pol.opt.step()
        curve.append(float(np.mean(rew)) if rew else None)
    return pol, curve


class Bilinear(torch.nn.Module):
    """No-engine control: logit_k = <A s, C o_k> + w . C o_k over standardised frozen features."""
    def __init__(self, ds, do, h=64):
        super().__init__()
        self.A, self.C, self.w = torch.nn.Linear(ds, h), torch.nn.Linear(do, h), torch.nn.Linear(h, 1)

    def forward(self, s, o, mask):
        lg = (self.A(s)[:, None, :] * self.C(o)).sum(-1) + self.w(self.C(o)).squeeze(-1)
        return torch.log_softmax(lg.masked_fill(~mask, -1e4), -1)


def linear_fb(bank, train, cal, test, rng, epochs=8, lr=1e-3, ent=0.01, batch=32):
    S = torch.stack([bank[it['id']]['feats'].flatten() for it in train]); mu_s, sd_s = S.mean(0), S.std(0) + 1e-6
    O = torch.cat([bank[it['id']]['opts'].flatten(1) for it in train]); mu_o, sd_o = O.mean(0), O.std(0) + 1e-6
    def batch_of(rows):
        s = (torch.stack([bank[it['id']]['feats'].flatten() for it in rows]) - mu_s) / sd_s
        K = max(len(it['labels']) for it in rows)
        o = torch.zeros(len(rows), K, O.shape[1]); m = torch.zeros(len(rows), K, dtype=torch.bool)
        for b, it in enumerate(rows):
            x = (bank[it['id']]['opts'].flatten(1) - mu_o) / sd_o; o[b, :len(x)] = x; m[b, :len(x)] = True
        return s, o, m
    net = Bilinear(S.shape[1], O.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    base = 0.5
    for ep in range(epochs):
        order = rng.permutation(len(train))
        for s0 in range(0, len(train), batch):
            rows = [train[i] for i in order[s0:s0 + batch]]
            lp = net(*batch_of(rows)); g = torch.tensor([gold(it) for it in rows])
            with torch.no_grad():
                a = torch.multinomial(lp.exp(), 1).squeeze(1); r = (a == g).float()
            loss = -((r - base) * lp[torch.arange(len(rows)), a]).mean() + ent * (lp.exp() * lp).sum(-1).mean()
            base = 0.9 * base + 0.1 * float(r.mean())
            opt.zero_grad(); loss.backward(); opt.step()
    def probs(rows):
        with torch.no_grad():
            out = []
            for s0 in range(0, len(rows), 64):
                rr = rows[s0:s0 + 64]; lp = net(*batch_of(rr))
                out += [lp[b, :len(it['labels'])].exp().numpy().astype(float) for b, it in enumerate(rr)]
            return out
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--kl', type=float, default=0.1); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    name = f'{a.method}_s{a.seed}{a.tag}'
    out = os.path.join(RES, name + '.json')
    if os.path.exists(out):
        print('skip', name); return
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    v, bank, train, cal, test = load_split()
    t0, info = time.time(), {}
    if a.method == 'zero':
        P = engine_probs(v, bank, test)
    elif a.method == 'ttt_auto':
        P, info = ttt_auto(v, bank, train, cal, test)
    elif a.method in ('sup_ts', 'feedback'):
        pol, curve = train_adapter(v, bank, train, 'sup' if a.method == 'sup_ts' else 'bandit', rng, kl=a.kl)
        Pc, Pt = [np.asarray(p, float) for p in predict(pol, bank, cal)], [np.asarray(p, float) for p in predict(pol, bank, test)]
        T = fit_T_labels(Pc, cal) if a.method == 'sup_ts' else fit_T_feedback(Pc, cal, rng)
        info = {'T': T, 'reward_curve': curve, 'uncalibrated': {k: v_ for k, v_ in metrics(test, Pt).items() if k != 'rows'}}
        P = temp(Pt, T)
    elif a.method == 'linear_fb':
        probs = linear_fb(bank, train, cal, test, rng)
        Pc, Pt = probs(cal), probs(test)
        T = fit_T_feedback(Pc, cal, rng)
        info = {'T': T, 'uncalibrated': {k: v_ for k, v_ in metrics(test, Pt).items() if k != 'rows'}}
        P = temp(Pt, T)
    else:
        raise SystemExit('unknown method')
    m = metrics(test, P)
    print(name, json.dumps({k: v_ for k, v_ in m.items() if k not in ('rows', 'by_domain')}), f'{time.time() - t0:.0f}s', flush=True)
    json.dump({'args': vars(a), 'n_train': len(train), 'n_cal': len(cal), 'n_test': len(test), 'test': m, 'info': info},
              open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
