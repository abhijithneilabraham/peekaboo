"""Stage 2: one method, one suite, one seed, on the features suite_cache.py captured from Vega.decide().

Methods, and what each may read from the suite's pool rows:
  released      nothing           the released model. No run: its records are decide()'s own, from suite_cache.py
  ttt           gold labels       VegaML's own adaptation, the TTT head on the same frozen features, "auto" routing
                                  (head where its cross-validated accuracy beats chance, engine elsewhere). On the
                                  public menu this is exactly the notebook's head ladder on its 50 fit rows per task.
  sup_ts        gold labels       a fresh gated adapter trained with cross-entropy + temperature scaling on held-out
                                  labelled rows (the supervised recipe of vegaml's own fine-tuning script)
  feedback      right/wrong only  OURS: the same adapter trained by REINFORCE on the reward of its own sampled answer,
                                  with a KL anchor (beta 0.1) to the released model, + one temperature fitted from
                                  fresh feedback on held-out rows. Gold labels are read only through the 0/1 reward.
  feedback_kl1  right/wrong only  ablation: KL weight 1
  linear_fb     right/wrong only  control without Vega's engine: a bilinear scorer on the same features, same feedback
  feedback_noise10/20/30, feedback_half/quarter, feedback_complaints
                right/wrong only  Peekaboo under realistic feedback: 10-30% of reports flipped; only half or a quarter of
                                  answers rated; complaints only (every wrong answer reported, 20% of right ones)
  logged_ips    right/wrong only  logged-feedback baseline: feedback on the released model's own answers, logged once,
                                  learned off-policy with clipped importance weights

Where the feedback comes from:
  public         the notebook's 50 "fit" rows per task (training split, never scored), the rows its TTT head gets
  decisionbench  other rows of the same 43 tasks (100 per task minus the README slice)
  jevbench       it has no training rows, so 5-fold cross-fitting: each fold is predicted by an adapter trained on
                 feedback from the other four, with paraphrase twins always in the same fold
In every pool, 20% of each task's rows are held out for the temperature; nothing is tuned on test rows.

    python adapt.py --suite public --method feedback --seed 0
Output: results*/methods/<suite>/<method>_s<seed>.json, one record per test row in the notebook's record format.
"""
import argparse, json, os, time
import numpy as np
import torch

import bench_eval as B
import paths
from bench_rl import Policy, predict

paths.harness()
TGRID = np.exp(np.linspace(np.log(0.25), np.log(32), 61))
# Feedback as a deployment would deliver it: (flip probability, P(reported | right), P(reported | wrong)).
FEEDBACK = {'feedback': (0.0, 1.0, 1.0), 'feedback_kl1': (0.0, 1.0, 1.0),
            'feedback_noise10': (0.1, 1.0, 1.0), 'feedback_noise20': (0.2, 1.0, 1.0), 'feedback_noise30': (0.3, 1.0, 1.0),
            'feedback_half': (0.0, 0.5, 0.5), 'feedback_quarter': (0.0, 0.25, 0.25),
            'feedback_complaints': (0.0, 0.2, 1.0)}
# Realistic-feedback tests of Peekaboo: noisy, missing, complaint-only feedback, and the logged-feedback baseline
# (off-policy: one answer per row sampled from the released model, feedback logged once, trained by clipped IPS).
REALISTIC = [k for k in FEEDBACK if k not in ('feedback', 'feedback_kl1')] + ['logged_ips']
FEEDBACK['feedback_kl0'] = (0.0, 1.0, 1.0)          # ablation: no leash
KL = {'feedback_kl1': 1.0, 'feedback_kl0': 0.0}      # every other feedback method uses 0.1
# v2 grid (after the review): one pass over the feedback rows (each row rated once, as a deployment would), logged
# argmax feedback (the released model's own answers rated once, learned off-policy), and typed-decisions with Vega's
# shipped adapters switched off ("_noship": the released typed_decisions adapter was fitted on that pool).
EPOCHS = {'feedback_1pass': 1, 'sup_1pass': 1, 'feedback_1pass_noship': 1}
FEEDBACK['feedback_1pass'] = FEEDBACK['feedback_noship'] = FEEDBACK['feedback_1pass_noship'] = (0.0, 1.0, 1.0)
REALISTIC = REALISTIC + ['logged_argmax']
ADAPTER_METHODS = ('sup_ts', 'feedback', 'feedback_kl1', 'feedback_kl0', 'feedback_1pass', 'sup_1pass',
                   'feedback_noship', 'sup_ts_noship', 'feedback_1pass_noship') + tuple(REALISTIC)
NOSHIP = lambda method: method.endswith('_noship')
KIND = lambda method: ('sup' if method.startswith('sup') else 'ips' if method == 'logged_ips' else 'argmax' if method == 'logged_argmax' else 'bandit')
POOL_SUITES = ('public', 'decisionbench', 'typed', 'emotion', 'agnews')   # suites whose training rows give feedback
EVAL_SUITES = POOL_SUITES + ('jevbench',)                                  # pooled Peekaboo never sees JevBench
CLEAN = (0.0, 1.0, 1.0)


def observe(r, spec):
    """True 0/1 correctness -> (observed reward, reported mask). Users report based on the truth, then a report
    may be flipped. Randomness is drawn only when the spec needs it, so clean feedback replays the paper exactly."""
    flip, p_right, p_wrong = spec
    seen = torch.ones_like(r, dtype=torch.bool)
    if p_right < 1 or p_wrong < 1:
        seen = torch.rand_like(r) < torch.where(r > 0, torch.full_like(r, p_right), torch.full_like(r, p_wrong))
    if flip > 0:
        r = torch.where(torch.rand_like(r) < flip, 1 - r, r)
    return r, seen
FOLDS = 5


def gold(it):
    return it['labels'].index(it['gold'])


def stable(s):
    import bench_tasks as BT
    return BT.stable_hash(s)


# ───────────────────────────── data ─────────────────────────────
def load_suite(suite):
    items = json.load(open(paths.items(f'items_{suite}.json')))['items']
    bank = torch.load(paths.bank(suite))
    released = json.load(open(os.path.join(paths.RESULTS, f'released_{suite}.json')))
    ok = lambda it: 'feats' in bank.get(it['id'], {})
    test = [it for it in items if it['split'] == 'test']
    pool = [it for it in items if it['split'] == 'pool' and ok(it)]
    return items, bank, released, test, pool


def cal_split(pool, cal_frac=0.2):
    """Per task, a fixed 20% of the pool held out for calibration (same rule as the paper's comparison)."""
    rng = np.random.default_rng(0)
    train, cal = [], []
    for t in sorted({it['task'] for it in pool}):
        rows = [it for it in pool if it['task'] == t]
        rows = [rows[i] for i in rng.permutation(len(rows))]
        k = int(round(cal_frac * len(rows)))
        cal += rows[:k]; train += rows[k:]
    return train, cal


def jev_fold(it):
    return stable(str(it.get('group') or it['id'])) % FOLDS


def jev_cal(it):
    """v2: the calibration fifth of a JevBench training fold is drawn by paraphrase group, so a paraphrase of a
    training row is never a calibration row (v1 drew it by row id)."""
    return not stable('cal' + str(it.get('group') or it['id'])) % 5


# ───────────────────────────── calibration ─────────────────────────────
def temp(P, T):
    out = []
    for p in P:
        l = np.log(np.maximum(p, 1e-12)) / T
        e = np.exp(l - l.max()); out.append(e / e.sum())
    return out


def fit_T_labels(P, rows):
    g = [gold(it) for it in rows]
    return float(min(TGRID, key=lambda T: -np.mean([np.log(max(p[gi], 1e-12)) for p, gi in zip(temp(P, T), g)])))


def fit_T_feedback(P, rows, rng, spec=CLEAN, actions=None):
    """Sample a ~ p, observe r = [a correct]; maximise sum log Bernoulli(r; p_T(a)). Labels only through r.
    `spec` delivers the feedback as in training (noisy / missing); `actions` replaces sampling with logged answers
    (valid because a calibrated model is right on any answer a with probability p(a), whoever chose a)."""
    logs = []
    flip, p_right, p_wrong = spec
    for i, (p, it) in enumerate(zip(P, rows)):
        a = int(rng.choice(len(p), p=p / p.sum())) if actions is None else int(actions[i])
        r = float(a == gold(it))
        if (p_right < 1 or p_wrong < 1) and rng.random() >= (p_right if r else p_wrong):
            continue
        if flip > 0 and rng.random() < flip:
            r = 1.0 - r
        logs.append((p, a, r))

    def ll(T):
        tot = 0.0
        for p, a, r in logs:
            q = min(max(temp([p], T)[0][a], 1e-6), 1 - 1e-6); tot += r * np.log(q) + (1 - r) * np.log(1 - q)
        return tot
    return float(max(TGRID, key=ll))


# ───────────────────────────── adapters ─────────────────────────────
def train_adapter(v, bank, train, method, rng, epochs=8, lr=1e-3, kl=0.1, batch=32, spec=CLEAN, cal=None, shipped=True):
    """Returns (policy, reward curve, early-stopping info). With `cal` given and method 'sup', the adapter state with
    the best argmax accuracy on the calibration rows (labels) is kept in es['state'] (the tuned answer-key arm)."""
    pol = Policy(v, lr, rank=8, shipped=shipped)
    base, curve, log = 0.5, [], None
    es = {'cal_acc': [], 'best_epoch': None, 'state': None}
    if method in ('ips', 'argmax'):                  # log one answer per row from the released model, once
        log = {}
        for it, p in zip(train, predict(pol, bank, train, adapter=False)):
            p = np.asarray(p, float); a = int(rng.choice(len(p), p=p / p.sum())) if method == 'ips' else int(p.argmax())
            log[it['id']] = (a, float(a == gold(it)), float(p[a]))
        base = float(np.mean([x[1] for x in log.values()]))
    for ep in range(epochs):
        order = rng.permutation(len(train)); rew = []
        for s in range(0, len(train), batch):
            rows = [train[i] for i in order[s:s + batch]]
            g = torch.tensor([gold(it) for it in rows], device=v.device)
            lp, mask = pol.logp(bank, rows)
            ar = torch.arange(len(rows), device=v.device)
            if method == 'sup':
                loss = -lp[ar, g].mean()
            elif method == 'ips':                    # clipped importance weights on the logged answers
                a = torch.tensor([log[it['id']][0] for it in rows], device=v.device)
                r = torch.tensor([log[it['id']][1] for it in rows], device=v.device)
                p0 = torch.tensor([log[it['id']][2] for it in rows], device=v.device)
                loss = -((r - base) * (lp[ar, a].exp() / p0).clamp(max=10.0)).mean(); rew.append(float(r.mean()))
            elif method == 'argmax':                 # the released model's best answer, rated once: right -> that answer is
                a = torch.tensor([log[it['id']][0] for it in rows], device=v.device)   # the label; wrong -> it is not
                r = torch.tensor([log[it['id']][1] for it in rows], device=v.device)
                pa = lp[ar, a].exp().clamp(max=1 - 1e-6)
                loss = -(r * lp[ar, a] + (1 - r) * torch.log1p(-pa)).mean(); rew.append(float(r.mean()))
            else:
                with torch.no_grad():
                    a = torch.multinomial(lp.exp(), 1).squeeze(1); r_true = (a == g).float()
                    r, seen = observe(r_true, spec)
                if bool(seen.all()):
                    loss = -((r - base) * lp[ar, a]).mean()
                    base = 0.9 * base + 0.1 * float(r.mean())
                else:
                    sf = seen.float(); n = float(sf.sum())
                    loss = -((r - base) * lp[ar, a] * sf).sum() / max(n, 1.0)
                    if n:
                        base = 0.9 * base + 0.1 * float((r * sf).sum() / n)
                rew.append(float(r_true.mean()))
            if kl:
                with torch.no_grad():
                    lq, _ = pol.logp(bank, rows, adapter=False)
                loss = loss + kl * (lp.exp() * (lp - lq)).masked_fill(~mask, 0).sum(-1).mean()
            pol.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.params, 1.0); pol.opt.step()
        curve.append(float(np.mean(rew)) if rew else None)
        if method == 'sup' and cal:                  # early stopping on the calibration rows' labels (the labels are
            acc = float(np.mean([int(np.argmax(p)) == gold(it) for p, it in zip(predict(pol, bank, cal), cal)]))   # the method's own)
            es['cal_acc'].append(acc)
            if es['best_epoch'] is None or acc > max(es['cal_acc'][:-1]):
                es['best_epoch'] = ep + 1; es['state'] = [p.detach().clone() for p in pol.params]
    return pol, curve, es


def switch_check(Pa, P0, cal, rng):
    """Feedback-checked switch, v2: on held-out rows the adapter's best answer and the released model's best answer are
    each rated right / wrong (two ratings per row), and the adapter is kept only if it was right more often. v1 rated
    one *sampled* answer from each, which measured the expected probability of the gold answer, a quantity the
    honesty fix lowers for the adapter and over-confidence raises for the base model; it is kept as 'sampled'."""
    if not cal:
        return {'adapter': 0.0, 'base': 0.0, 'on': False, 'n': 0, 'sampled': None}
    acc = lambda P: float(np.mean([int(np.argmax(p)) == gold(it) for p, it in zip(P, cal)]))
    rate = lambda P: float(np.mean([int(rng.choice(len(p), p=p / p.sum())) == gold(it) for p, it in zip(P, cal)]))
    a, b = acc(Pa), acc(P0)
    sa, sb = rate(Pa), rate(P0)
    return {'adapter': a, 'base': b, 'on': bool(a > b), 'n': len(cal), 'sampled': {'adapter': sa, 'base': sb, 'on': bool(sa > sb)}}


def run_adapter(v, bank, train, cal, test, method, rng):
    kind = KIND(method)
    spec = FEEDBACK.get(method, CLEAN)
    pol, curve, es = train_adapter(v, bank, train, kind, rng, epochs=EPOCHS.get(method, 8), kl=KL.get(method, 0.1), spec=spec,
                                   cal=cal, shipped=not NOSHIP(method))
    Pc = [np.asarray(p, float) for p in predict(pol, bank, cal)]
    Pt = [np.asarray(p, float) for p in predict(pol, bank, test)]
    P0c = [np.asarray(p, float) for p in predict(pol, bank, cal, adapter=False)]
    out_es = None
    if kind == 'sup':
        T = fit_T_labels(Pc, cal)
        if es['state'] is not None:                  # the early-stopped answer-key adapter, scored the same way
            final = [p.detach().clone() for p in pol.params]
            with torch.no_grad():
                for p, q in zip(pol.params, es['state']):
                    p.copy_(q)
            Pc_es = [np.asarray(p, float) for p in predict(pol, bank, cal)]
            Pt_es = [np.asarray(p, float) for p in predict(pol, bank, test)]
            out_es = temp(Pt_es, fit_T_labels(Pc_es, cal))
            with torch.no_grad():
                for p, q in zip(pol.params, final):
                    p.copy_(q)
    elif kind == 'ips':                              # calibration from answers logged by the released model too
        T = fit_T_feedback(Pc, cal, rng, actions=[int(rng.choice(len(p), p=p / p.sum())) for p in P0c])
    elif kind == 'argmax':
        T = fit_T_feedback(Pc, cal, rng, actions=[int(np.argmax(p)) for p in P0c])
    else:
        T = fit_T_feedback(Pc, cal, rng, spec=spec)
    sw = None
    if kind != 'sup':                                # drawn after everything above, so earlier results are unchanged
        sw = switch_check(temp(Pc, T), P0c, cal, rng)
    calp = {'ids': [it['id'] for it in cal], 'adapter': [[round(float(x), 6) for x in p] for p in Pc], 'base': [[round(float(x), 6) for x in p] for p in P0c]}
    info = {'T': T, 'reward_curve': curve, 'n_train': len(train), 'n_cal': len(cal), 'feedback': spec, 'switch': sw, 'epochs': EPOCHS.get(method, 8),
            'early_stop': {'cal_acc': es['cal_acc'], 'best_epoch': es['best_epoch']} if kind == 'sup' else None, 'cal': calp}
    return temp(Pt, T), info, pol, Pt, out_es


class Bilinear(torch.nn.Module):
    """No-engine control: logit_k = <A s, C o_k> + w . C o_k over standardised frozen features."""
    def __init__(self, ds, do, h=64):
        super().__init__()
        self.A, self.C, self.w = torch.nn.Linear(ds, h), torch.nn.Linear(do, h), torch.nn.Linear(h, 1)

    def forward(self, s, o, mask):
        lg = (self.A(s)[:, None, :] * self.C(o)).sum(-1) + self.w(self.C(o)).squeeze(-1)
        # -1e9, not -1e4: on Vega 4B (15,360-dim features) real logits can fall below -1e4, and the softmax then puts all
        # the mass on padded slots (NaN crash found in the 4B reproduction, reproductions/4b/0001-*.patch). Not -inf: the
        # entropy term multiplies exp(lp) * lp over all slots.
        return torch.log_softmax(lg.masked_fill(~mask, -1e9), -1)


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
    Pc, Pt = probs(cal), probs(test)
    T = fit_T_feedback(Pc, cal, rng)
    return temp(Pt, T), {'T': T, 'n_train': len(train), 'n_cal': len(cal)}


# ───────────────────────────── VegaML's TTT head ─────────────────────────────
def ttt(suite, bank, released, pool, test):
    """public: the notebook's own ladder (shots 6/12/25/50 on the fit rows, the largest fitted head, used where it
    beats chance). decisionbench: one head per task on its pool rows, same "auto" rule. Records in notebook format.
    v2 also returns the records under vegaml 0.12.3's rule ('guard': a head is used only where its cross-validated
    accuracy beats both chance and the released engine's accuracy on the same fit rows)."""
    import bench_readouts as RO
    from vegaml import TTTHead
    X = lambda it: bank[it['id']]['feats'].numpy().ravel()
    recs, recs_guard, info = {}, {}, {}
    for t in sorted({it['task'] for it in test}):
        fit = [it for it in pool if it['task'] == t]
        rows = [it for it in test if it['task'] == t]
        labels = rows[0]['labels']
        head, cv, chance = None, None, 1.0 / max(len(labels), 2)
        eng_acc = float(np.mean([released.get(it['id'], {}).get('pred') == it['gold'] for it in fit])) if fit else None
        if suite == 'public':
            ladder = RO.fit_head_ladder([X(it) for it in fit], [it['gold'] for it in fit], labels, [6, 12, 25, 50], folds=5)
            fitted = [k for k in sorted(ladder) if ladder[k].get('fitted')]
            if fitted:
                best = ladder[fitted[-1]]
                head, cv = best['head'], best.get('cv_accuracy')
                chance = best.get('chance', chance)
                info[t] = {'shots': fitted[-1], 'cv': cv, 'used': bool(best.get('beats_chance')), 'engine_acc_fit': eng_acc}
                if not best.get('beats_chance'):
                    head = None
        else:
            y = [it['gold'] for it in fit]
            labs = sorted({l for it in fit for l in it['labels']})
            if len(set(y)) >= 2 and len(fit) >= 6:
                h = TTTHead(labs).fit([X(it) for it in fit], y)
                cv = h.cross_val_accuracy([X(it) for it in fit], y, folds=5)
                chance = 1.0 / max(len(labs), 2)
                if cv is not None and cv > chance:
                    head = h
                info[t] = {'cv': cv, 'used': head is not None, 'engine_acc_fit': eng_acc}
        guard = head is not None and cv is not None and eng_acc is not None and cv > max(chance, eng_acc)
        if t in info:
            info[t]['used_guard'] = bool(guard)
        for it in rows:
            eng = dict(released[it['id']]); eng['readout'] = 'engine'
            if head is not None and 'feats' in bank.get(it['id'], {}) and set(it['labels']) <= set(head.labels):
                r = RO.head_record(head, X(it), it, it['labels'], allowed=it['labels'])
                r.update(labels=it['labels'], readout='ttt', latency_ms=eng.get('latency_ms'))
                recs[it['id']] = r
                recs_guard[it['id']] = r if guard else eng
            else:
                recs[it['id']] = eng; recs_guard[it['id']] = eng
    return recs, recs_guard, {'tasks': info, 'rows_by_head': sum(r.get('readout') == 'ttt' for r in recs.values()),
                              'rows_by_head_guard': sum(r.get('readout') == 'ttt' for r in recs_guard.values())}


# ───────────────────────────── records ─────────────────────────────
def to_records(rows, P, released, extra_ms=0.0):
    out = {}
    for it, p in zip(rows, P):
        probs = {lab: float(x) for lab, x in zip(it['labels'], p)}
        lat = released.get(it['id'], {}).get('latency_ms')
        out[it['id']] = {'id': it['id'], 'gold': it['gold'], 'pred': max(probs, key=probs.get), 'probs': probs,
                         'labels': it['labels'], 'latency_ms': None if lat is None else lat + extra_ms}
    return out


def overhead_ms(v, bank, rows, pol=None):
    """p50 engine time per decision, batch 1, on this device: the released engine, or with our adapter."""
    ts = []
    with torch.no_grad():
        for it in rows[:220]:
            t = time.perf_counter()
            B.run(v, [it], bank) if pol is None else pol.logp(bank, [it])
            ts.append(1000 * (time.perf_counter() - t))
    return float(np.percentile(ts[20:] or ts, 50))


# ───────────────────────────── pooled Peekaboo ─────────────────────────────
def pooled(seed, rng):
    """One Peekaboo adapter trained on the right / wrong feedback from every suite's training rows together
    (POOL_SUITES), then used on every suite. JevBench gives no feedback at all: it is pure transfer. Each suite's
    temperature and switch use only that suite's held-out feedback rows (JevBench: the union of all of them)."""
    data, bank_all, trains, cals = {}, {}, [], {}
    tag = lambda s, it: dict(it, id=f'{s}|{it["id"]}', _orig=it['id'])
    for s in EVAL_SUITES:
        if not (os.path.exists(paths.bank(s)) and os.path.exists(os.path.join(paths.RESULTS, f'released_{s}.json'))):
            continue
        items, bank, released, test, pool = load_suite(s)
        data[s] = (released, test)
        for k, x in bank.items():
            bank_all[f'{s}|{k}'] = x
        if s in POOL_SUITES and pool:
            tr, ca = cal_split(pool)
            trains += [tag(s, it) for it in tr]; cals[s] = [tag(s, it) for it in ca]
    print(f'{paths.LABEL} pooled s{seed}: {len(trains)} feedback rows from {sorted(cals)}; evaluating {sorted(data)}', flush=True)
    t0 = time.time()
    v = B.engine()
    pol, curve, _ = train_adapter(v, bank_all, trains, 'bandit', rng, kl=0.1)
    all_cal = [it for c in cals.values() for it in c]
    for s, (released, test) in data.items():
        usable = [tag(s, it) for it in test if 'feats' in bank_all.get(f'{s}|{it["id"]}', {})]
        cal = cals.get(s) or all_cal
        Pc = [np.asarray(p, float) for p in predict(pol, bank_all, cal)]
        T = fit_T_feedback(Pc, cal, rng)
        Pt = [np.asarray(p, float) for p in predict(pol, bank_all, usable)]
        sw = switch_check(temp(Pc, T), [np.asarray(p, float) for p in predict(pol, bank_all, cal, adapter=False)], cal, rng) if s in cals else None
        orig = [dict(it, id=it['_orig']) for it in usable]
        recs, raw = to_records(orig, temp(Pt, T), released), to_records(orig, Pt, released)
        swr = recs if (sw is None or sw['on']) else {it['id']: dict(released[it['id']], labels=it['labels']) for it in orig}
        miss = lambda it: {'id': it['id'], 'gold': it['gold'], 'pred': None, 'probs': None, 'labels': it['labels'], 'error': 'no features'}
        out_dir = os.path.join(paths.RESULTS, 'methods', s); os.makedirs(out_dir, exist_ok=True)
        acc = np.mean([recs.get(it['id'], {}).get('pred') == it['gold'] for it in test])
        print(f'{paths.LABEL} {s} pooled s{seed}: accuracy {acc:.4f} (T={T:.2f}, switch {sw and sw["on"]})', flush=True)
        json.dump({'suite': s, 'method': 'pooled', 'seed': seed, 'size': paths.SIZE,
                   'info': {'T': T, 'switch': sw, 'n_train': len(trains), 'feedback_suites': sorted(cals), 'reward_curve': curve},
                   'records': [recs.get(it['id']) or miss(it) for it in test], 'records_raw': [raw.get(it['id']) or miss(it) for it in test],
                   'records_switch': [swr.get(it['id']) or miss(it) for it in test]},
                  open(os.path.join(out_dir, f'pooled_s{seed}.json'), 'w'))
    open(os.path.join(paths.RESULTS, 'methods', f'pooled_s{seed}.done'), 'w').write(f'{time.time() - t0:.0f}s\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', required=True); ap.add_argument('--method', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.method == 'pooled':                         # one adapter for every suite: python adapt.py --suite all --method pooled
        os.makedirs(os.path.join(paths.RESULTS, 'methods'), exist_ok=True)
        if os.path.exists(os.path.join(paths.RESULTS, 'methods', f'pooled_s{a.seed}.done')):
            print('skip pooled', a.seed); return
        torch.manual_seed(a.seed)
        return pooled(a.seed, np.random.default_rng(a.seed))
    out_dir = os.path.join(paths.RESULTS, 'methods', a.suite); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{a.method}_s{a.seed}.json')
    if os.path.exists(out):
        print('skip', a.suite, a.method, a.seed); return
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    items, bank, released, test, pool = load_suite(a.suite)
    usable = [it for it in test if 'feats' in bank.get(it['id'], {})]
    if os.environ.get('SMOKE'):                      # a quick code-path check on a few rows (never for results)
        pool, usable, test = pool[:160], usable[:64], test[:64]
    t0, info, recs, raw, swr = time.time(), {}, {}, {}, {}

    guard, esr = {}, {}
    if a.method == 'ttt':
        if a.suite == 'jevbench':
            raise SystemExit('ttt does not apply to jevbench (no labelled rows; the notebook fits nothing there)')
        recs, guard, info = ttt(a.suite, bank, released, pool, test)
    elif a.method in ('replay', 'noship_released'):  # check: the engine on cached features must equal decide();
        v = B.engine()                               # noship_released: the released engine with its shipped adapters off
        P = B.run(v, usable, bank, adapters='off' if a.method == 'noship_released' else None)
        info = {'rows': len(usable)}
        if a.method == 'replay':
            info['max_abs_dp'] = B.replay_diff(usable, P, a.suite)
            print(f'{paths.LABEL} {a.suite}: replay vs decide() max |dp| = {info["max_abs_dp"]:.2e} over {len(usable)} rows', flush=True)
        recs = to_records(usable, P, released)
    else:
        if a.suite == 'jevbench':                    # cross-fitting: no pool exists
            folds = [(f, [it for it in usable if jev_fold(it) != f], [it for it in usable if jev_fold(it) == f]) for f in range(FOLDS)]
        else:
            folds = [(0, pool, usable)]
        info['folds'] = []
        for f, tr_pool, te in folds:
            train, cal = cal_split(tr_pool) if a.suite != 'jevbench' else (
                [it for it in tr_pool if not jev_cal(it)], [it for it in tr_pool if jev_cal(it)])
            if a.method in ADAPTER_METHODS:
                v = B.engine()
                base_ms = overhead_ms(v, bank, te)
                P, fi, pol, Praw, Pes = run_adapter(v, bank, train, cal, te, a.method, rng)
                raw.update(to_records(te, Praw, released))
                if Pes is not None:
                    esr.update(to_records(te, Pes, released))
                on = (fi.get('switch') or {}).get('on', True)
                swr.update(to_records(te, P, released) if on else {it['id']: dict(released[it['id']], labels=it['labels']) for it in te})
                fi['engine_ms_p50'] = {'released': base_ms, 'with_adapter': overhead_ms(v, bank, te, pol)}
                extra = max(0.0, fi['engine_ms_p50']['with_adapter'] - base_ms)
                del pol, v
            elif a.method == 'linear_fb':
                P, fi = linear_fb(bank, train, cal, te, rng); extra = 0.0
            else:
                raise SystemExit('unknown method ' + a.method)
            recs.update(to_records(te, P, released, extra))
            info['folds'].append(dict(fi, fold=f, n_test=len(te)))
        info['adapter_extra_ms'] = float(np.mean([fi.get('engine_ms_p50', {}).get('with_adapter', 0) - fi.get('engine_ms_p50', {}).get('released', 0)
                                                  for fi in info['folds']])) if a.method in ADAPTER_METHODS else None
    for it in test:                                  # rows decide() could not read count as wrong, as in the notebook
        if it['id'] not in recs:
            recs[it['id']] = {'id': it['id'], 'gold': it['gold'], 'pred': None, 'probs': None, 'labels': it['labels'],
                              'error': released.get(it['id'], {}).get('error', 'no features')}
    acc = np.mean([r['pred'] == r['gold'] for r in recs.values()])
    print(f'{paths.LABEL} {a.suite} {a.method} s{a.seed}: accuracy {acc:.4f} on {len(recs)} rows, {time.time() - t0:.0f}s', flush=True)
    extra = {}
    # Peekaboo without the honesty fix; with the switch; the early-stopped answer-key adapter; TTT under the 0.12.3 rule
    for name, d in (('records_raw', raw), ('records_switch', swr), ('records_es', esr), ('records_guard', guard)):
        if d:
            extra[name] = [d.get(it['id'], recs[it['id']]) for it in test]
    json.dump(dict({'suite': a.suite, 'method': a.method, 'seed': a.seed, 'size': paths.SIZE, 'info': info,
                    'records': [recs[it['id']] for it in test]}, **extra), open(out + '.tmp', 'w'))
    os.replace(out + '.tmp', out)


if __name__ == '__main__':
    main()
