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
ADAPTER_METHODS = ('sup_ts', 'feedback', 'feedback_kl1')
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


def fit_T_feedback(P, rows, rng):
    """Sample a ~ p, observe r = [a correct]; maximise sum log Bernoulli(r; p_T(a)). Labels only through r."""
    logs = []
    for p, it in zip(P, rows):
        a = int(rng.choice(len(p), p=p / p.sum())); logs.append((p, a, float(a == gold(it))))

    def ll(T):
        tot = 0.0
        for p, a, r in logs:
            q = min(max(temp([p], T)[0][a], 1e-6), 1 - 1e-6); tot += r * np.log(q) + (1 - r) * np.log(1 - q)
        return tot
    return float(max(TGRID, key=ll))


# ───────────────────────────── adapters ─────────────────────────────
def train_adapter(v, bank, train, method, rng, epochs=8, lr=1e-3, kl=0.1, batch=32):
    pol = Policy(v, lr, rank=8)
    base, curve = 0.5, []
    for ep in range(epochs):
        order = rng.permutation(len(train)); rew = []
        for s in range(0, len(train), batch):
            rows = [train[i] for i in order[s:s + batch]]
            g = torch.tensor([gold(it) for it in rows], device=v.device)
            lp, mask = pol.logp(bank, rows)
            ar = torch.arange(len(rows), device=v.device)
            if method == 'sup':
                loss = -lp[ar, g].mean()
            else:
                with torch.no_grad():
                    a = torch.multinomial(lp.exp(), 1).squeeze(1); r = (a == g).float()
                loss = -((r - base) * lp[ar, a]).mean()
                base = 0.9 * base + 0.1 * float(r.mean()); rew.append(float(r.mean()))
            if kl:
                with torch.no_grad():
                    lq, _ = pol.logp(bank, rows, adapter=False)
                loss = loss + kl * (lp.exp() * (lp - lq)).masked_fill(~mask, 0).sum(-1).mean()
            pol.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.params, 1.0); pol.opt.step()
        curve.append(float(np.mean(rew)) if rew else None)
    return pol, curve


def run_adapter(v, bank, train, cal, test, method, rng):
    kind = 'sup' if method == 'sup_ts' else 'bandit'
    pol, curve = train_adapter(v, bank, train, kind, rng, kl=1.0 if method == 'feedback_kl1' else 0.1)
    Pc = [np.asarray(p, float) for p in predict(pol, bank, cal)]
    Pt = [np.asarray(p, float) for p in predict(pol, bank, test)]
    T = fit_T_labels(Pc, cal) if kind == 'sup' else fit_T_feedback(Pc, cal, rng)
    return temp(Pt, T), {'T': T, 'reward_curve': curve, 'n_train': len(train), 'n_cal': len(cal)}, pol


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
    beats chance). decisionbench: one head per task on its pool rows, same "auto" rule. Records in notebook format."""
    import bench_readouts as RO
    from vegaml import TTTHead
    X = lambda it: bank[it['id']]['feats'].numpy().ravel()
    recs, info = {}, {}
    for t in sorted({it['task'] for it in test}):
        fit = [it for it in pool if it['task'] == t]
        rows = [it for it in test if it['task'] == t]
        labels = rows[0]['labels']
        head = None
        if suite == 'public':
            ladder = RO.fit_head_ladder([X(it) for it in fit], [it['gold'] for it in fit], labels, [6, 12, 25, 50], folds=5)
            fitted = [k for k in sorted(ladder) if ladder[k].get('fitted')]
            if fitted:
                best = ladder[fitted[-1]]
                head = best['head'] if best.get('beats_chance') else None
                info[t] = {'shots': fitted[-1], 'cv': best.get('cv_accuracy'), 'used': head is not None}
        else:
            y = [it['gold'] for it in fit]
            labs = sorted({l for it in fit for l in it['labels']})
            if len(set(y)) >= 2 and len(fit) >= 6:
                h = TTTHead(labs).fit([X(it) for it in fit], y)
                cv = h.cross_val_accuracy([X(it) for it in fit], y, folds=5)
                if cv is not None and cv > 1.0 / max(len(labs), 2):
                    head = h
                info[t] = {'cv': cv, 'used': head is not None}
        for it in rows:
            eng = dict(released[it['id']])
            if head is not None and 'feats' in bank.get(it['id'], {}) and set(it['labels']) <= set(head.labels):
                r = RO.head_record(head, X(it), it, it['labels'], allowed=it['labels'])
                r.update(labels=it['labels'], readout='ttt', latency_ms=eng.get('latency_ms'))
                recs[it['id']] = r
            else:
                eng['readout'] = 'engine'; recs[it['id']] = eng
    return recs, {'tasks': info, 'rows_by_head': sum(r.get('readout') == 'ttt' for r in recs.values())}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', required=True); ap.add_argument('--method', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    out_dir = os.path.join(paths.RESULTS, 'methods', a.suite); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{a.method}_s{a.seed}.json')
    if os.path.exists(out):
        print('skip', a.suite, a.method, a.seed); return
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    items, bank, released, test, pool = load_suite(a.suite)
    usable = [it for it in test if 'feats' in bank.get(it['id'], {})]
    t0, info, recs = time.time(), {}, {}

    if a.method == 'ttt':
        if a.suite == 'jevbench':
            raise SystemExit('ttt does not apply to jevbench (no labelled rows; the notebook fits nothing there)')
        recs, info = ttt(a.suite, bank, released, pool, test)
    elif a.method == 'replay':                       # check: the engine on cached features must equal decide()
        v = B.engine()
        P = B.run(v, usable, bank)
        info = {'max_abs_dp': B.replay_diff(usable, P, a.suite), 'rows': len(usable)}
        recs = to_records(usable, P, released)
        print(f'{paths.LABEL} {a.suite}: replay vs decide() max |dp| = {info["max_abs_dp"]:.2e} over {len(usable)} rows', flush=True)
    else:
        if a.suite == 'jevbench':                    # cross-fitting: no pool exists
            folds = [(f, [it for it in usable if jev_fold(it) != f], [it for it in usable if jev_fold(it) == f]) for f in range(FOLDS)]
        else:
            folds = [(0, pool, usable)]
        info['folds'] = []
        for f, tr_pool, te in folds:
            train, cal = cal_split(tr_pool) if a.suite != 'jevbench' else (
                [it for it in tr_pool if stable('cal' + it['id']) % 5], [it for it in tr_pool if not stable('cal' + it['id']) % 5])
            if a.method in ADAPTER_METHODS:
                v = B.engine()
                base_ms = overhead_ms(v, bank, te)
                P, fi, pol = run_adapter(v, bank, train, cal, te, a.method, rng)
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
    json.dump({'suite': a.suite, 'method': a.method, 'seed': a.seed, 'size': paths.SIZE, 'info': info,
               'records': [recs[it['id']] for it in test]}, open(out + '.tmp', 'w'))
    os.replace(out + '.tmp', out)


if __name__ == '__main__':
    main()
