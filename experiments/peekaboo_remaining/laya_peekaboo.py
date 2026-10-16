"""Laya on its own headline benchmark (LocalLLaMA/typed-decisions): base Laya, Laya's own answer-key fine-tune,
and Laya + Peekaboo.

  VEGA_SIZE=laya      base Laya, the English checkpoint as released (convaiinnovations/laya, 421M): README 0.362
  VEGA_SIZE=laya_td   laya-typed-decisions, Laya's released fine-tune of the same model on the same training split
                      with the answer key (its RLCD recipe): README 0.766
  Laya + Peekaboo     base Laya plus Peekaboo trained on that training split with right / wrong only

Same rows, splits, record format and metrics as the Vega runs (experiments/vega_policy). A yes/no question is asked
as Laya's own "noul" type, whose options are [false, true], the same order as Vega's labels.

  cache   read every row once through Laya's own prediction call (agent.system_one): the official record, and the
          tokenised input Laya built for it, captured from inside the call
  train   --method feedback (Peekaboo): a trainable copy of Laya's decision head (the encoder stays frozen), trained
          by REINFORCE on right/wrong only, KL anchor (0.1) to the released head, then one temperature fitted from
          feedback on held-out training rows. Head learning rate 1e-4, as in Laya's own recipe.
          --method replay: check that the frozen model on the captured input reproduces the official answer

    VEGA_SIZE=laya python laya_peekaboo.py cache --suite typed
    VEGA_SIZE=laya python laya_peekaboo.py train --suite typed --method feedback --seed 0
Output (VEGA_OUT): bank_typed_<size>.pt, results_<size>/released_typed.json, results_<size>/methods/typed/*.json
"""
import argparse, copy, json, math, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'vega_policy'))
os.environ.setdefault('VEGA_SIZE', 'laya')          # paths.py files everything under *_laya / *_laya_td
import numpy as np
import torch

import paths
import adapt as A

CHECKPOINT = {'laya': 'convaiinnovations/laya', 'laya_td': 'typed-decisions'}   # laya.load resolves the alias


def to_laya(q):
    q = dict(q)
    if q['type'] == 'boolean':
        q['type'] = 'noul'
    return q


def answer_probs(ans, labels):
    if ans.get('type') == 'noul':
        p = float(ans['noul']); return {'false': 1.0 - p, 'true': p}
    pr = ans.get('probabilities') or {}
    return {lab: float(pr.get(lab, 0.0)) for lab in labels}


def load_agent():
    import laya
    ag = laya.load(CHECKPOINT[paths.SIZE], device=paths.device())
    ag.model.eval()
    return ag


# ───────────────────────────── cache: base Laya + its inputs ─────────────────────────────
def cache(suite):
    from suite_cache import load_items
    items = load_items(suite)
    os.makedirs(paths.RESULTS, exist_ok=True)
    bank_path, rec_path = paths.bank(suite), os.path.join(paths.RESULTS, f'released_{suite}.json')
    marker = os.path.join(paths.RESULTS, f'cache_{suite}.done')
    bank = torch.load(bank_path) if os.path.exists(bank_path) else {}
    recs = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
    todo = [it for it in items if it['id'] not in recs or it['id'] not in bank]
    print(f'Laya {suite}: {len(items)} rows, {len(todo)} to read', flush=True)
    if todo:
        ag = load_agent()
        cap, orig = {}, ag._infer

        def hooked(b):
            cap['b'] = {k: v.detach().cpu().clone() for k, v in b.items() if torch.is_tensor(v)}
            return orig(b)
        ag._infer = hooked

        def save():
            torch.save(bank, bank_path + '.tmp'); os.replace(bank_path + '.tmp', bank_path)
            json.dump(recs, open(rec_path + '.tmp', 'w')); os.replace(rec_path + '.tmp', rec_path)
        t0 = time.time()
        for n, it in enumerate(todo, 1):
            cap.clear()
            t = time.perf_counter()
            try:
                out = ag.system_one(it['state'], {'q': to_laya(it['question'])})
                ans = (out.get('answers') or out)['q']
                probs = answer_probs(ans, it['labels'])
                rec = {'id': it['id'], 'gold': it['gold'], 'pred': max(probs, key=probs.get), 'probs': probs,
                       'latency_ms': 1000 * (time.perf_counter() - t), 'truncated': bool((out.get('usage') or {}).get('truncated'))}
            except Exception as exc:                  # scored as wrong, as VegaML's harness does
                rec = {'id': it['id'], 'gold': it['gold'], 'pred': None, 'probs': None, 'error': '%s: %s' % (type(exc).__name__, str(exc)[:200])}
            rec['labels'] = it['labels']; recs[it['id']] = rec
            b = cap.get('b')
            if b is not None and 'error' not in rec:
                L = int(b['attention_mask'][0].sum())
                mm = b['marker_mask'][0].bool()
                bank[it['id']] = {'ids': b['input_ids'][0, :L].to(torch.int32), 'markers': b['marker_pos'][0][mm].to(torch.int32),
                                  'qtype': int(b['qtype'][0])}
                if len(bank[it['id']]['markers']) != len(it['labels']):
                    bank[it['id']] = {'error': 'option count mismatch'}
            else:
                bank[it['id']] = {'error': rec.get('error', 'no input captured')}
            if n % 500 == 0 or n == len(todo):
                save(); el = time.time() - t0
                print(f'  {n}/{len(todo)}  {el:.0f}s  ({el / n:.3f} s/row)', flush=True)
    if all(it['id'] in recs and it['id'] in bank for it in items):
        errs = sum('error' in r for r in recs.values())
        open(marker, 'w').write(f'{len(items)} rows, {errs} unanswerable\n')
        print(f'done: {len(recs)} records, {errs} rows Laya could not answer', flush=True)


# ───────────────────────────── the frozen encoder, a head on top ─────────────────────────────
class Laya:
    def __init__(self):
        from laya.common import temp_bucket
        self.ag = load_agent()
        self.m, self.dev = self.ag.model, torch.device(paths.device())
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.pad = self.ag.tok.pad_token_id
        self.tb = temp_bucket

    def ship_temp(self, r):
        k = len(r['markers'])
        return float(self.ag.temperature_by_options.get(self.tb(r['qtype'], k), self.ag.temperature[r['qtype']]))

    def batch(self, bank, rows):
        R = [bank[it['id']] for it in rows]
        L, K = max(len(r['ids']) for r in R), max(len(r['markers']) for r in R)
        ids = torch.full((len(R), L), self.pad, dtype=torch.long); att = torch.zeros(len(R), L, dtype=torch.long)
        mpos = torch.zeros(len(R), K, dtype=torch.long); mmask = torch.zeros(len(R), K, dtype=torch.bool)
        for i, r in enumerate(R):
            ids[i, :len(r['ids'])] = r['ids'].long(); att[i, :len(r['ids'])] = 1
            mpos[i, :len(r['markers'])] = r['markers'].long(); mmask[i, :len(r['markers'])] = True
        qt = torch.tensor([r['qtype'] for r in R]); T = torch.tensor([self.ship_temp(r) for r in R])
        d = self.dev
        return ids.to(d), att.to(d), mpos.to(d), mmask.to(d), qt.to(d), T.to(d)

    def encode(self, ids, att):
        with torch.no_grad(), torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == 'cuda'):
            return self.m.encoder(input_ids=ids, attention_mask=att).last_hidden_state.float()

    def head_logits(self, mods, h, att, mpos, mmask, qt):
        """DecisionModel.forward after the encoder, with any head (the released one or a trainable copy)."""
        h = h + mods['type_emb'](qt)[:, None, :]
        pad = ~att.bool()
        for layer in mods['head'].layers:
            h = layer(h, src_key_padding_mask=pad)
        idx = mpos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        return mods['scorer'](torch.gather(h, 1, idx)).squeeze(-1).float().masked_fill(~mmask, -1e4)

    def released(self):
        return {'head': self.m.head, 'type_emb': self.m.type_emb, 'scorer': self.m.scorer}

    def logp(self, bank, rows, mods, h=None):
        ids, att, mpos, mmask, qt, T = self.batch(bank, rows)
        h = self.encode(ids, att) if h is None else h
        lg = self.head_logits(mods, h, att, mpos, mmask, qt) / T[:, None]
        return torch.log_softmax(lg.masked_fill(~mmask, -1e4), -1), mmask, h

    def predict(self, bank, rows, mods, bs=32):
        out = []
        with torch.no_grad():
            for s in range(0, len(rows), bs):
                lp, mmask, _ = self.logp(bank, rows[s:s + bs], mods)
                out += [lp[b, :int(mmask[b].sum())].exp().cpu().numpy().astype(float) for b in range(len(lp))]
        return out


def peekaboo(L, bank, train, cal, test, method, rng, epochs=8, lr=1e-4, kl=0.1, batch=32):
    """Peekaboo on Laya: the same loop as Vega's (adapt.train_adapter), on a trainable copy of the decision head."""
    spec = A.FEEDBACK.get(method, A.CLEAN)
    ref = L.released()
    mods = {k: copy.deepcopy(v).eval() for k, v in ref.items()}
    params = [p for m in mods.values() for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    base, curve, log = 0.5, [], None
    if method == 'logged_ips':
        log = {}
        for it, p in zip(train, L.predict(bank, train, ref)):
            a = int(rng.choice(len(p), p=p / p.sum())); log[it['id']] = (a, float(a == A.gold(it)), float(p[a]))
        base = float(np.mean([x[1] for x in log.values()]))
    for ep in range(epochs):
        order = rng.permutation(len(train)); rew = []
        for s in range(0, len(train), batch):
            rows = [train[i] for i in order[s:s + batch]]
            g = torch.tensor([A.gold(it) for it in rows], device=L.dev)
            lp, mask, h = L.logp(bank, rows, mods)
            ar = torch.arange(len(rows), device=L.dev)
            if method == 'logged_ips':
                a = torch.tensor([log[it['id']][0] for it in rows], device=L.dev)
                r = torch.tensor([log[it['id']][1] for it in rows], device=L.dev)
                p0 = torch.tensor([log[it['id']][2] for it in rows], device=L.dev)
                loss = -((r - base) * (lp[ar, a].exp() / p0).clamp(max=10.0)).mean(); rew.append(float(r.mean()))
            else:
                with torch.no_grad():
                    a = torch.multinomial(lp.exp(), 1).squeeze(1); r_true = (a == g).float()
                    r, seen = A.observe(r_true, spec)
                sf = seen.float(); n = float(sf.sum())
                loss = -((r - base) * lp[ar, a] * sf).sum() / max(n, 1.0)
                if n:
                    base = 0.9 * base + 0.1 * float((r * sf).sum() / n)
                rew.append(float(r_true.mean()))
            with torch.no_grad():
                lq, _, _ = L.logp(bank, rows, ref, h=h)
            loss = loss + kl * (lp.exp() * (lp - lq)).masked_fill(~mask, 0).sum(-1).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        curve.append(float(np.mean(rew)))
    Pc, Pt = L.predict(bank, cal, mods), L.predict(bank, test, mods)
    if method == 'logged_ips':
        acts = [int(rng.choice(len(p), p=p / p.sum())) for p in L.predict(bank, cal, ref)]
        T = A.fit_T_feedback(Pc, cal, rng, actions=acts)
    else:
        T = A.fit_T_feedback(Pc, cal, rng, spec=spec)
    return A.temp(Pt, T), {'T': T, 'reward_curve': curve, 'n_train': len(train), 'n_cal': len(cal), 'feedback': spec}


def train_main(suite, method, seed):
    out_dir = os.path.join(paths.RESULTS, 'methods', suite); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{method}_s{seed}.json')
    if os.path.exists(out):
        print('skip', suite, method, seed); return
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    items, bank, released, test, _ = A.load_suite(suite)
    ok = lambda it: 'ids' in bank.get(it['id'], {})
    pool = [it for it in items if it['split'] == 'pool' and ok(it)]
    usable = [it for it in test if ok(it)]
    L = Laya(); t0 = time.time(); recs, info = {}, {'folds': []}
    if method == 'replay':
        P = L.predict(bank, usable, L.released())
        diffs = [max(abs(released[it['id']]['probs'][lab] - float(p[j])) for j, lab in enumerate(it['labels'])) for it, p in zip(usable, P)]
        info = {'max_abs_dp': max(diffs) if diffs else None, 'median_abs_dp': float(np.median(diffs)) if diffs else None, 'rows': len(usable)}
        recs = A.to_records(usable, P, released)
        print(f'Laya {suite}: replay vs official max |dp| = {info["max_abs_dp"]:.2e}', flush=True)
    else:
        folds = ([(f, [it for it in usable if A.jev_fold(it) != f], [it for it in usable if A.jev_fold(it) == f]) for f in range(A.FOLDS)]
                 if suite == 'jevbench' else [(0, pool, usable)])
        for f, tr_pool, te in folds:
            if suite == 'jevbench':
                train = [it for it in tr_pool if A.stable('cal' + it['id']) % 5]; cal = [it for it in tr_pool if not A.stable('cal' + it['id']) % 5]
            else:
                train, cal = A.cal_split(tr_pool)
            P, fi = peekaboo(L, bank, train, cal, te, method, rng)
            recs.update(A.to_records(te, P, released))   # same architecture and cost as base Laya: no extra latency
            info['folds'].append(dict(fi, fold=f, n_test=len(te)))
    for it in test:
        if it['id'] not in recs:
            recs[it['id']] = {'id': it['id'], 'gold': it['gold'], 'pred': None, 'probs': None, 'labels': it['labels'],
                              'error': released.get(it['id'], {}).get('error', 'no input')}
    acc = np.mean([r['pred'] == r['gold'] for r in recs.values()])
    print(f'Laya {suite} {method} s{seed}: accuracy {acc:.4f} on {len(recs)} rows, {time.time() - t0:.0f}s', flush=True)
    json.dump({'suite': suite, 'method': method, 'seed': seed, 'size': 'laya', 'info': info,
               'records': [recs[it['id']] for it in test]}, open(out + '.tmp', 'w'))
    os.replace(out + '.tmp', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['cache', 'train']); ap.add_argument('--suite', required=True)
    ap.add_argument('--method', default='feedback'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    cache(a.suite) if a.stage == 'cache' else train_main(a.suite, a.method, a.seed)
