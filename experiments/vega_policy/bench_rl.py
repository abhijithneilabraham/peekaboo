"""Can Vega learn from deployment feedback? DecisionBench as a contextual bandit.

Each row is one decision. The policy samples an option from Vega's own calibrated distribution and is told only
whether that option was right (reward 1/0); the gold label is never shown. That is the feedback a deployed
decision model actually gets (a thumbs up/down, an outcome observed later).

What is trained: one fresh gated LoRA adapter on top of the shipped engine (the shipped adapters, calibration and
halting stay exactly as released). Methods:
    zero        the released model
    bandit      REINFORCE on the sampled option, running-mean baseline, + KL(pi || released) anchor
    sup         cross-entropy on the gold label (full information: an upper bound, not a feedback method)
Splits: 'rows'  (15 rows per task train, 10 test, all 43 tasks seen)
        'tasks' (33 tasks train, 10 tasks held out entirely)

    python bench_rl.py --method bandit --split rows --seed 0
"""
import argparse, json, os, time
import numpy as np
import torch

import bench_eval as B
import paths
from vegaml._common import QTYPES, attach_adapter, thermal_features, thermal_tau

HERE = os.path.dirname(os.path.abspath(__file__))
RES = paths.RESULTS; os.makedirs(RES, exist_ok=True)


def split(items, how, seed=0, n_train=15, n_test=10):
    rng = np.random.default_rng(seed)
    tasks = sorted({it['task'] for it in items})
    if how == 'tasks':
        test_t = set(rng.permutation(tasks)[:10])
        return [it for it in items if it['task'] not in test_t], [it for it in items if it['task'] in test_t]
    tr, te = [], []
    for t in tasks:
        rows = [it for it in items if it['task'] == t]
        rows = [rows[i] for i in rng.permutation(len(rows))]
        te += rows[-n_test:]; tr += rows[:min(n_train, len(rows) - n_test)]     # test rows fixed whatever n_train is
    return tr, te


class Policy:
    """The released engine plus one trainable adapter. probs() mirrors Vega._view (route + thermal temperature)."""

    def __init__(self, v, lr, rank=8, shipped=True):
        self.v, self.eng = v, v.engine
        self.n_ship = len(v.adapters)
        self.j = attach_adapter(self.eng, rank=rank, alpha=float(rank))
        with torch.no_grad():
            self.eng.gate_nets[self.j][-1].bias.fill_(3.0)      # start open; the gate can still learn to close
            if not shipped:                                      # v2 "_noship": the shipped adapters' gates held shut, so
                for i in range(self.n_ship):                     # the released reference is the base engine (as adapters="off")
                    self.eng.gate_nets[i][-1].bias.fill_(-1e4)
        self.eng.eval()
        for p in self.eng.parameters():
            p.requires_grad_(False)
        self.params = [p for n, p in self.eng.named_parameters()
                       if (('lora_A' in n or 'lora_B' in n) and n.endswith(f'.{self.j}')) or n.startswith(f'gate_nets.{self.j}.')]
        for p in self.params:
            p.requires_grad_(True)
        self.opt = torch.optim.Adam(self.params, lr=lr)

    def batch(self, bank, rows):
        R = [bank[it['id']] for it in rows]
        K = max(r['opts'].shape[0] for r in R)
        feats = torch.stack([r['feats'] for r in R])
        opts = torch.zeros((len(R), K) + tuple(R[0]['opts'].shape[1:])); mask = torch.zeros(len(R), K, dtype=torch.bool)
        for b, r in enumerate(R):
            k = r['opts'].shape[0]; opts[b, :k] = r['opts']; mask[b, :k] = True
        chunks = torch.stack([r['chunks'] for r in R]) if R[0]['chunks'] is not None else None
        cmask = torch.stack([r['cmask'] for r in R]) if R[0]['cmask'] is not None else None
        qt = torch.tensor([QTYPES[it['question']['type']] for it in rows])
        dev = self.v.device
        mv = lambda x: None if x is None else x.to(dev)
        return feats.to(dev), opts.to(dev), mask.to(dev), qt.to(dev), mv(chunks), mv(cmask)

    def logp(self, bank, rows, adapter=True):
        """log pi(.|row) with gradient through the new adapter; shipped adapters route and calibrate as released."""
        feats, opts, mask, qt, chunks, cmask = self.batch(bank, rows)
        if not adapter:
            # the released model: zero the new adapter's contribution by closing its gate
            saved = self.eng.gate_nets[self.j][-1].bias.data.clone()
            self.eng.gate_nets[self.j][-1].bias.data.fill_(-1e4)
        out = self.eng(feats, opts, mask, qt, halt=self.v.halt, return_traj=False, chunks=chunks, chunk_mask=cmask)
        if not adapter:
            self.eng.gate_nets[self.j][-1].bias.data.copy_(saved)
        with torch.no_grad():
            phys = thermal_features(out, mask, self.eng.max_steps).cpu().numpy()
            g = out['gates'].float().cpu().numpy()[:, :self.n_ship] if out.get('gates') is not None else None
            route = self.v._route(g)
            route = np.full(len(rows), -1) if route is None else route
            qn = qt.cpu().numpy()
            tau = np.array([thermal_tau(self.v._cal(int(rt))['theta'], qn[b:b + 1], phys[b:b + 1])[0]
                            for b, rt in enumerate(route)]).reshape(-1)
        lg = out['logits'].float() / torch.as_tensor(tau, dtype=torch.float32, device=out['logits'].device)[:, None]
        return torch.log_softmax(lg.masked_fill(~mask, -1e4), -1), mask


def predict(pol, bank, rows, adapter=True, T=1.0):
    P = []
    with torch.no_grad():
        for s in range(0, len(rows), 64):
            lp, mask = pol.logp(bank, rows[s:s + 64], adapter=adapter)
            lp = torch.log_softmax(lp / T, -1)
            P += [lp[b, :int(mask[b].sum())].exp().cpu().numpy() for b in range(len(lp))]
    return P


def evaluate(pol, bank, rows, adapter=True, T=1.0, keep=False):
    P = predict(pol, bank, rows, adapter, T)
    out = B.score(rows, P)
    if keep:      # per-row correctness and gold probability, for paired tests
        g = [it['labels'].index(it['gold']) for it in rows]
        out['rows'] = {'correct': [int(np.argmax(p) == gi) for p, gi in zip(P, g)], 'p_gold': [float(p[gi]) for p, gi in zip(P, g)]}
    return out


def feedback_temperature(pol, bank, rows, rng, samples=1):
    """Fit one temperature from bandit feedback alone: each logged (sampled option a, reward r) says whether a was
    right, and a calibrated model is right on a sampled option exactly p(a) of the time. So maximise the Bernoulli
    likelihood of r under p_T(a) over a grid of T. No labels are read except through r."""
    with torch.no_grad():
        lp = []
        for s in range(0, len(rows), 64):
            l, mask = pol.logp(bank, rows[s:s + 64]); lp += [l[b, :int(mask[b].sum())] for b in range(len(l))]
    logs = []
    for l, it in zip(lp, rows):
        for _ in range(samples):
            a = int(torch.multinomial(l.exp(), 1)); logs.append((l, a, float(a == it['labels'].index(it['gold']))))
    best = None
    for T in np.exp(np.linspace(np.log(0.5), np.log(8), 41)):
        ll = 0.0
        for l, a, r in logs:
            p = float(torch.softmax(l / T, -1)[a]); p = min(max(p, 1e-6), 1 - 1e-6)
            ll += r * np.log(p) + (1 - r) * np.log(1 - p)
        if best is None or ll > best[1]:
            best = (float(T), ll)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='bandit'); ap.add_argument('--split', default='rows')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--kl', type=float, default=0.1)
    ap.add_argument('--batch', type=int, default=32); ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--items', default='decisionbench_items.json'); ap.add_argument('--n_train', type=int, default=15)
    ap.add_argument('--n_test', type=int, default=10); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    name = f'db_{a.method}_{a.split}_s{a.seed}{a.tag}'
    out = os.path.join(RES, name + '.json')
    if os.path.exists(out):
        print('skip', name); return
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    v, items, bank = B.load(a.items)
    tr, te = split(items, a.split, seed=0, n_train=a.n_train, n_test=a.n_test)          # the split is fixed; the seed varies sampling and init
    pol = Policy(v, a.lr, rank=a.rank)
    hist = [{'epoch': 0, 'test': evaluate(pol, bank, te, adapter=False, keep=True), 'train': evaluate(pol, bank, tr, adapter=False)}]
    print(json.dumps(hist[-1]), flush=True)
    if a.method != 'zero':
        base, t0 = 0.5, time.time()
        for ep in range(1, a.epochs + 1):
            order = rng.permutation(len(tr)); rew = []
            for s in range(0, len(tr), a.batch):
                rows = [tr[i] for i in order[s:s + a.batch]]
                gold = torch.tensor([it['labels'].index(it['gold']) for it in rows], device=pol.v.device)
                lp, mask = pol.logp(bank, rows)
                if a.method == 'sup':
                    loss = -lp[torch.arange(len(rows)), gold].mean()
                else:
                    with torch.no_grad():
                        act = torch.multinomial(lp.exp(), 1).squeeze(1)
                        r = (act == gold).float()                    # the only thing the environment reveals
                    loss = -((r - base) * lp[torch.arange(len(rows)), act]).mean()
                    base = 0.9 * base + 0.1 * float(r.mean()); rew.append(float(r.mean()))
                if a.kl:
                    with torch.no_grad():
                        lq, _ = pol.logp(bank, rows, adapter=False)
                    p = lp.exp()
                    loss = loss + a.kl * (p * (lp - lq)).masked_fill(~mask, 0).sum(-1).mean()
                pol.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.params, 1.0); pol.opt.step()
            hist.append({'epoch': ep, 'secs': round(time.time() - t0), 'train_reward': float(np.mean(rew)) if rew else None,
                         'test': evaluate(pol, bank, te), 'train': evaluate(pol, bank, tr)})
            print(json.dumps(hist[-1]), flush=True)
    final = {'test': evaluate(pol, bank, te, keep=True)}
    if a.method != 'zero':
        T = feedback_temperature(pol, bank, tr, rng)
        final.update(T=T, test_recal=evaluate(pol, bank, te, T=T, keep=True))
        print('feedback temperature', round(T, 3), json.dumps({k: v for k, v in final['test_recal'].items() if k != 'rows'}), flush=True)
    json.dump({'args': vars(a), 'n_train': len(tr), 'n_test': len(te), 'hist': hist, 'final': final}, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
