"""Train Vega as a self-play policy in Kuhn poker or text hide-and-seek, on cached backbone features.

    python vtrain.py --env kuhn --brain vega0                 # zero-shot, no training
    python vtrain.py --env kuhn --brain engine --seed 0       # PPO on all 14M engine parameters
    python vtrain.py --env kuhn --brain adapter --seed 0      # PPO on a fresh gated LoRA adapter (base frozen)
    python vtrain.py --env kuhn --brain adapter --kl 0.1      # ... plus a KL anchor to the zero-shot engine

Same game loop, budget, evaluation and opponent pool as ../laya_policy/train.py, so the numbers compare directly.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'laya_policy'))
import train as T                                   # noqa: E402  (game loops, evaluation)
from lpol import kuhn as K                          # noqa: E402
from vpol.vega import FeatureBank, VegaPolicy       # noqa: E402
from extract import QUESTIONS                       # noqa: E402

RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))


class VegaBrain:
    kind = 'text'

    def __init__(self, bank, train, lr, gate='auto', rank=16, contrast=1.0):
        self.P = VegaPolicy(bank, QUESTIONS, train=train, lr=lr, rank=rank, gate=gate, contrast=contrast)
        self.opp = self.P.make_opponent()
        self.opt = self.P.opt
        self.ref = None         # the KL anchor: None = zero-shot Vega; set_magnet() moves it to the current policy

    def set_magnet(self):
        if self.ref is None:
            self.ref = self.P.make_opponent()
        self.P.load_into(self.ref, self.P.state())

    def logits(self, obs, qk, opp=False):
        return self.P.logits(obs, qk, opp=self.opp if opp else None)

    def prior(self, obs, qk):
        if self.ref is None:
            return self.P.prior_logits(obs, qk)
        with torch.no_grad():
            return self.P.logits(obs, qk, opp=self.ref)

    def snapshot(self):
        return self.P.state()

    def set_opponent(self, snap):
        self.P.load_into(self.opp, snap if snap is not None else self.P.state())

    def params(self):
        return self.P.params


class TableKL(T.TableBrain):
    """Control: the tabular policy with the same KL machinery, anchored to uniform (or to its own magnet)."""

    def __init__(self, keys, n_act, lr):
        super().__init__(keys, n_act, lr); self.ref = None

    def prior(self, obs, qk):
        return torch.zeros(len(obs), 2) if self.ref is None else torch.stack([self.ref[o] for o in obs])

    def set_magnet(self):
        self.ref = self.snapshot()


def ppo_update(brain, trans, V, epochs=2, clip=0.2, ent=0.01, alpha=0.1, kl=0.0):
    """train.ppo_update plus an optional KL(pi || zero-shot Vega) penalty per visited state."""
    keys = [t[5] for t in trans]
    ret = np.array([t[4] for t in trans], float)
    adv = ret - np.array([V.get(k, 0.0) for k in keys])
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for k, r in zip(keys, ret):
        V[k] = V.get(k, 0.0) + alpha * (r - V.get(k, 0.0))
    groups = {}
    for j, t in enumerate(trans):
        groups.setdefault(t[1], []).append(j)
    chunks = [(qk, idx[i:i + 64]) for qk, idx in groups.items() for i in range(0, len(idx), 64)]
    for _ in range(epochs):
        brain.opt.zero_grad()
        for qk, idx in chunks:
            obs = [trans[j][0] for j in idx]
            logp = torch.log_softmax(brain.logits(obs, qk), -1)
            a = torch.tensor([trans[j][2] for j in idx])
            ratio = torch.exp(logp[torch.arange(len(idx)), a] - torch.tensor([trans[j][3] for j in idx], dtype=torch.float32))
            A = torch.tensor(adv[idx], dtype=torch.float32)
            loss = -torch.min(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).sum()
            loss = loss + ent * (logp.exp() * logp).sum()
            if kl:
                lq = torch.log_softmax(brain.prior(obs, qk), -1)
                loss = loss + kl * (logp.exp() * (logp - lq)).sum()
            (loss / len(trans)).backward()
        torch.nn.utils.clip_grad_norm_(brain.params(), 1.0)
        brain.opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='kuhn'); ap.add_argument('--brain', default='adapter')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--updates', type=int, default=150); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--eval_every', type=int, default=10); ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--kl', type=float, default=0.0); ap.add_argument('--gate', default='auto')
    ap.add_argument('--rank', type=int, default=16); ap.add_argument('--script_frac', type=float, default=0.0)
    ap.add_argument('--contrast', type=float, default=1.0); ap.add_argument('--ent', type=float, default=0.01)
    ap.add_argument('--magnet', type=int, default=0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    tag = f'{a.env}_{a.brain}_pool{a.pool}_s{a.seed}{a.tag}'
    out = os.path.join(RES, tag + '.json')
    if os.path.exists(out):
        print('skip', tag); return
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    bank = FeatureBank(os.path.join(HERE, f'bank_{a.env}.pt'))
    game, game4 = K.Kuhn(3), K.Kuhn(4)
    train = {'vega0': None, 'engine': 'engine', 'adapter': 'adapter', 'table': 'table'}[a.brain]
    lr = a.lr or {'engine': 1e-4, 'adapter': 1e-3, 'table': 5e-2}.get(a.brain, 0)
    brain = TableKL(game.infosets, 2, lr) if train == 'table' else VegaBrain(bank, train, lr, gate=a.gate, rank=a.rank, contrast=a.contrast)
    updates = 0 if train is None else a.updates
    V, pool, hist = {}, [], []
    t0 = time.time()
    for u in range(updates + 1):
        if u % a.eval_every == 0 or u == updates:
            ev = {'update': u, 'secs': round(time.time() - t0, 1)}
            if a.env == 'kuhn':
                s = T.kuhn_strategy(game, brain)
                ev.update(exploitability=game.exploitability(s), strategy={k: round(v, 3) for k, v in s.items()})
                if brain.kind == 'text':
                    s4 = T.kuhn_strategy(game4, brain)
                    ev.update(exploitability_4card=game4.exploitability(s4), strategy_4card={k: round(v, 3) for k, v in s4.items()})
            else:
                ev['hider_train'], ev['seeker_train'] = T.grid_eval(brain, (7, 0.18), 32, 777)
                ev['hider_test'], ev['seeker_test'] = T.grid_eval(brain, (9, 0.25), 32, 888)
            hist.append(ev)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items() if not k.startswith('strategy')}), flush=True)
        if u == updates:
            break
        brain.set_opponent(pool[rng.integers(len(pool) + 1) - 1] if (a.pool and pool and rng.random() < 0.75) else None)
        if a.env == 'kuhn':
            trans, _ = T.kuhn_batch(game, brain, a.batch, rng)
        else:
            opp = 'script' if rng.random() < a.script_frac else 'self'
            trans, _, _ = T.grid_batch(brain, a.batch, rng, (7, 0.18), opponent=opp)
        ppo_update(brain, trans, V, kl=a.kl, ent=a.ent)
        if a.magnet and u % a.magnet == a.magnet - 1:
            brain.set_magnet()
        if a.pool and u % 10 == 9:
            pool.append(brain.snapshot()); pool = pool[-4:]
    json.dump(dict(args=vars(a), hist=hist, secs=time.time() - t0), open(out, 'w'), indent=1)
    print('done', tag, round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
