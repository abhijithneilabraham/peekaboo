"""Train a policy by self-play PPO in Kuhn poker or text hide-and-seek.

    python train.py --env kuhn --brain laya  --seed 0
    python train.py --env grid --brain mlp   --seed 0 --pool 0
brain: laya (Laya's decision head as the policy) | mlp (from scratch) | table (Kuhn only) | laya0 (Laya, no training)
pool:  1 = play against a pool of past selves (league); 0 = always against the current self (naive self-play)
"""
import argparse, copy, json, os, time
import numpy as np
import torch

from lpol import kuhn as K, grid_hs as G

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)


# ───────────────────────────────────────── brains ────────────────────────────────────────────────
class LayaBrain:
    kind = 'text'

    def __init__(self, questions, lr, train=True):
        from lpol.policies import LayaPolicy
        self.P = LayaPolicy(lr=lr, train_head=train)
        self.Q = {k: self.P.question(i, o) for k, (i, o) in questions.items()}
        self.opp = self.P.make_opponent_head()
        self.opt = self.P.opt

    def logits(self, obs, qk, opp=False):
        return self.P.logits(obs, self.Q[qk], qk, self.opp if opp else None)

    def snapshot(self):
        return self.P.head_state()

    def set_opponent(self, snap):
        self.P.load_into(self.opp, snap if snap is not None else self.P.head_state())

    def params(self):
        return self.P.params


class MLPBrain:
    kind = 'feat'

    def __init__(self, sizes, lr, seed):
        torch.manual_seed(seed)
        mk = lambda n_in, n_act: torch.nn.Sequential(torch.nn.Linear(n_in, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64),
                                                     torch.nn.Tanh(), torch.nn.Linear(64, n_act))
        self.nets = {k: mk(*v) for k, v in sizes.items()}
        self.oppnets = copy.deepcopy(self.nets)
        self.opt = torch.optim.Adam([p for n in self.nets.values() for p in n.parameters()], lr=lr)

    def logits(self, obs, qk, opp=False):
        return (self.oppnets if opp else self.nets)[qk](torch.as_tensor(np.asarray(obs), dtype=torch.float32))

    def snapshot(self):
        return copy.deepcopy({k: n.state_dict() for k, n in self.nets.items()})

    def set_opponent(self, snap):
        for k, n in self.oppnets.items():
            n.load_state_dict(snap[k] if snap is not None else self.nets[k].state_dict())

    def params(self):
        return [p for n in self.nets.values() for p in n.parameters()]


class TableBrain:
    """One free logit vector per information set: the most flexible tabular baseline (Kuhn only)."""
    kind = 'key'

    def __init__(self, keys, n_act, lr):
        self.t = {k: torch.zeros(n_act, requires_grad=True) for k in keys}
        self.o = {k: v.detach().clone() for k, v in self.t.items()}
        self.opt = torch.optim.Adam(list(self.t.values()), lr=lr)

    def logits(self, obs, qk, opp=False):
        src = self.o if opp else self.t
        return torch.stack([src[o] for o in obs])

    def snapshot(self):
        return {k: v.detach().clone() for k, v in self.t.items()}

    def set_opponent(self, snap):
        self.o = {k: v.detach().clone() for k, v in (snap or self.snapshot()).items()}

    def params(self):
        return list(self.t.values())


def sample(brain, obs, qk, rng, opp=False):
    with torch.no_grad():
        p = torch.softmax(brain.logits(obs, qk, opp), -1).numpy()
    a = np.array([rng.choice(len(pi), p=pi / pi.sum()) for pi in p])
    return a, np.log(p[np.arange(len(a)), a] + 1e-12), p


# ───────────────────────────────────────── kuhn ──────────────────────────────────────────────────
def kuhn_obs(game, brain, card, h):
    return {'text': game.describe(card, h), 'feat': game.features(card, h), 'key': card + h}[brain.kind]


def kuhn_batch(game, brain, B, rng):
    """B hands; the learner sits in a random seat, the opponent is brain's opponent head."""
    deals = [game.deals()[i] for i in rng.integers(0, len(game.deals()), B)]
    seat = rng.integers(0, 2, B)
    hist = [''] * B
    trans = []      # (hand index, obs, qk, action, logp)
    while True:
        live = [i for i in range(B) if game.payoff(deals[i], hist[i]) is None]
        if not live:
            break
        for is_learner in (True, False):
            idx = [i for i in live if game.payoff(deals[i], hist[i]) is None
                   and ((len(hist[i]) % 2) == seat[i]) == is_learner]
            moves = {}
            for qk in ('bet', 'call'):          # decide every move first, then apply them
                sub = [i for i in idx if game.facing(hist[i]) == (qk == 'call')]
                if not sub:
                    continue
                obs = [kuhn_obs(game, brain, deals[i][len(hist[i]) % 2], hist[i]) for i in sub]
                a, lp, _ = sample(brain, obs, qk, rng, opp=not is_learner)
                for i, ai, li, o in zip(sub, a, lp, obs):
                    if is_learner:
                        trans.append([i, o, qk, ai, li])
                    moves[i] = game.actions(hist[i])[ai]
            for i, mv in moves.items():
                hist[i] = game.nxt(hist[i], mv)
    ret = [game.payoff(deals[i], hist[i]) * (1 if seat[i] == 0 else -1) for i in range(B)]
    return [(o, qk, a, lp, ret[i], o if brain.kind != 'feat' else tuple(o)) for i, o, qk, a, lp in trans], float(np.mean(ret))


def kuhn_strategy(game, brain):
    strat = {}
    for qk in ('bet', 'call'):
        sets = [s for s in game.infosets if game.facing(s[1:]) == (qk == 'call')]
        obs = [kuhn_obs(game, brain, s[0], s[1:]) for s in sets]
        with torch.no_grad():
            p = torch.softmax(brain.logits(obs, qk), -1).numpy()
        strat.update({s: float(pi[1]) for s, pi in zip(sets, p)})
    return strat


# ───────────────────────────────────────── grid ──────────────────────────────────────────────────
def grid_obs(env, brain, who):
    return env.describe(who) if brain.kind == 'text' else env.features(who)


def grid_batch(brain, B, rng, cfg, opponent='self', learner_roles=None, gamma=0.97):
    """B parallel episodes. opponent: 'self' (brain's opponent head) or 'script' (chase/flee)."""
    envs = [G.GridHS(size=cfg[0], wall_frac=cfg[1], rng=np.random.default_rng(rng.integers(1 << 30))).reset() for _ in range(B)]
    roles = learner_roles if learner_roles is not None else ['hider' if rng.random() < .5 else 'seeker' for _ in range(B)]
    steps = [[] for _ in range(B)]
    total = np.zeros(B)
    done = False
    while not done:
        acts = [{} for _ in range(B)]
        for who in ('hider', 'seeker'):
            for is_learner in (True, False):
                idx = [i for i in range(B) if (roles[i] == who) == is_learner]
                if not idx:
                    continue
                if not is_learner and opponent == 'script':
                    for i in idx:
                        acts[i][who] = G.chase(envs[i]) if who == 'seeker' else G.flee(envs[i])
                    continue
                obs = [grid_obs(envs[i], brain, who) for i in idx]
                a, lp, _ = sample(brain, obs, who, rng, opp=not is_learner)
                for i, ai, li, o in zip(idx, a, lp, obs):
                    acts[i][who] = G.ACTS[ai]
                    if is_learner:
                        steps[i].append([o, who, ai, li, 0.0])
        for i, e in enumerate(envs):
            r, done = e.step(acts[i])
            steps[i][-1][4] = r[roles[i]]
            total[i] += r[roles[i]]
    trans = []
    for st in steps:
        G_ = 0.0
        for o, who, a, lp, r in reversed(st):
            G_ = r + gamma * G_
            trans.append((o, who, a, lp, G_, o if isinstance(o, str) else tuple(np.round(o, 2))))
    return trans, roles, total


def grid_eval(brain, cfg, n, seed):
    rng = np.random.default_rng(seed)
    _, _, th = grid_batch(brain, n, rng, cfg, opponent='script', learner_roles=['hider'] * n)
    _, _, ts = grid_batch(brain, n, rng, cfg, opponent='script', learner_roles=['seeker'] * n)
    return float(th.mean()), float(ts.mean())


# ───────────────────────────────────────── PPO ───────────────────────────────────────────────────
def ppo_update(brain, trans, V, epochs=2, clip=0.2, ent=0.01, alpha=0.1):
    keys = [t[5] for t in trans]
    ret = np.array([t[4] for t in trans], float)
    base = np.array([V.get(k, 0.0) for k in keys])
    adv = ret - base
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for k, r in zip(keys, ret):
        V[k] = V.get(k, 0.0) + alpha * (r - V.get(k, 0.0))
    groups = {}
    for j, t in enumerate(trans):
        groups.setdefault(t[1], []).append(j)
    # gradients accumulate over chunks so a big batch never holds every activation at once
    chunks = [(qk, idx[i:i + 32]) for qk, idx in groups.items() for i in range(0, len(idx), 32)]
    for _ in range(epochs):
        brain.opt.zero_grad()
        for qk, idx in chunks:
            lg = brain.logits([trans[j][0] for j in idx], qk)
            logp = torch.log_softmax(lg, -1)
            a = torch.tensor([trans[j][2] for j in idx])
            lp = logp[torch.arange(len(idx)), a]
            ratio = torch.exp(lp - torch.tensor([trans[j][3] for j in idx], dtype=torch.float32))
            A = torch.tensor(adv[idx], dtype=torch.float32)
            pg = -torch.min(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).sum()
            H = -(logp.exp() * logp).sum(-1).sum()
            ((pg - ent * H) / len(trans)).backward()
        torch.nn.utils.clip_grad_norm_(brain.params(), 1.0)
        brain.opt.step()


# ───────────────────────────────────────── main ──────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='kuhn'); ap.add_argument('--brain', default='laya')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--updates', type=int, default=120); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--eval_every', type=int, default=10); ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--tag', default=''); ap.add_argument('--script_frac', type=float, default=0.0); ap.add_argument('--dense_until', type=int, default=0)
    a = ap.parse_args()
    tag = f'{a.env}_{a.brain}_pool{a.pool}_s{a.seed}{a.tag}'
    out = os.path.join(RES, tag + '.json')
    if os.path.exists(out):
        print('skip', tag); return
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    if a.env == 'kuhn':
        game, game4 = K.Kuhn(3), K.Kuhn(4)
        Qs = {'bet': K.QUESTION[False], 'call': K.QUESTION[True]}
        sizes = {'bet': (7, 2), 'call': (7, 2)}
    else:
        Qs = {'hider': G.QUESTION['hider'], 'seeker': G.QUESTION['seeker']}
        sizes = {'hider': (19, 5), 'seeker': (19, 5)}
    if a.brain in ('laya', 'laya0'):
        brain = LayaBrain(Qs, lr=a.lr or 3e-5, train=a.brain == 'laya')
    elif a.brain == 'mlp':
        brain = MLPBrain(sizes, lr=a.lr or 3e-3, seed=a.seed)
    else:
        brain = TableBrain(game.infosets, 2, lr=a.lr or 5e-2)
    updates = 0 if a.brain == 'laya0' else a.updates
    V, pool, hist = {}, [], []
    t0 = time.time()
    for u in range(updates + 1):
        if u % a.eval_every == 0 or u == updates or (u < a.dense_until and u % 10 == 0):
            ev = {'update': u, 'secs': round(time.time() - t0, 1)}
            if a.env == 'kuhn':
                s = kuhn_strategy(game, brain)
                ev['exploitability'] = game.exploitability(s)
                ev['strategy'] = {k: round(v, 3) for k, v in s.items()}
                if brain.kind == 'text':            # language transfer: a deck it never trained on
                    s4 = kuhn_strategy(game4, brain)
                    ev['exploitability_4card'] = game4.exploitability(s4)
                    ev['strategy_4card'] = {k: round(v, 3) for k, v in s4.items()}
            else:
                ev['hider_train'], ev['seeker_train'] = grid_eval(brain, (7, 0.18), 32, 777)
                ev['hider_test'], ev['seeker_test'] = grid_eval(brain, (9, 0.25), 32, 888)
            if brain.kind == 'text':
                ev['encoded_states'] = brain.P.encoded
            hist.append(ev)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items() if not k.startswith('strategy')}), flush=True)
        if u == updates:
            break
        brain.set_opponent(pool[rng.integers(len(pool) + 1) - 1] if (a.pool and pool and rng.random() < 0.75) else None)
        if a.env == 'kuhn':
            trans, _ = kuhn_batch(game, brain, a.batch, rng)
        else:
            opp = 'script' if rng.random() < a.script_frac else 'self'   # scripted sparring partners mixed into self-play
            trans, _, _ = grid_batch(brain, a.batch, rng, (7, 0.18), opponent=opp)
        ppo_update(brain, trans, V)
        if a.pool and u % 10 == 9:
            pool.append(brain.snapshot()); pool = pool[-4:]
    json.dump(dict(args=vars(a), hist=hist, secs=time.time() - t0), open(out, 'w'), indent=1)
    print('done', tag, round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
