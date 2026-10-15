"""Can each parameterisation fit a card-dependent policy at all? Supervised CE to a Nash-like target."""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'laya_policy'))   # lpol (the game code)
from lpol import kuhn as K
from vpol.vega import FeatureBank, VegaPolicy
from extract import QUESTIONS
import paths as P
torch.manual_seed(0)
bank = FeatureBank(P.bank('kuhn')); g = K.Kuhn(3)
# a Nash equilibrium of Kuhn (alpha=0): P(bet/call)
tgt = {'J':0,'Q':0,'K':0,'Jc':1/3,'Qc':0,'Kc':1,'Jb':0,'Qb':1/3,'Kb':1,'Jcb':0,'Qcb':1/3,'Kcb':1}
data = {qk: [(g.describe(s[0], s[1:]), tgt[s]) for s in g.infosets if g.facing(s[1:]) == (qk == 'call')] for qk in ('bet','call')}
def run(mode, lr, steps=300):
    if mode == 'linear':
        heads = {qk: torch.nn.Linear(next(iter(bank.d.values()))['feats'].numel(), 2) for qk in data}
        params = [p for h in heads.values() for p in h.parameters()]
        def lg(st, qk): return heads[qk](torch.stack([bank.d[(s,qk)]['feats'].flatten() for s in st]))
    else:
        P = VegaPolicy(bank, QUESTIONS, train=mode, lr=lr, gate='on' if mode=='adapter' else 'auto'); params = P.params
        lg = P.logits
    opt = torch.optim.Adam(params, lr=lr)
    for t in range(steps + 1):
        opt.zero_grad(); L = 0; err = []
        for qk, d in data.items():
            st = [s for s, _ in d]; y = torch.tensor([[1-p, p] for _, p in d], dtype=torch.float32)
            lp = torch.log_softmax(lg(st, qk), -1); L = L - (y * lp).sum(-1).mean()
            err += list((lp.exp()[:,1].detach() - y[:,1]).abs().numpy())
        L.backward(); opt.step()
        if t % 100 == 0: print(f'{mode:8s} lr {lr:g} step {t:4d} loss {L.item():.4f} mean|p-target| {np.mean(err):.3f}', flush=True)
for mode, lr in [('linear',1e-3), ('engine',1e-4), ('engine',1e-3), ('adapter',1e-3), ('adapter',1e-2)]:
    run(mode, lr)
