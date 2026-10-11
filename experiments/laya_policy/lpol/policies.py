"""Policies an RL loop can train.

LayaPolicy turns every decision into a Laya `choice` question: the situation is the state text,
the legal actions are the options, and Laya's option probabilities are the policy. The 395M
ModernBERT encoder stays frozen and its output is cached per unique situation; the 26.5M
decision head (2 transformer layers + scorer) is what RL trains.

MLPPolicy is the hide-and-seek-era baseline: a small network trained from scratch on a
numeric encoding of the same observation, no language at all.
"""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault('USE_TF', '0')
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))


class LayaPolicy:
    def __init__(self, lr=3e-5, max_cache=3000, train_head=True):
        import laya
        from laya.common import QTYPES
        self.ag = laya.load('convaiinnovations/laya', device='cpu')
        self.m = self.ag.model.eval()
        for p in self.m.encoder.parameters():
            p.requires_grad_(False)
        self.params = [p for n, p in self.m.named_parameters() if not n.startswith('encoder')]
        for p in self.params:
            p.requires_grad_(train_head)
        self.T = float(self.ag.temperature[QTYPES['choice']])     # Laya's own calibrated choice temperature
        self.opt = torch.optim.Adam(self.params, lr=lr) if train_head else None
        self.cache = OrderedDict()
        self.max_cache = max_cache
        self.encoded = 0

    @staticmethod
    def question(instructions, options):
        """options: list of (name, description). Order is the action index order."""
        return {'act': {'type': 'choice', 'instructions': instructions, 'criteria': dict(options)}}

    def _capture(self, states, q):
        """Have Laya build its own input batch, without running its forward pass."""
        box = {}

        def grab(b):
            box['b'] = b
            B, M = b['marker_pos'].shape
            return torch.zeros(B, M), torch.zeros(B, 2)
        orig, self.ag._infer = self.ag._infer, grab
        try:
            self.ag.predict_batch(states, q)
        finally:
            self.ag._infer = orig
        return box['b']

    def _ensure(self, states, q, qkey):
        miss = [s for s in dict.fromkeys(states) if (s, qkey) not in self.cache]
        for i in range(0, len(miss), 16):
            chunk = miss[i:i + 16]
            b = self._capture(chunk, q)
            with torch.no_grad():
                h = self.m.encoder(input_ids=b['input_ids'], attention_mask=b['attention_mask']).last_hidden_state
            for r, s in enumerate(chunk):
                L = int(b['attention_mask'][r].sum())
                self.cache[(s, qkey)] = (h[r, :L].to(torch.float16), b['marker_pos'][r], b['marker_mask'][r], b['qtype'][r])
                self.encoded += 1
        for s in states:
            self.cache.move_to_end((s, qkey))
        while len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)

    def head_modules(self):
        return {'head': self.m.head, 'type_emb': self.m.type_emb, 'scorer': self.m.scorer}

    def head_state(self):
        """A frozen copy of the trainable head (fp16, ~53 MB) for the opponent pool."""
        return {k: v.detach().to(torch.float16).clone() for k, v in self.m.state_dict().items() if not k.startswith('encoder')}

    def make_opponent_head(self):
        import copy
        return {k: copy.deepcopy(v).requires_grad_(False) for k, v in self.head_modules().items()}

    @staticmethod
    def load_into(mods, state):
        for name, mod in mods.items():
            sub = {k[len(name) + 1:]: v.float() for k, v in state.items() if k.startswith(name + '.')}
            mod.load_state_dict(sub)

    def logits(self, states, q, qkey, mods=None):
        """Option logits (already divided by Laya's temperature), with grad through the head."""
        self._ensure(states, q, qkey)
        rows = [self.cache[(s, qkey)] for s in states]
        L = max(r[0].shape[0] for r in rows)
        d = rows[0][0].shape[1]
        h = torch.zeros(len(rows), L, d)
        mask = torch.zeros(len(rows), L, dtype=torch.bool)
        for i, r in enumerate(rows):
            h[i, :r[0].shape[0]] = r[0].float(); mask[i, :r[0].shape[0]] = True
        mpos = torch.stack([r[1] for r in rows]); mmask = torch.stack([r[2] for r in rows]); qt = torch.stack([r[3] for r in rows])
        m = mods or self.head_modules()
        h = h + m['type_emb'](qt)[:, None, :]
        for layer in m['head'].layers:
            h = layer(h, src_key_padding_mask=~mask)
        idx = mpos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        lg = m['scorer'](torch.gather(h, 1, idx)).squeeze(-1).float().masked_fill(~mmask, -1e4)
        return lg / self.T

    def probs(self, states, q, qkey, mods=None):
        with torch.no_grad():
            return torch.softmax(self.logits(states, q, qkey, mods), -1).numpy()


class MLPPolicy:
    """From-scratch baseline: one small network per decision point, numeric features in."""

    def __init__(self, n_in, n_act, lr=3e-3, hidden=64, seed=0):
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, n_act))
        self.params = list(self.net.parameters())
        self.opt = torch.optim.Adam(self.params, lr=lr)

    def logits(self, feats):
        return self.net(torch.as_tensor(np.asarray(feats), dtype=torch.float32))

    def probs(self, feats):
        with torch.no_grad():
            return torch.softmax(self.logits(feats), -1).numpy()
