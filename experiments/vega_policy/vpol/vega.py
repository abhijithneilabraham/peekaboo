"""Vega as an RL policy.

Two stages, so the 0.8B backbone never has to sit in memory during training:
  1. FeatureBank.extract(...)   run the frozen backbone once per distinct (state text, question), save pooled features
  2. VegaPolicy(...)            load only the 14M-parameter engine and act / learn on the cached features

Every decision is a Vega `choice` question; the options are the legal moves. The policy is Vega's own readout:
pi(a|s) = softmax(-E_a(s) / tau(s)), with tau from the shipped thermal calibration.
"""
import copy, json, os
import numpy as np
import torch

from vegaml import hub
from vegaml._common import (QTYPES, VegaEngine, attach_adapter, build_prompt, thermal_features)

QWEN = os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17')


def model_dir():
    return hub.fetch()


def question(instructions, options):
    """options: list of (key, description). Vega renders each as 'key - description'."""
    return {'t': 'choice', 'ins': instructions, 'crit': dict(options)}


class FeatureBank:
    """(state text, question key) -> pooled frozen features. Saved as one .pt file per environment."""

    def __init__(self, path):
        self.path = path
        self.d = torch.load(path) if os.path.exists(path) else {}

    def save(self):
        torch.save(self.d, self.path + '.tmp'); os.replace(self.path + '.tmp', self.path)

    def missing(self, pairs):
        return [p for p in pairs if p not in self.d]

    def extract(self, pairs, questions, batch=8, log=print):
        """pairs: list of (state_text, qkey). Loads the backbone, encodes what is missing, frees it again."""
        todo = self.missing(pairs)
        if not todo:
            return
        from vegaml._common import Perception
        mdir = model_dir()
        cfg = json.load(open(os.path.join(mdir, 'vega_config.json')))
        perc = Perception(QWEN, torch.device('cpu'), cfg['feature_layers'], cfg['option_layers'])
        C = cfg.get('memory_chunks', 0)
        import time
        t0 = time.time()
        for s in range(0, len(todo), batch):
            chunk = todo[s:s + batch]
            items = []
            for text, qk in chunk:
                ids, spans = build_prompt(perc.tok, text, questions[qk], cfg['max_len'], template=0,
                                          opt_max_tokens=cfg['opt_max_tokens'], query_first_max_state=cfg.get('query_first_max_state', 0))
                items.append({'ids': np.asarray(ids, dtype=np.int32), 'spans': spans})
            if C:
                feats, opt_flat, chunks, cmask = perc.pooled(items, chunks=C, prefix_min=cfg.get('prefix_min_tokens', 4096),
                                                             chunk_min_tokens=cfg.get('memory_min_tokens', 2048),
                                                             chunk_tokens=cfg.get('memory_chunk_tokens', 128))
            else:
                (feats, opt_flat), chunks, cmask = perc.pooled(items), None, None
            o = 0
            for b, (key, it) in enumerate(zip(chunk, items)):
                k = len(it['spans']['options'])
                self.d[key] = {'feats': feats[b].clone().float(), 'opts': opt_flat[o:o + k].clone().float(),
                               'chunks': None if chunks is None else chunks[b].clone().float(),
                               'cmask': None if cmask is None else cmask[b].clone()}
                o += k
            if (s // batch) % 10 == 0:
                log(f'encoded {s + len(chunk)}/{len(todo)}  {time.time() - t0:.0f}s')
                self.save()
        self.save()
        del perc
        import gc; gc.collect()


class VegaPolicy:
    """The Vega engine on cached features.

    train: 'adapter' (fresh gated rank-r LoRA, base frozen; the gate decides per question how much of it to use),
           'engine'  (all 14M engine parameters, the analogue of training Laya's head), or None (zero-shot).
    """

    def __init__(self, bank, questions, train='adapter', lr=1e-3, rank=16, alpha=16.0, gate='auto', contrast=1.0):
        from safetensors.torch import load_file
        mdir = model_dir()
        self.cfg = json.load(open(os.path.join(mdir, 'vega_config.json')))
        eng = VegaEngine(self.cfg['d_model'], len(self.cfg['feature_layers']), len(self.cfg['option_layers']), self.cfg)
        eng.load_state_dict(load_file(os.path.join(mdir, 'engine.safetensors')))
        eng.eval()
        for p in eng.parameters():
            p.requires_grad_(False)
        self.train_mode = train
        if train == 'adapter':
            j = attach_adapter(eng, rank=rank, alpha=alpha)
            eng.adapter_mode = gate          # 'auto' = learned gate, 'on' = always on
            self.params = [p for n, p in eng.named_parameters() if 'lora_' in n or n.startswith(f'gate_nets.{j}')]
        elif train == 'engine':
            self.params = list(eng.parameters())
        else:
            self.params = []
        for p in self.params:
            p.requires_grad_(True)
        self.eng = eng
        self.bank, self.Q = bank, questions
        self.theta = np.asarray(self.cfg['calibration']['theta'], dtype=np.float64)
        self.opt = torch.optim.Adam(self.params, lr=lr) if self.params else None
        self.encoded = len(bank.d)
        # contrast > 1 exaggerates how each state differs from the average state of its question (zero parameters)
        self.contrast = contrast
        self.mu = {}
        for (txt, qk), r in bank.d.items():
            self.mu.setdefault(qk, []).append(r['feats'])
        self.mu = {qk: torch.stack(v).mean(0) for qk, v in self.mu.items()}

    # batching cached features for one question key
    def _batch(self, states, qk):
        rows = [self.bank.d[(s, qk)] for s in states]
        feats = torch.stack([r['feats'] for r in rows])
        if self.contrast != 1.0:
            feats = self.mu[qk] + self.contrast * (feats - self.mu[qk])
        K = max(r['opts'].shape[0] for r in rows)
        opts = torch.zeros((len(rows), K) + tuple(rows[0]['opts'].shape[1:]))
        mask = torch.zeros(len(rows), K, dtype=torch.bool)
        for b, r in enumerate(rows):
            opts[b, :r['opts'].shape[0]] = r['opts']; mask[b, :r['opts'].shape[0]] = True
        chunks = cmask = None
        if rows[0]['chunks'] is not None:
            chunks = torch.stack([r['chunks'] for r in rows]); cmask = torch.stack([r['cmask'] for r in rows])
        qtype = torch.full((len(rows),), QTYPES['choice'], dtype=torch.long)
        return feats, opts, mask, qtype, chunks, cmask

    def forward(self, states, qk, eng=None, mode=None):
        """Returns (logits / tau, raw engine output). tau is the shipped thermal calibration, held fixed (no gradient)."""
        eng = eng or self.eng
        old = eng.adapter_mode
        if mode is not None:
            eng.adapter_mode = mode
        feats, opts, mask, qtype, chunks, cmask = self._batch(states, qk)
        out = eng(feats, opts, mask, qtype, return_traj=False, chunks=chunks, chunk_mask=cmask)
        eng.adapter_mode = old
        with torch.no_grad():
            phys = thermal_features(out, mask, eng.max_steps).double().numpy()
            th = self.theta[qtype.numpy()]
            tau = np.exp(np.clip(th[:, 0] + (th[:, 1:] * phys).sum(-1), np.log(0.2), np.log(20.0)))
        logits = out['logits'] / torch.as_tensor(tau, dtype=torch.float32).unsqueeze(-1)
        return logits.masked_fill(~mask, -1e4), out

    def logits(self, states, qk, opp=None):
        return self.forward(states, qk, eng=opp)[0]

    def prior_logits(self, states, qk):
        """The untouched base engine (adapter switched off): the zero-shot policy to stay close to."""
        with torch.no_grad():
            return self.forward(states, qk, mode='off')[0] if self.train_mode == 'adapter' else None

    def state(self):
        return {n: p.detach().clone() for n, p in self.eng.named_parameters() if p.requires_grad}

    def make_opponent(self):
        o = copy.deepcopy(self.eng)
        for p in o.parameters():
            p.requires_grad_(False)
        return o

    @staticmethod
    def load_into(eng, state):
        sd = dict(eng.named_parameters())
        with torch.no_grad():
            for n, v in state.items():
                sd[n].copy_(v)
