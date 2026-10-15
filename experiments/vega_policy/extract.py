"""Encode every situation the experiments can produce, once, with the frozen Vega backbone.

Kuhn poker: 12 (3-card) + 16 (4-card) information sets.
Grid hide-and-seek: the egocentric description takes only finitely many forms
(role x direction x distance band x sight x blocked sides x phase = 1,536), so all of them are encoded up front
and training never needs the backbone again.
"""
import itertools, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'laya_policy'))   # lpol (the game code)
from lpol import kuhn as K, grid_hs as G
import paths as P
from vpol.vega import FeatureBank, question

HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS = {'bet': question(*K.QUESTION[False]), 'call': question(*K.QUESTION[True]),
             'hider': question(*G.QUESTION['hider']), 'seeker': question(*G.QUESTION['seeker'])}


def kuhn_pairs():
    out = []
    for n in (3, 4):
        g = K.Kuhn(n)
        for c in g.cards:
            for h in K.HISTS:
                out.append((g.describe(c, h), 'call' if g.facing(h) else 'bet'))
    return out


class _Fixed(G.GridHS):
    """A grid whose observation we set directly, to enumerate every possible description."""
    def __init__(self, ang, db, see, blocked, t):
        super().__init__(); self._r, self._see, self._bl, self.t = (ang, db, 0), see, blocked, t
        self.pos = {'hider': (0, 0), 'seeker': (0, 0)}
    def rel(self, who): return self._r
    def sees(self, a=None, b=None): return self._see
    def blocked(self, c, d): return d in self._bl


def grid_pairs():
    out = []
    sides = ('north', 'south', 'east', 'west')
    for who, ang, db, see, t in itertools.product(('hider', 'seeker'), range(8), range(3), (False, True), (0, 5)):
        for k in range(16):
            bl = {s for j, s in enumerate(sides) if k >> j & 1}
            out.append((_Fixed(ang, db, see, bl, t).describe(who), who))
    return out


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'kuhn'
    pairs = kuhn_pairs() if which == 'kuhn' else grid_pairs()
    if len(sys.argv) > 2:
        pairs = pairs[:int(sys.argv[2])]
    bank = FeatureBank(P.bank(which))
    print(which, len(pairs), 'situations,', len(bank.missing(pairs)), 'to encode', flush=True)
    bank.extract(pairs, QUESTIONS, batch=8, log=lambda m: print(m, flush=True))
    print('done', len(bank.d), flush=True)
