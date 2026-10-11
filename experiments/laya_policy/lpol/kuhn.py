"""Kuhn poker with an N-card deck (3 = classic Kuhn: Jack, Queen, King; 4 adds the Ace).

Each player antes 1 and gets one card; one betting round (check/bet, then fold/call).
The game is small enough to compute exploitability exactly: 0 means unexploitable (Nash).
A strategy is a dict  infoset -> P(aggressive action), aggressive = bet or call.
"""
import itertools

import numpy as np

ALL = ['J', 'Q', 'K', 'A']
WORD = {'J': 'Jack', 'Q': 'Queen', 'K': 'King', 'A': 'Ace'}
HISTS = ['', 'c', 'b', 'cb']


class Kuhn:
    def __init__(self, n_cards=3):
        self.cards = ALL[:n_cards]
        self.infosets = [c + h for c in self.cards for h in HISTS]
        self.p1 = [c + h for c in self.cards for h in ['', 'cb']]
        self.p2 = [c + h for c in self.cards for h in ['c', 'b']]

    def rank_words(self, card):
        i, n = self.cards.index(card), len(self.cards)
        pos = 'the highest card' if i == n - 1 else 'the lowest card' if i == 0 else \
              ('the middle card' if n == 3 else f'the {"second" if i == 1 else "third"} lowest card')
        return f'the {WORD[card]}, {pos}'

    def describe(self, card, h):
        deck = ', '.join(WORD[c] for c in self.cards)
        seat = 'first' if len(h) % 2 == 0 else 'second'
        story = {'': 'Nobody has acted yet.', 'c': 'Your opponent checked.', 'b': 'Your opponent bet 1 chip.',
                 'cb': 'You checked, then your opponent bet 1 chip.'}[h]
        return (f'Poker, one betting round. The deck has {len(self.cards)} cards, ranked low to high: {deck}. '
                f'Each player paid 1 chip and got one card. You hold {self.rank_words(card)}. You act {seat}. {story}')

    def features(self, card, h):           # numeric baseline input: only defined for the training deck
        f = np.zeros(len(self.cards) + 4)
        f[self.cards.index(card)] = 1
        f[len(self.cards) + HISTS.index(h)] = 1
        return f

    # ── rules ──
    @staticmethod
    def facing(h):
        return h.endswith('b')

    @staticmethod
    def actions(h):
        return ('fold', 'call') if h.endswith('b') else ('check', 'bet')

    @staticmethod
    def nxt(h, a):
        return h + {'check': 'c', 'bet': 'b', 'call': 'c', 'fold': 'f'}[a]

    def payoff(self, cards, h):
        win = 1 if self.cards.index(cards[0]) > self.cards.index(cards[1]) else -1
        return {'cc': win, 'bc': 2 * win, 'cbc': 2 * win, 'bf': 1, 'cbf': -1}.get(h)

    def deals(self):
        return list(itertools.permutations(self.cards, 2))

    # ── exact evaluation ──
    def value(self, s1, s2):
        return sum(self._v(c, '', s1, s2) for c in self.deals()) / len(self.deals())

    def _v(self, cards, h, s1, s2):
        p = self.payoff(cards, h)
        if p is not None:
            return p
        pl = len(h) % 2
        s = (s1 if pl == 0 else s2)[cards[pl] + h]
        a0, a1 = self.actions(h)
        return (1 - s) * self._v(cards, self.nxt(h, a0), s1, s2) + s * self._v(cards, self.nxt(h, a1), s1, s2)

    def _br(self, s, seat):
        sets = self.p1 if seat == 0 else self.p2
        best = -1e9
        for bits in itertools.product([0.0, 1.0], repeat=len(sets)):
            br = dict(zip(sets, bits))
            best = max(best, self.value(br, s) if seat == 0 else -self.value(s, br))
        return best

    def exploitability(self, s):
        """Mean of what a best responder wins in each seat; the game value cancels, Nash = 0."""
        return (self._br(s, 0) + self._br(s, 1)) / 2


QUESTION = {
    False: ('Choose your poker action. Win as many chips as possible over many hands.',
            [('check', 'check: put in no more chips'), ('bet', 'bet: put in 1 more chip')]),
    True: ('Choose your poker action. Win as many chips as possible over many hands.',
           [('fold', 'fold: give up the pot'), ('call', 'call: match the 1 chip bet')]),
}
