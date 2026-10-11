"""Text hide-and-seek on a grid.

Random walled maps; a hider and a seeker move one cell per turn (north/south/east/west/stay).
They "see" each other when they share a row or column with no wall between them. Like OpenAI's
game: the first PREP turns the seeker waits, then every turn the hider scores +1 while unseen
and -1 while seen (zero-sum; the seeker gets the negative). Observations are egocentric and
coarse, so the same description works on any map size.
"""
from collections import deque

import numpy as np

MOVES = {'north': (-1, 0), 'south': (1, 0), 'east': (0, 1), 'west': (0, -1), 'stay': (0, 0)}
ACTS = list(MOVES)
DIR8 = ['north', 'north-east', 'east', 'south-east', 'south', 'south-west', 'west', 'north-west']


class GridHS:
    def __init__(self, size=7, wall_frac=0.18, steps=16, prep=3, rng=None):
        self.n, self.wf, self.T, self.prep = size, wall_frac, steps, prep
        self.rng = rng or np.random.default_rng(0)

    # ── map ──
    def reset(self):
        r = self.rng
        while True:
            W = r.random((self.n, self.n)) < self.wf
            free = [tuple(c) for c in np.argwhere(~W)]
            if len(free) < self.n * 2:
                continue
            h, s = (free[i] for i in r.choice(len(free), 2, replace=False))
            if abs(h[0] - s[0]) + abs(h[1] - s[1]) >= 3 and self._reach(W, h, s):
                break
        self.W, self.pos, self.t = W, {'hider': h, 'seeker': s}, 0
        return self

    def _reach(self, W, a, b):
        q, seen = deque([a]), {a}
        while q:
            c = q.popleft()
            if c == b:
                return True
            for d in MOVES.values():
                nx = (c[0] + d[0], c[1] + d[1])
                if 0 <= nx[0] < self.n and 0 <= nx[1] < self.n and not W[nx] and nx not in seen:
                    seen.add(nx); q.append(nx)
        return False

    def blocked(self, c, d):
        nx = (c[0] + MOVES[d][0], c[1] + MOVES[d][1])
        return not (0 <= nx[0] < self.n and 0 <= nx[1] < self.n) or self.W[nx]

    def sees(self, a=None, b=None):
        a, b = a or self.pos['hider'], b or self.pos['seeker']
        if a[0] == b[0]:
            lo, hi = sorted((a[1], b[1]))
            return not self.W[a[0], lo + 1:hi].any()
        if a[1] == b[1]:
            lo, hi = sorted((a[0], b[0]))
            return not self.W[lo + 1:hi, a[1]].any()
        return False

    # ── dynamics ──
    def step(self, acts):
        for who in ('hider', 'seeker'):
            if who == 'seeker' and self.t < self.prep:
                continue
            if not self.blocked(self.pos[who], acts[who]):
                d = MOVES[acts[who]]
                self.pos[who] = (self.pos[who][0] + d[0], self.pos[who][1] + d[1])
        self.t += 1
        r = 0.0 if self.t <= self.prep else (-1.0 if self.sees() else 1.0) / (self.T - self.prep)
        return {'hider': r, 'seeker': -r}, self.t >= self.T

    # ── observations ──
    def rel(self, who):
        me, other = self.pos[who], self.pos['seeker' if who == 'hider' else 'hider']
        dy, dx = other[0] - me[0], other[1] - me[1]
        dist = abs(dy) + abs(dx)
        ang = int(np.round(np.arctan2(dx, -dy) / (np.pi / 4))) % 8
        db = 0 if dist <= 1 else 1 if dist <= 3 else 2
        return ang, db, dist

    def describe(self, who):
        other = 'seeker' if who == 'hider' else 'hider'
        ang, db, dist = self.rel(who)
        where = ['right next to you', 'close', 'far away'][db]
        see = 'You can see each other.' if self.sees() else 'You cannot see each other.'
        bl = [d for d in ('north', 'south', 'east', 'west') if self.blocked(self.pos[who], d)]
        op = [d for d in ('north', 'south', 'east', 'west') if d not in bl]
        walls = (f"Blocked: {', '.join(bl)}. " if bl else '') + (f"Open: {', '.join(op)}." if op else 'Every side is blocked.')
        phase = 'The seeker is still counting and cannot move yet.' if self.t < self.prep else 'The seeker is hunting.'
        return (f'Hide-and-seek on a grid with walls. You are the {who}. The {other} is {where}, to the {DIR8[ang]}. '
                f'{see} {walls} {phase} Sight works only along a straight row or column with no wall in between.')

    def features(self, who):
        ang, db, _ = self.rel(who)
        f = np.zeros(2 + 8 + 3 + 1 + 4 + 1)
        f[0 if who == 'hider' else 1] = 1
        f[2 + ang] = 1; f[10 + db] = 1; f[13] = self.sees()
        for i, d in enumerate(('north', 'south', 'east', 'west')):
            f[14 + i] = self.blocked(self.pos[who], d)
        f[18] = self.t < self.prep
        return f


QUESTION = {
    'hider': ('You are the hider. Choose your move. You score every turn the seeker cannot see you.',
              [(a, f'{a}: move one cell {a}' if a != 'stay' else 'stay: do not move') for a in ACTS]),
    'seeker': ('You are the seeker. Choose your move. You score every turn you can see the hider.',
               [(a, f'{a}: move one cell {a}' if a != 'stay' else 'stay: do not move') for a in ACTS]),
}


# ── scripted opponents for evaluation ──
def bfs_next(env, start, goal):
    q, prev = deque([start]), {start: None}
    while q:
        c = q.popleft()
        if c == goal:
            break
        for a, d in MOVES.items():
            if a == 'stay' or env.blocked(c, a):
                continue
            nx = (c[0] + d[0], c[1] + d[1])
            if nx not in prev:
                prev[nx] = (c, a); q.append(nx)
    c, first = goal, 'stay'
    while prev.get(c):
        c, first = prev[c]
    return first


def chase(env):
    """Seeker: walk the shortest path to the hider."""
    return bfs_next(env, env.pos['seeker'], env.pos['hider'])


def flee(env):
    """Hider: pick the move that breaks line of sight, then maximises path distance."""
    best, bestv = 'stay', -1e9
    for a in ACTS:
        if env.blocked(env.pos['hider'], a):
            continue
        d = MOVES[a]
        nh = (env.pos['hider'][0] + d[0], env.pos['hider'][1] + d[1])
        v = (0 if env.sees(nh, env.pos['seeker']) else 100) + abs(nh[0] - env.pos['seeker'][0]) + abs(nh[1] - env.pos['seeker'][1])
        if v > bestv:
            best, bestv = a, v
    return best
