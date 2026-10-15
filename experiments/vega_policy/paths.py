"""Which Vega checkpoint a script works on, and where its caches and results live.

    VEGA_SIZE    0.8b or 4b. reproduce.py sets it for every step; each size has its own caches and results.
    VEGA_OUT     output root (default: this folder, which holds the paper's own results). reproduce.py points it
                 at a fresh folder, so a reproduction never reads or skips on the shipped results.
    VEGA_DEVICE  torch device (default: cuda if available, else cpu).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = os.environ.get('VEGA_SIZE', '0.8b').lower().replace('08b', '0.8b')
SFX = '' if SIZE == '0.8b' else '_' + SIZE.replace('.', '')
LABEL = 'Vega ' + SIZE.upper()
ROOT = os.path.abspath(os.environ.get('VEGA_OUT') or HERE)
HARNESS = os.path.join(HERE, 'harness')


def device():
    import torch
    return os.environ.get('VEGA_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')


def bank(env):
    """Feature cache of one suite or game for this size: decisionbench, public, jevbench, kuhn."""
    return os.path.join(ROOT, f'bank_{env}{SFX}.pt')


def items(name):
    """Row lists are the same for every size (the rows do not depend on the model)."""
    return os.path.join(ROOT, name)


RESULTS = os.path.join(ROOT, 'results' + SFX)
COMPARE = os.path.join(RESULTS, 'compare')
FIGURES = os.path.join(ROOT, 'figures') if os.environ.get('VEGA_OUT') else os.path.join(HERE, '..', '..', 'reports', 'figures')


def harness():
    """Make VegaML's vendored benchmark modules importable (they import each other by bare name)."""
    import sys
    if HARNESS not in sys.path:
        sys.path.insert(0, HARNESS)
