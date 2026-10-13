"""Which Vega checkpoint the experiments use, and where its feature cache and results live.

    VEGA_SIZE    0.8b (default) or 4b. Each size gets its own feature cache and results folder, so the two never mix.
    VEGA_DEVICE  device for the backbone during feature extraction (default: cuda if available, else cpu).
                 Training and evaluation run the small engine on cached features, on the CPU.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = os.environ.get('VEGA_SIZE', '0.8b').lower().replace('08b', '0.8b')
SFX = '' if SIZE == '0.8b' else '_' + SIZE.replace('.', '')
LABEL = 'Vega ' + SIZE.upper()


def device():
    import torch
    return os.environ.get('VEGA_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')


def bank(env):
    """Feature cache for one environment: 'decisionbench' or 'kuhn'."""
    return os.path.join(HERE, f'bank_{env}{SFX}.pt')


RESULTS = os.path.join(HERE, 'results' + SFX)
COMPARE = os.path.join(RESULTS, 'compare')
FIGURES = os.path.join(HERE, '..', '..', 'reports', 'figures')
