"""Per-trainer hyperparameter search spaces.

Train mode samples configurations from these definitions; Claude mode adds
new entries here when registering a new trainer.

Convention:
- list value → discrete choice (e.g. [3, 6, 9])
- tuple (low, high) → continuous range; ints if both endpoints are ints,
  floats otherwise.
"""
from __future__ import annotations
import numpy as np


SEARCH_SPACES: dict[str, dict] = {
    'lightgbm': {
        'num_leaves':         [15, 31, 63, 127],
        'max_depth':          [4, 6, 8, -1],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 500, 1000],
        'min_child_samples':  [20, 50, 100, 200],
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.6, 1.0),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.0, 1.0),
        'pos_class_weight':   (1.0, 8.0),
    },
    'xgboost': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.5),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
    },
    # When Claude mode adds new trainers (LSTM, LoRA, RL, ...), it appends here.
}


def sample(trainer: str, rng: np.random.RandomState | None = None) -> dict:
    """Draw one configuration for a given trainer."""
    if trainer not in SEARCH_SPACES:
        raise ValueError(f'No search space defined for {trainer!r}. '
                         f'Add one in models/search_spaces.py.')
    if rng is None:
        rng = np.random
    space = SEARCH_SPACES[trainer]
    cfg = {}
    for key, val in space.items():
        if isinstance(val, list):
            cfg[key] = type(val[0])(rng.choice(val))
        elif isinstance(val, tuple) and len(val) == 2:
            lo, hi = val
            if isinstance(lo, int) and isinstance(hi, int):
                cfg[key] = int(rng.randint(lo, hi + 1))
            else:
                cfg[key] = float(rng.uniform(lo, hi))
        else:
            raise ValueError(f'Bad search space value for {key}: {val!r}')
    return cfg


def list_trainers() -> list[str]:
    return sorted(SEARCH_SPACES.keys())
