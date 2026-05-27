"""Archive of trainers retired from active selection.

Phase 3C (2026-05-27). Pruning rule (refined from the plan):
    n_iters >= 20 AND gate_passed_count = 0 AND best_wp < 4

The original plan rule was `n_iters >= 20 AND gate_passed_count = 0`
without a best_wp filter. We tightened it because Phase 3A's AMP
(Ablation Must Precede Pivot) clause forces depth-search whenever a
trainer's best historical windows_passed >= 4 — so archiving such
trainers would silently cancel the AMP signal. Five trainers match
the refined rule (all best_wp <= 3):

    xgboost                101 iters  best_wp=3   plain vanilla XGB
    xgb_regime_blend        31 iters  best_wp=2   regime-conditional blend never broke 2/7
    gaussian_nb             31 iters  best_wp=1   class-conditional Gaussian assumptions wrong for this universe
    label_spreading         26 iters  best_wp=2   transductive learning (reached W7 once, never broke 2/7 overall)
    torch_itransformer      22 iters  best_wp=2   variates-as-tokens transformer

Resurrection criteria — restore a key from TRAINERS_ARCHIVED back into
TRAINERS in models/trainers.py (and re-add its search space in
models/search_spaces.py) when ALL of the following hold:
    1. A new HP or architectural finding makes a >=4 windows_passed result
       plausible (cite the memory file or research brief).
    2. The owning operator manually verifies the change against the local
       data freshness state (see scripts/return_gate.py VAULT_SPLIT).
    3. The brief generator is updated to recommend the family with a
       hypothesis tying back to the original failure mode.

Operationally — `get_trainer(name)` raises with a clear archive notice
if `name in TRAINERS_ARCHIVED`. `claude_mode.detect_brief_deviation`
also flags any pick of an archived name as a deviation, triggering the
existing git-revert + active_case-skip path.
"""
from __future__ import annotations


# Names only — the actual classes still live in models/trainers.py. Archiving
# is a registration decision (does the active loop see this trainer?), not a
# code-deletion decision. The archive is intentionally reversible.
TRAINERS_ARCHIVED = {
    'xgboost': 'XGBoostTrainer',
    'xgb_regime_blend': 'XGBoostRegimeBlendTrainer',
    'gaussian_nb': 'GaussianNaiveBayesClassifierTrainer',
    'label_spreading': 'LabelSpreadingTrainer',
    'torch_itransformer': 'TorchITransformerTrainer',
}


def is_archived(name: str) -> bool:
    return name in TRAINERS_ARCHIVED


def archive_notice(name: str) -> str:
    """Return a human-readable explanation that the trainer is archived
    and how to restore it. Used by `get_trainer` and the deviation log."""
    if name not in TRAINERS_ARCHIVED:
        return ''
    return (
        f'trainer {name!r} is archived (Phase 3C, 2026-05-27). '
        f'See models/trainers_archive.py for resurrection criteria; '
        f'to restore, move the key back into TRAINERS in '
        f'models/trainers.py and re-add its HP space in '
        f'models/search_spaces.py.'
    )
