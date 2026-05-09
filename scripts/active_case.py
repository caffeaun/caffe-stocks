"""Active experimental case — handoff state between claude_mode and train_mode.

Single tiny piece of state living at ``models/active_case.json``. Claude mode
writes it at the end of each successful structural-change run; train mode
reads it at startup to decide which trainer family to sweep within.

Schema (all fields optional except ``trainer``)::

    {
      "trainer": "xgb_ranker",
      "feature_set": "default",
      "dataset": {"seq_len": 20, "lookahead": 10, "label": "v1"},
      "designated_at": "2026-05-06T11:30:00",
      "claude_iter_id": 80,
      "rationale": "WR 49.5%, zero negative-ann windows — strongest base"
    }

Pure JSON I/O. No DB. The file is the only piece of state — if it's missing
or malformed, callers should fall back to their own defaults.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional


BASE_PATH = Path(os.path.expanduser('~/projects/caffe-stocks'))
ACTIVE_CASE_PATH = BASE_PATH / 'models' / 'active_case.json'

# Conservative defaults applied when claude's report doesn't specify them.
DEFAULT_FEATURE_SET = 'default'
DEFAULT_DATASET = {'seq_len': 20, 'lookahead': 10, 'label': 'v1'}


def read(path: Optional[Path] = None) -> Optional[dict]:
    """Return the active case dict, or None if absent/malformed.

    Never raises. Callers treat None as "fall back to defaults".
    """
    p = Path(path) if path else ACTIVE_CASE_PATH
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get('trainer'):
        return None
    return data


def write(case: dict, path: Optional[Path] = None) -> Path:
    """Write the active case to ``models/active_case.json`` atomically.

    Adds ``designated_at`` if missing. Creates parent dirs as needed.
    """
    if not isinstance(case, dict) or not case.get('trainer'):
        raise ValueError(f'active case must be a dict with a "trainer" key, got {case!r}')
    p = Path(path) if path else ACTIVE_CASE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)

    case = dict(case)  # don't mutate caller's dict
    case.setdefault('designated_at', datetime.now().isoformat(timespec='seconds'))
    case.setdefault('feature_set', DEFAULT_FEATURE_SET)
    case.setdefault('dataset', dict(DEFAULT_DATASET))

    # Atomic replace: write to temp file in same dir, then rename.
    fd, tmp_path = tempfile.mkstemp(prefix='.active_case-', suffix='.tmp', dir=str(p.parent))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(case, f, indent=2, sort_keys=False, default=str)
            f.write('\n')
        os.replace(tmp_path, p)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return p


def from_report(report: dict, claude_iter_id: Optional[int] = None) -> Optional[dict]:
    """Build an active_case from a Claude-mode JSON report.

    Order of preference:
      1. ``report['active_case']`` if present (claude opted in to the contract).
      2. Synthesized from ``report['trainer']`` + defaults (back-compat — keeps
         the writer correct even when the prompt is older than the contract).

    Returns None if even the trainer is missing — caller should skip the write.
    """
    block = report.get('active_case')
    if isinstance(block, dict) and block.get('trainer'):
        case = dict(block)
        case.setdefault('feature_set', DEFAULT_FEATURE_SET)
        case.setdefault('dataset', dict(DEFAULT_DATASET))
        if claude_iter_id is not None:
            case.setdefault('claude_iter_id', claude_iter_id)
        case.setdefault(
            'rationale',
            report.get('hypothesis') or 'no rationale supplied',
        )
        return case

    trainer = report.get('trainer')
    if not trainer:
        return None
    return {
        'trainer': trainer,
        'feature_set': DEFAULT_FEATURE_SET,
        'dataset': dict(DEFAULT_DATASET),
        'claude_iter_id': claude_iter_id,
        'rationale': (report.get('hypothesis')
                      or 'inferred from trainer field — no active_case block in report'),
    }


def main():
    """CLI: print the current active case (or 'none')."""
    case = read()
    if case is None:
        print('no active case')
        return
    print(json.dumps(case, indent=2))


if __name__ == '__main__':
    main()
