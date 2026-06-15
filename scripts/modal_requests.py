#!/usr/bin/env python3
"""Shared reader/writer for the Modal-trainer request queue.

`models/modal_requests.json` is a standing list of trainers the operator
(or claude_mode / research) wants to try on Modal but which are too
expensive for the hourly local sweep. The morning driver
(`scripts/modal_request_ping.py`) surfaces each one as a Telegram approval
ask; on `yes` it flips the entry to ``approved`` and `train_mode` drains it
one Modal config per fire via `next_approved_job` / `decrement`.

Both the ping driver and train_mode import this module so there is exactly
one schema + one writer. All functions are crash-safe — a missing or
malformed file yields an empty queue rather than raising, so a bad edit can
never wedge the training loop.

Entry schema (one dict per request)::

    {
      "trainer": "torch_scarf",         # registry key (must self-route to Modal)
      "gpu": "A100-40GB",               # modal_budget.GPU_HOURLY_USD key
      "configs": 8,                     # how many HP configs the bounded sweep runs
      "focus": "unsupervised",          # supervised | unsupervised (informational)
      "note": "…",                      # shown in the Telegram ask
      "added": "2026-06-15",            # ISO date the request was queued
      "status": "open",                 # open | approved | done | skipped
      "remaining_configs": null          # set to `configs` on approval, counts down
    }

Status lifecycle::

    open ──yes──▶ approved ──(K fires)──▶ done
      │
      └──skip──▶ skipped         (no <30d: stays open & quiet; no ≥30d: re-asked)
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

REQUESTS_PATH = os.path.expanduser(
    '~/projects/caffe-stocks/models/modal_requests.json')

# Statuses that still want a morning ping / can still execute.
ACTIVE_STATUSES = ('open', 'approved')


def load(path: str | None = None) -> list[dict]:
    """Return the request list, or [] if the file is missing/malformed.

    Never raises — a hand-edit typo must not wedge train_mode."""
    p = path or REQUESTS_PATH
    try:
        with open(p) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict) and e.get('trainer')]


def save(requests: list[dict], path: str | None = None) -> None:
    """Atomically rewrite the queue (temp file + os.replace)."""
    p = path or REQUESTS_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(p), prefix='.modal_requests.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(requests, f, indent=2)
            f.write('\n')
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def find(trainer: str, statuses: tuple[str, ...] | None = None,
         path: str | None = None) -> Optional[dict]:
    """First request for `trainer`, optionally filtered to `statuses`."""
    for e in load(path):
        if e['trainer'] != trainer:
            continue
        if statuses is None or e.get('status') in statuses:
            return e
    return None


def next_approved_job(path: str | None = None) -> Optional[dict]:
    """First request that is approved with configs still to run, else None.

    This is the executor hook: train_mode calls it before resolving its
    normal trainer so an approved Modal sweep preempts the local rotation."""
    for e in load(path):
        if e.get('status') == 'approved' and int(e.get('remaining_configs') or 0) > 0:
            return e
    return None


def set_status(trainer: str, status: str, *, remaining_configs: int | None = None,
               path: str | None = None, **extra) -> bool:
    """Update the first active (open/approved) request for `trainer`.

    Returns True if a row was updated. `remaining_configs` and any `extra`
    keys are merged onto the entry."""
    reqs = load(path)
    for e in reqs:
        if e['trainer'] == trainer and e.get('status') in ACTIVE_STATUSES:
            e['status'] = status
            if remaining_configs is not None:
                e['remaining_configs'] = remaining_configs
            e.update(extra)
            save(reqs, path)
            return True
    return False


def decrement(trainer: str, path: str | None = None) -> Optional[dict]:
    """Count one executed Modal config down for `trainer`'s approved job.

    Flips status to ``done`` when it hits zero. Returns the updated entry,
    or None if no approved job for that trainer was found."""
    reqs = load(path)
    for e in reqs:
        if e['trainer'] == trainer and e.get('status') == 'approved':
            remaining = int(e.get('remaining_configs') or 0) - 1
            e['remaining_configs'] = max(0, remaining)
            if remaining <= 0:
                e['status'] = 'done'
            save(reqs, path)
            return e
    return None


if __name__ == '__main__':  # tiny inspector
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == 'next':
        print(next_approved_job())
    else:
        for e in load():
            print(f"{e['trainer']:20s} status={e.get('status'):9s} "
                  f"remaining={e.get('remaining_configs')} configs={e.get('configs')}")
