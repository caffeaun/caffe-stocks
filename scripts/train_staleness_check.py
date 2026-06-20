#!/usr/bin/env python3
"""Emit a one-line alert if the hourly train_mode sweep has gone stale.

The sweep fires every hour (08:30–21:30) and records an iteration per config.
If it stops recording for a long stretch, something is wrong — a jammed Modal
job, a crash, a bad active_case, a lock left held. That exact failure happened
2026-06-15→20: an approved torch_scarf Modal job preempted every fire and died
on Modal's exhausted prepaid balance, recording nothing, and the rotation sat
dead for 5 days unnoticed. A jammed sweep records ZERO iterations, so
"no train iteration in N hours" is the precise signal.

Prints the alert text to stdout (the 06:00 morning script forwards it to
Telegram) when stale, prints nothing when healthy. ALWAYS exits 0 and fails
soft — a DB hiccup must never break the morning cron. Threshold via
MAX_TRAIN_STALENESS_HOURS (default 24h: the sweep runs daily, so a full day
with no iteration is unambiguous and won't false-positive on the overnight
22:00–08:00 gap or weekends).
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime

DB_PATH = os.path.expanduser('~/projects/caffe-stocks/data/ml-feedback.db')
THRESHOLD_H = float(os.environ.get('MAX_TRAIN_STALENESS_HOURS', '24'))


def last_train_iter() -> tuple[int, str, str] | None:
    """(id, trainer, finished_at) of the most recent recorded train iteration."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        r = conn.execute(
            "SELECT id, trainer, finished_at FROM iterations "
            "WHERE mode='train' AND finished_at IS NOT NULL "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return r


def check() -> str | None:
    """Return an alert string if the sweep is stale, else None."""
    try:
        r = last_train_iter()
    except Exception:
        return None  # fail soft — never break the cron on a DB hiccup
    if not r:
        return ('🟠 Train sweep: no train_mode iteration has *ever* been '
                'recorded — the hourly rotation is not running.')
    _id, trainer, finished = r
    try:
        age_h = (datetime.now() - datetime.fromisoformat(finished)).total_seconds() / 3600
    except Exception:
        return None
    if age_h > THRESHOLD_H:
        return (f'🟠 Train sweep STALE — {age_h:.0f}h since the last train_mode '
                f'iteration (#{_id} {trainer}, {finished[:16]}). The hourly '
                f'rotation may be jammed. Check models/modal_requests.json '
                f'(a held/failing Modal job) and logs/ml-loop.log.')
    return None


def main() -> int:
    alert = check()
    if alert:
        print(alert)
    return 0


if __name__ == '__main__':
    sys.exit(main())
