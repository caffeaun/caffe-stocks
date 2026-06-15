#!/usr/bin/env python3
"""Morning driver for the Modal-trainer approval workflow.

Runs once each weekday morning (hooked into daily-feature-refresh.sh). For
every active entry in ``models/modal_requests.json`` it surfaces — and
re-surfaces — a Telegram approval ask with an estimated Modal cost, and
acts on the operator's ``yes / no / skip`` reply:

    open / never asked  → create_approval + send_ask        (first surfacing)
    pending             → send_ask again                     (daily re-ping)
    no   (<30 days)     → stay quiet                          (deferred)
    no   (>=30 days)    → new approval + send_ask             (monthly re-ask)
    yes                 → flip request to `approved`          (train_mode drains it)
    skip                → flip request to `skipped`           (never ping again)

Reuses the existing approval gateway (`scripts/ml_approval.py`) and budget
estimator (`scripts/modal_budget.py`) — see those modules for the
`ml_approvals` schema and Modal pricing. Idempotent: safe to run repeatedly;
a still-`pending` ask is just re-sent, no duplicate DB row.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta

BASE = os.path.expanduser('~/projects/caffe-stocks')
sys.path.insert(0, BASE)

from scripts import ml_approval, modal_requests
from scripts.modal_budget import estimate_cost, monthly_spend, BUDGET_USD

DB_PATH = os.path.join(BASE, 'data', 'ml-feedback.db')

# A `no` is honoured for this long before we ask again ("ping again next month").
DEFER_DAYS = 30
# Fallback per-config runtime when the trainer has no history yet.
DEFAULT_SECONDS_PER_CONFIG = 1600.0


def _topic(trainer: str) -> str:
    return f'modal_{trainer}'


def _latest_approval(topic: str) -> dict | None:
    """Most recent ml_approvals row for a topic, or None."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        r = conn.execute(
            'SELECT id, status, note, created_at, answered_at '
            'FROM ml_approvals WHERE topic = ? ORDER BY id DESC LIMIT 1',
            (topic,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None  # table not created yet
    finally:
        conn.close()
    return dict(r) if r else None


def _median_seconds_per_config(trainer: str) -> float | None:
    """Median elapsed_seconds of this trainer's recent iterations, or None."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        rows = [r[0] for r in conn.execute(
            'SELECT elapsed_seconds FROM iterations '
            'WHERE trainer = ? AND elapsed_seconds IS NOT NULL '
            'ORDER BY id DESC LIMIT 25', (trainer,)).fetchall()]
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    if not rows:
        return None
    rows.sort()
    return float(rows[len(rows) // 2])


def _cost_line(req: dict) -> tuple[str, float]:
    """Build the human cost estimate for the ask; returns (text, est_usd)."""
    gpu = req.get('gpu', 'A100-40GB')
    configs = int(req.get('configs', 1))
    per_cfg = (_median_seconds_per_config(req['trainer'])
               or float(req.get('est_seconds_per_config') or DEFAULT_SECONDS_PER_CONFIG))
    est = estimate_cost(per_cfg * configs, gpu)
    spent = monthly_spend()
    remaining = BUDGET_USD - spent
    warn = '  ⚠️ exceeds remaining budget' if est > remaining else ''
    text = (f'{configs} configs on Modal {gpu} ≈ ${est:.2f} '
            f'(~${est / max(1, configs):.2f}/config, ~{per_cfg / 60:.0f} min each)\n'
            f'Budget: spent ${spent:.2f} / ${BUDGET_USD:.0f} this month, '
            f'${remaining:.2f} left{warn}')
    return text, est


def _question(req: dict) -> str:
    cost_text, _ = _cost_line(req)
    note = req.get('note', '')
    focus = req.get('focus', '')
    head = f'Run {req["trainer"]}' + (f' ({focus})' if focus else '') + ' on Modal?'
    return f'{head}\n\n{cost_text}' + (f'\n\n{note}' if note else '')


def _ask(req: dict) -> int:
    """Create a fresh approval row + send the Telegram ask. Returns its id."""
    aid = ml_approval.create_approval(_topic(req['trainer']), _question(req))
    ml_approval.send_ask(aid, header='🖥️ Modal request')
    return aid


def _resend(req: dict, approval_id: int) -> None:
    ml_approval.send_ask(approval_id, header='🖥️ Modal request (reminder)')


def _is_stale_no(approval: dict) -> bool:
    """True if a `no` was answered at least DEFER_DAYS ago."""
    ts = approval.get('answered_at') or approval.get('created_at')
    if not ts:
        return True
    try:
        when = datetime.fromisoformat(ts)
    except ValueError:
        return True
    return datetime.now() - when >= timedelta(days=DEFER_DAYS)


def process_one(req: dict) -> str:
    """Advance a single request's state machine; return a log string."""
    trainer = req['trainer']
    topic = _topic(trainer)
    latest = _latest_approval(topic)

    # yes already consumed → request is approved/draining: nothing to ping.
    if req.get('status') == 'approved':
        return f'{trainer}: approved (draining, remaining={req.get("remaining_configs")})'

    if latest is None:
        aid = _ask(req)
        return f'{trainer}: first ask sent (ml #{aid})'

    status = (latest.get('status') or 'pending').lower()

    if status == 'pending':
        _resend(req, latest['id'])
        return f'{trainer}: re-pinged pending ml #{latest["id"]}'

    if status == 'yes':
        modal_requests.set_status(
            trainer, 'approved', remaining_configs=int(req.get('configs', 1)))
        return (f'{trainer}: APPROVED → flipped to approved, '
                f'{req.get("configs")} configs queued for train_mode')

    if status == 'skip':
        modal_requests.set_status(trainer, 'skipped')
        return f'{trainer}: skipped (will not ask again)'

    if status == 'no':
        if _is_stale_no(latest):
            aid = _ask(req)
            return f'{trainer}: monthly re-ask sent (ml #{aid}, prior no >= {DEFER_DAYS}d)'
        return f'{trainer}: deferred (no answered < {DEFER_DAYS}d ago) — quiet'

    # Unknown status — treat conservatively as pending and remind.
    _resend(req, latest['id'])
    return f'{trainer}: unknown status {status!r} — re-pinged ml #{latest["id"]}'


def main() -> int:
    reqs = [r for r in modal_requests.load()
            if r.get('status') in modal_requests.ACTIVE_STATUSES]
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if not reqs:
        print(f'[{stamp}] modal_request_ping: no active requests.')
        return 0
    print(f'[{stamp}] modal_request_ping: {len(reqs)} active request(s)')
    for req in reqs:
        try:
            print('  ' + process_one(req))
        except Exception as e:  # one bad request must not stop the rest
            print(f'  {req.get("trainer")}: ERROR {e!r}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
