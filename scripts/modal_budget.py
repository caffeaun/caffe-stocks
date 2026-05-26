"""Monthly spend gate for Modal calls.

Schema lives in data/ml-feedback.db `modal_usage`:
    id, called_at, trainer, gpu, duration_s, cost_usd, status

check_budget() raises ModalBudgetExceeded if the running monthly total
plus the estimated cost of the next call would exceed BUDGET_USD.
record_call() writes the actual cost after the call returns.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path('/home/kanoonth-ai/projects/caffe-stocks/data/ml-feedback.db')
BUDGET_USD = float(os.environ.get('MODAL_MONTHLY_BUDGET_USD', '30.0'))

# Modal A100-40GB on-demand pricing (rough; check modal.com/pricing).
# Updated 2026-05 — $2.0/hr placeholder; tighten when actuals land.
GPU_HOURLY_USD = {
    'A100-40GB': 2.0,
    'A100-80GB': 3.4,
    'H100': 4.0,
    'T4': 0.6,
    'L4': 0.8,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS modal_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    called_at TEXT NOT NULL,
    trainer TEXT NOT NULL,
    gpu TEXT NOT NULL,
    duration_s REAL NOT NULL,
    cost_usd REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'ok'
)
"""


class ModalBudgetExceeded(RuntimeError):
    pass


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _month_key(dt: datetime) -> str:
    return dt.strftime('%Y-%m')


def monthly_spend(now: datetime | None = None) -> float:
    now = now or datetime.now()
    prefix = _month_key(now) + '%'
    with _conn() as conn:
        r = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM modal_usage "
            "WHERE called_at LIKE ? AND status = 'ok'",
            (prefix,),
        ).fetchone()
    return float(r[0])


def estimate_cost(duration_s: float, gpu: str = 'A100-40GB') -> float:
    rate = GPU_HOURLY_USD.get(gpu, 2.0)
    return duration_s / 3600.0 * rate


def check_budget(estimated_duration_s: float, gpu: str = 'A100-40GB') -> None:
    """Raise ModalBudgetExceeded if running this call would push us past
    BUDGET_USD for the current calendar month."""
    spent = monthly_spend()
    cost = estimate_cost(estimated_duration_s, gpu)
    if spent + cost > BUDGET_USD:
        raise ModalBudgetExceeded(
            f'Modal monthly cap ${BUDGET_USD:.2f} would be exceeded: '
            f'spent=${spent:.2f}, this call est. ${cost:.2f}'
        )


def record_call(
    trainer: str,
    duration_s: float,
    gpu: str = 'A100-40GB',
    status: str = 'ok',
    cost_usd: float | None = None,
) -> None:
    """Insert a row recording an actual call. Pass cost_usd to override
    the duration-based estimate (e.g. from Modal's billing API)."""
    if cost_usd is None:
        cost_usd = estimate_cost(duration_s, gpu)
    with _conn() as conn:
        conn.execute(
            "INSERT INTO modal_usage "
            "(called_at, trainer, gpu, duration_s, cost_usd, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(timespec='seconds'),
                trainer, gpu, duration_s, cost_usd, status,
            ),
        )


def report() -> str:
    """Human-readable monthly summary; for `python -m scripts.modal_budget`."""
    now = datetime.now()
    spent = monthly_spend(now)
    with _conn() as conn:
        rows = conn.execute(
            "SELECT trainer, COUNT(*) n, SUM(duration_s) tot_s, "
            "SUM(cost_usd) tot_usd FROM modal_usage "
            "WHERE called_at LIKE ? AND status = 'ok' "
            "GROUP BY trainer ORDER BY tot_usd DESC",
            (_month_key(now) + '%',),
        ).fetchall()
    lines = [
        f'Modal spend — {_month_key(now)}',
        f'  budget : ${BUDGET_USD:.2f}',
        f'  spent  : ${spent:.2f}',
        f'  remain : ${BUDGET_USD - spent:.2f}',
    ]
    if rows:
        lines.append('  by trainer:')
        for r in rows:
            lines.append(
                f'    {r["trainer"]:24}  n={r["n"]:<3}  '
                f'{r["tot_s"]:7.1f} s  ${r["tot_usd"]:.2f}'
            )
    return '\n'.join(lines)


if __name__ == '__main__':
    print(report())
