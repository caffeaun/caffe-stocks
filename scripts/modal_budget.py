"""Monthly spend gate for Modal calls.

Schema lives in data/ml-feedback.db `modal_usage`:
    id, called_at, trainer, gpu, duration_s, cost_usd, status

check_budget() raises ModalBudgetExceeded if the running monthly total
plus the estimated cost of the next call would exceed BUDGET_USD.
record_call() writes the actual cost after the call returns.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time as _time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path('/home/kanoonth-ai/projects/caffe-stocks/data/ml-feedback.db')
BUDGET_USD = float(os.environ.get('MODAL_MONTHLY_BUDGET_USD', '30.0'))

# Real Modal-side workspace spend is cached here (the billing endpoint is
# aggressively rate-limited, and check_budget runs before every Modal fold).
_WS_CACHE_PATH = DB_PATH.parent / '.modal_workspace_spend.json'
_WS_CACHE_TTL_S = 600  # 10 min

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


def _ws_cache_read():
    try:
        d = json.loads(_WS_CACHE_PATH.read_text())
        return float(d['ts']), (None if d['value'] is None else float(d['value']))
    except Exception:
        return None, None


def _ws_cache_write(value: float | None) -> None:
    try:
        _WS_CACHE_PATH.write_text(json.dumps({'ts': _time.time(), 'value': value}))
    except Exception:
        pass


def workspace_spend(ttl_s: int = _WS_CACHE_TTL_S,
                    now: datetime | None = None) -> float | None:
    """Real Modal-side workspace spend this calendar month (settled days).

    The AUTHORITATIVE figure — the same `modal.billing.workspace_billing_report`
    source the approval ping uses, and what Modal's own spend cap is enforced
    against. Independent of (and higher than) our `modal_usage` tracker, which
    only sees calls it recorded. File-cached for `ttl_s` because Modal
    rate-limits the billing endpoint hard and check_budget runs before every
    fold; on rate-limit / error returns the last cached value (possibly stale),
    else None."""
    ts, cached = _ws_cache_read()
    if ts is not None and (_time.time() - ts) < ttl_s:
        return cached
    try:
        import modal.billing as _mb
        n = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        start = n.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        rows = _mb.workspace_billing_report(start=start, end=n)
        val = float(sum(float(r.get('cost', 0) or 0) for r in rows))
        _ws_cache_write(val)
        return val
    except Exception:
        return cached  # stale cache if present, else None


def effective_spend(now: datetime | None = None) -> float:
    """Best estimate of real cycle spend for gating: the larger of the
    internal tracker (captures TODAY's recorded calls — the billing API's
    settled-days view omits the partial current day) and Modal's authoritative
    settled spend (captures history our tracker missed). max() never
    under-gates."""
    internal = monthly_spend(now)
    ws = workspace_spend(now=now)
    return max(internal, ws) if ws is not None else internal


def check_budget(estimated_duration_s: float, gpu: str = 'A100-40GB') -> None:
    """Raise ModalBudgetExceeded if running this call would push us past
    BUDGET_USD this cycle.

    Gates on the REAL Modal workspace spend (same billing source as the
    approval ping), not just our internal tracker — so a run is stopped
    *before* it trips Modal's own cap with a mid-fold ResourceExhaustedError
    (the iter #2801 failure), instead of after."""
    spent = effective_spend()
    cost = estimate_cost(estimated_duration_s, gpu)
    if spent + cost > BUDGET_USD:
        raise ModalBudgetExceeded(
            f'Modal cap ${BUDGET_USD:.2f} would be exceeded: '
            f'spent=${spent:.2f} (real workspace), this call est. ${cost:.2f}'
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
    ws = workspace_spend(now=now)
    eff = effective_spend(now)
    lines = [
        f'Modal spend — {_month_key(now)}',
        f'  budget    : ${BUDGET_USD:.2f}',
        f'  tracker   : ${spent:.2f}  (modal_usage table — undercounts)',
        (f'  workspace : ${ws:.2f}  (Modal billing, settled days — authoritative)'
         if ws is not None else '  workspace : n/a  (billing API unavailable)'),
        f'  remain    : ${BUDGET_USD - eff:.2f}  (gated on max of the two)',
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
