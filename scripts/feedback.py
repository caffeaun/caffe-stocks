"""SQLite-backed iteration log + promotion panel.

Single source of truth for the ML loop. Both train_mode and claude_mode
write iteration rows; promotion script and the bot read.

WAL mode is enabled so the bot can read while train_mode is mid-iteration.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


BASE_PATH = Path(os.path.expanduser('~/projects/caffe-stocks'))
DB_PATH = BASE_PATH / 'data' / 'ml-feedback.db'

_TELEGRAM_CONF_PATHS = (
    BASE_PATH.parent / 'ops' / 'telegram' / 'telegram.conf',
    Path.home() / 'kanoonth' / 'scripts' / 'telegram.conf',
)


def _maybe_telegram_warn(message: str) -> None:
    """Best-effort Telegram alert. Swallows every exception — never lets
    a telemetry failure block an iteration write. Mirrors the conf-loading
    pattern from scripts/leaderboard.py:telegram()."""
    try:
        import urllib.parse
        import urllib.request
        bot_token = chat_id = None
        for p in _TELEGRAM_CONF_PATHS:
            if not p.exists():
                continue
            with open(p) as f:
                for line in f:
                    s = line.strip()
                    if s.startswith('TELEGRAM_BOT_TOKEN='):
                        bot_token = s.split('=', 1)[1].strip().strip('"\'')
                    elif s.startswith('TELEGRAM_CHAT_ID='):
                        chat_id = s.split('=', 1)[1].strip().strip('"\'')
            if bot_token and chat_id:
                break
        if not (bot_token and chat_id):
            return
        data = urllib.parse.urlencode({
            'chat_id': chat_id,
            'text': message[:3900],
            'disable_web_page_preview': 'true',
        }).encode()
        urllib.request.urlopen(
            f'https://api.telegram.org/bot{bot_token}/sendMessage',
            data=data, timeout=10,
        )
    except Exception:
        pass

SCHEMA = """
CREATE TABLE IF NOT EXISTS iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('train', 'claude')),
    trainer TEXT NOT NULL,
    hyperparams TEXT NOT NULL,
    code_changes TEXT,
    hypothesis TEXT,
    backbone TEXT,
    git_sha TEXT,
    gate_passed INTEGER NOT NULL,
    windows_passed INTEGER NOT NULL,
    windows_total INTEGER NOT NULL,
    avg_annualized_return REAL,
    avg_win_rate REAL,
    avg_max_dd REAL,
    total_trades INTEGER,
    model_dir TEXT,
    elapsed_seconds INTEGER,
    full_result TEXT,
    lessons TEXT
);

CREATE INDEX IF NOT EXISTS idx_iter_finished      ON iterations(finished_at);
CREATE INDEX IF NOT EXISTS idx_iter_passed_return ON iterations(gate_passed, avg_annualized_return);
CREATE INDEX IF NOT EXISTS idx_iter_mode_finished ON iterations(mode, finished_at);

CREATE TABLE IF NOT EXISTS iteration_windows (
    iteration_id INTEGER NOT NULL REFERENCES iterations(id),
    window_idx INTEGER NOT NULL,
    train_start TEXT, train_end TEXT,
    test_start TEXT,  test_end TEXT,
    threshold REAL,
    n_trades INTEGER,
    win_rate REAL,
    avg_pnl REAL,
    avg_win REAL,
    avg_loss REAL,
    annualized_return REAL,
    max_drawdown REAL,
    final_equity REAL,
    PRIMARY KEY (iteration_id, window_idx)
);

CREATE TABLE IF NOT EXISTS production_panel (
    rank INTEGER PRIMARY KEY CHECK(rank IN (1, 2, 3)),
    iteration_id INTEGER NOT NULL REFERENCES iterations(id),
    promoted_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    UNIQUE(iteration_id)
);

CREATE TABLE IF NOT EXISTS data_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_id INTEGER NOT NULL REFERENCES iterations(id),
    requested_at TEXT NOT NULL,
    request_text TEXT NOT NULL,
    fulfilled INTEGER DEFAULT 0
);
"""


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute('PRAGMA journal_mode = WAL')
    conn.execute('PRAGMA synchronous = NORMAL')
    conn.execute('PRAGMA foreign_keys = ON')
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Idempotent schema creation."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # Idempotent ALTER for the Bayesian-gate column (Phase 1E, 2026-05-27).
        # Stored alongside windows_passed; the deterministic column still
        # drives gate_passed promotion until the A/B comparison concludes.
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(iterations)"
        ).fetchall()}
        if 'bayes_windows_passed' not in cols:
            conn.execute(
                'ALTER TABLE iterations ADD COLUMN '
                'bayes_windows_passed INTEGER DEFAULT NULL'
            )
        if 'bayes_windows_passed' not in {r[1] for r in conn.execute(
                "PRAGMA table_info(iteration_windows)"
        ).fetchall()}:
            conn.execute(
                'ALTER TABLE iteration_windows ADD COLUMN '
                'bayes_passed INTEGER DEFAULT NULL'
            )


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ['git', '-C', str(BASE_PATH), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, timeout=2)
        return out.decode().strip()
    except Exception:
        return None


def record_iteration(*, mode: str, trainer: str, hyperparams: dict,
                      gate_result: dict, model_dir: str,
                      started_at: str, finished_at: str,
                      elapsed_seconds: int,
                      code_changes: Optional[list] = None,
                      hypothesis: Optional[str] = None,
                      backbone: Optional[str] = None,
                      lessons: Optional[str] = None) -> int:
    """Append an iteration row + per-window detail. Returns iteration id."""
    init_db()
    agg = _aggregate(gate_result)

    # Sanity check: catch the zero-aggregate bug class at write-time.
    # Pattern: walk-forward DID produce windows (wp>0) but aggregates collapsed
    # to zero — diagnostic of a schema mismatch between producer and _aggregate
    # (the 2026-05 claude/train shape mismatch was the original cause). We
    # warn + Telegram but still commit the row so the per-window data is
    # preserved for later forensic recovery via backfill_zero_aggregates.py.
    wp = int(gate_result.get('windows_passed', 0))
    if (wp > 0 and agg['total_trades'] == 0
            and agg['avg_win_rate'] == 0.0
            and agg['avg_annualized_return'] == 0.0):
        gr_keys = list(gate_result.keys())
        results = gate_result.get('results') or []
        r0_keys = list(results[0].keys()) if results else []
        msg = (
            f'⚠️ aggregate sanity failed: {trainer} (mode={mode}, '
            f'wp={wp}/{gate_result.get("windows_total", 7)}) — '
            f'wr=0% total_trades=0 avg_ann=0% despite wp>0. '
            f'Likely a new gate_result schema. '
            f'gate_result keys: {gr_keys}. result[0] keys: {r0_keys}. '
            f'Recover via scripts/backfill_zero_aggregates.py once '
            f'scripts/feedback.py:_aggregate handles the new shape.'
        )
        print(msg, file=sys.stderr)
        _maybe_telegram_warn(msg)

    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO iterations
            (started_at, finished_at, mode, trainer, hyperparams, code_changes,
             hypothesis, backbone, git_sha, gate_passed, windows_passed,
             windows_total, avg_annualized_return, avg_win_rate, avg_max_dd,
             total_trades, model_dir, elapsed_seconds, full_result, lessons,
             bayes_windows_passed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            started_at, finished_at, mode, trainer,
            json.dumps(hyperparams, default=str),
            json.dumps(code_changes) if code_changes else None,
            hypothesis, backbone, _git_sha(),
            int(bool(gate_result.get('gate_passed'))),
            int(gate_result.get('windows_passed', 0)),
            int(gate_result.get('windows_total', 0)),
            agg['avg_annualized_return'], agg['avg_win_rate'],
            agg['avg_max_dd'], agg['total_trades'],
            str(model_dir), int(elapsed_seconds),
            json.dumps(gate_result, default=str), lessons,
            agg.get('bayes_windows_passed'),
        ))
        iter_id = cur.lastrowid

        for idx, r in enumerate(gate_result.get('results', []), 1):
            m = _window_metrics(r)
            bp = r.get('bayes_passed')
            conn.execute("""
                INSERT INTO iteration_windows
                (iteration_id, window_idx, train_start, train_end, test_start, test_end,
                 threshold, n_trades, win_rate, avg_pnl, avg_win, avg_loss,
                 annualized_return, max_drawdown, final_equity, bayes_passed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                iter_id, idx,
                *_split_period(r.get('train', '')),
                *_split_period(r.get('test', '')),
                m.get('threshold'),
                m.get('n_trades'),
                m.get('win_rate'),
                m.get('avg_pnl'),
                m.get('avg_win'),
                m.get('avg_loss'),
                m.get('annualized_return'),
                m.get('max_drawdown'),
                m.get('final_equity'),
                int(bool(bp)) if bp is not None else None,
            ))
        return iter_id


def _split_period(s: str) -> tuple[Optional[str], Optional[str]]:
    """'2023-05-01..2023-10-31' → ('2023-05-01', '2023-10-31')."""
    if not s or '..' not in s:
        return None, None
    a, b = s.split('..', 1)
    return a, b


def _window_metrics(r: dict) -> dict:
    """Per-window metric dict.

    train_mode emits `{"best": {n_trades, win_rate, annualized_return, ...}}`
    (the winning threshold from its sweep). claude_mode emits the same
    fields flat on the result itself (it picks one threshold and reports
    that). This helper accepts either shape.
    """
    if isinstance(r.get('best'), dict):
        return r['best']
    return r


def _aggregate(gate_result: dict) -> dict:
    """Aggregate per-window metrics into iteration-level averages.

    Per-key fallback so both train_mode and claude_mode JSON shapes
    work without prompt changes. For each field independently:
      1. Use the top-level value on `gate_result` if it's present AND
         non-zero (zeros are treated as "unfilled" — both producers
         leave some aggregates at the default 0.0).
      2. Else compute from per-window data, accepting either nested
         `best` sub-dict (train_mode shape) or flat per-window keys
         (claude_mode shape).
      3. Else zero (truly empty).
    """
    results = gate_result.get('results', []) or []
    metrics = [_window_metrics(r) for r in results]
    metrics = [m for m in metrics if m]
    n = len(metrics)

    def computed(key: str, cast=float) -> float:
        if n == 0:
            return cast(0)
        if cast is int:
            return cast(sum(m.get(key, 0) or 0 for m in metrics))
        return cast(sum(m.get(key, 0) or 0 for m in metrics) / n)

    tier_used: dict[str, str] = {}

    def pick(top_key: str, window_key: str, cast=float):
        top = gate_result.get(top_key)
        if top is not None and float(top) != 0.0:
            tier_used[top_key] = 'top'
            return cast(top)
        tier_used[top_key] = 'computed' if n > 0 else 'zero'
        return computed(window_key, cast=cast)

    # Bayesian-gate count is summed directly from per-window `bayes_passed`.
    # None when the producer didn't emit it (legacy iters / minimal claude
    # shape). Stored alongside deterministic windows_passed for A/B analysis.
    bayes_flags = [r.get('bayes_passed') for r in results]
    if any(b is not None for b in bayes_flags):
        bayes_wp = sum(1 for b in bayes_flags if b)
    else:
        bayes_wp = None

    out = {
        'avg_annualized_return': pick('avg_annualized_return', 'annualized_return'),
        'avg_win_rate':          pick('avg_win_rate', 'win_rate'),
        'avg_max_dd':            pick('avg_max_dd', 'max_drawdown'),
        'total_trades':          pick('total_trades', 'n_trades', cast=int),
        'bayes_windows_passed':  bayes_wp,
    }
    # Forensic trail — one line per call, lands in the same log path as the
    # caller (claude-mode-*.log / ml-loop.log). Helps post-hoc analysis when
    # an aggregate looks off; if a future producer drifts schema, the
    # `tier_used` summary makes the misroute obvious.
    print(
        f'[feedback._aggregate] tiers='
        f'avg_ann={tier_used["avg_annualized_return"]} '
        f'avg_wr={tier_used["avg_win_rate"]} '
        f'avg_max_dd={tier_used["avg_max_dd"]} '
        f'total_trades={tier_used["total_trades"]} '
        f'n_windows={n}',
        file=sys.stderr,
    )
    return out


# ---------------------- query helpers ----------------------

def top_n_recent(n: int = 3, days: int = 30) -> list[dict]:
    """Return up to n recent iterations in the last `days`, sorted by
    avg_annualized_return DESC then avg_max_dd ASC. No policy filtering —
    callers (e.g. promotion.py) apply gate_passed / trainer filters as needed.
    """
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM iterations
            WHERE finished_at >= ?
            ORDER BY avg_annualized_return DESC, avg_max_dd ASC
            LIMIT ?
        """, (cutoff, n)).fetchall()
    return [dict(r) for r in rows]


def get_promotion_panel() -> list[dict]:
    """Return current panel ordered by rank."""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT pp.rank, pp.iteration_id, pp.promoted_at, pp.expires_at,
                   it.trainer, it.hyperparams, it.avg_annualized_return,
                   it.avg_win_rate, it.avg_max_dd, it.total_trades, it.model_dir
            FROM production_panel pp
            JOIN iterations it ON it.id = pp.iteration_id
            ORDER BY pp.rank
        """).fetchall()
    return [dict(r) for r in rows]


def set_panel(iteration_ids: list[int]) -> list[dict]:
    """Replace the production panel with up to 3 iteration ids (rank by order).
    Pure DB writer — eligibility / dedupe policy belongs in the caller
    (see promotion.py). Pass [] to clear the panel.
    """
    init_db()
    if len(iteration_ids) > 3:
        raise ValueError(f'panel holds at most 3 entries, got {len(iteration_ids)}')
    now = datetime.now().isoformat()
    expires = (datetime.now() + timedelta(days=30)).isoformat()
    with get_conn() as conn:
        conn.execute('DELETE FROM production_panel')
        for rank, iter_id in enumerate(iteration_ids, 1):
            conn.execute("""
                INSERT INTO production_panel (rank, iteration_id, promoted_at, expires_at)
                VALUES (?, ?, ?, ?)
            """, (rank, iter_id, now, expires))
    return get_promotion_panel()


def best_candidate_ann_return() -> Optional[float]:
    """Highest avg_annualized_return among prior gate-passing candidates.
    Returns None if no candidate has passed yet — first candidate floor is then
    applied by the caller (avg ann > 0)."""
    init_db()
    with get_conn() as conn:
        row = conn.execute("""
            SELECT MAX(avg_annualized_return) FROM iterations
            WHERE gate_passed = 1
        """).fetchone()
    if row is None or row[0] is None:
        return None
    return float(row[0])


def consecutive_failures(limit: int = 10) -> int:
    """Count consecutive trailing iterations where gate_passed = 0.

    Informational only — surfaced in the claude_mode start alert. No longer
    gates execution (auto-pause was removed).
    """
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT gate_passed FROM iterations
            ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
    streak = 0
    for r in rows:
        if r['gate_passed'] == 0:
            streak += 1
        else:
            break
    return streak


def log_data_request(iteration_id: int, request_text: str) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO data_requests (iteration_id, requested_at, request_text)
            VALUES (?, ?, ?)
        """, (iteration_id, datetime.now().isoformat(), request_text))


def stats() -> dict:
    """Aggregate iteration stats."""
    init_db()
    with get_conn() as conn:
        total = conn.execute('SELECT COUNT(*) FROM iterations').fetchone()[0]
        passed = conn.execute('SELECT COUNT(*) FROM iterations WHERE gate_passed = 1').fetchone()[0]
        by_mode = dict(conn.execute(
            'SELECT mode, COUNT(*) FROM iterations GROUP BY mode').fetchall())
        by_trainer = dict(conn.execute(
            'SELECT trainer, COUNT(*) FROM iterations GROUP BY trainer').fetchall())
        last_pass = conn.execute(
            'SELECT finished_at FROM iterations WHERE gate_passed = 1 '
            'ORDER BY finished_at DESC LIMIT 1').fetchone()
    return {
        'total': total,
        'passed': passed,
        'by_mode': by_mode,
        'by_trainer': by_trainer,
        'last_pass': last_pass[0] if last_pass else None,
        'consecutive_failures': consecutive_failures(),
    }


if __name__ == '__main__':
    init_db()
    print(json.dumps(stats(), indent=2))
