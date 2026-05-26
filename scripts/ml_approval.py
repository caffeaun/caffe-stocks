"""Helper for creating and polling ML approval requests through the
ops Telegram gateway.

The gateway exposes `ml <id> yes|no|skip [note]` commands that update
the `ml_approvals` table in data/ml-feedback.db. This module wraps the
caffe-stocks side: creating an approval row, sending the Telegram ask,
and polling the row for the user's answer.

Usage:
    from scripts.ml_approval import create_approval, check_approval, send_ask

    aid = create_approval(
        topic='modal_tabicl_v2',
        question='Approve TabICLv2 on Modal A100 at $0.12/run?',
    )
    send_ask(aid)  # posts to Telegram with the new `ml <id> ...` syntax
    # ... later ...
    status, note = check_approval(aid)
    # status in {'pending', 'yes', 'no', 'skip'}
"""

from __future__ import annotations

import sqlite3
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

DB_PATH = Path('/home/kanoonth-ai/projects/caffe-stocks/data/ml-feedback.db')

SCHEMA = """
CREATE TABLE IF NOT EXISTS ml_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    note TEXT,
    created_at TEXT NOT NULL,
    answered_at TEXT
)
"""

CONF_PATHS = [
    Path.home() / 'projects' / 'ops' / 'telegram' / 'telegram.conf',
    Path.home() / 'kanoonth' / 'scripts' / 'telegram.conf',
]


def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def _telegram_credentials():
    for p in CONF_PATHS:
        if not p.exists():
            continue
        bot_token = chat_id = None
        with open(p) as f:
            for line in f:
                s = line.strip()
                if s.startswith('TELEGRAM_BOT_TOKEN='):
                    bot_token = s.split('=', 1)[1].strip().strip('"\'')
                elif s.startswith('TELEGRAM_CHAT_ID='):
                    chat_id = s.split('=', 1)[1].strip().strip('"\'')
        if bot_token and chat_id:
            return bot_token, chat_id
    raise FileNotFoundError(
        f'No telegram.conf found in: {[str(p) for p in CONF_PATHS]}'
    )


def create_approval(topic: str, question: str) -> int:
    with _conn() as conn:
        cur = conn.execute(
            'INSERT INTO ml_approvals (topic, question, status, created_at) '
            "VALUES (?, ?, 'pending', ?)",
            (topic, question, datetime.now().isoformat(timespec='seconds')),
        )
        conn.commit()
        return cur.lastrowid


def check_approval(approval_id: int) -> tuple[str, str | None]:
    with _conn() as conn:
        r = conn.execute(
            'SELECT status, note FROM ml_approvals WHERE id = ?',
            (approval_id,),
        ).fetchone()
    if not r:
        raise KeyError(f'ml_approvals #{approval_id} not found')
    return r['status'], r['note']


def send_ask(approval_id: int, header: str | None = None) -> None:
    """Post the approval ask to Telegram with the `ml <id> ...` reply syntax."""
    with _conn() as conn:
        r = conn.execute(
            'SELECT topic, question FROM ml_approvals WHERE id = ?',
            (approval_id,),
        ).fetchone()
    if not r:
        raise KeyError(f'ml_approvals #{approval_id} not found')
    parts = []
    if header:
        parts.append(header)
    parts.append(f'ml #{approval_id} — {r["topic"]}')
    parts.append('')
    parts.append(r['question'])
    parts.append('')
    parts.append(
        f'Reply: `ml {approval_id} yes` / `ml {approval_id} no` / '
        f'`ml {approval_id} skip` (optional [note])'
    )
    msg = '\n'.join(parts)
    bot_token, chat_id = _telegram_credentials()
    data = urllib.parse.urlencode({
        'chat_id': chat_id,
        'text': msg,
        'disable_web_page_preview': 'true',
    }).encode()
    urllib.request.urlopen(
        f'https://api.telegram.org/bot{bot_token}/sendMessage',
        data=data, timeout=10,
    )


def _cli():
    """Tiny CLI for manual use:
       ml_approval.py create <topic> <question>
       ml_approval.py show   <id>
       ml_approval.py send   <id>
    """
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == 'create' and len(sys.argv) >= 4:
        aid = create_approval(sys.argv[2], ' '.join(sys.argv[3:]))
        print(aid)
    elif cmd == 'show' and len(sys.argv) >= 3:
        with _conn() as conn:
            r = conn.execute(
                'SELECT * FROM ml_approvals WHERE id = ?', (int(sys.argv[2]),)
            ).fetchone()
        print(dict(r) if r else 'not found')
    elif cmd == 'send' and len(sys.argv) >= 3:
        send_ask(int(sys.argv[2]))
        print('sent')
    else:
        print('Usage: create <topic> <question> | show <id> | send <id>')


if __name__ == '__main__':
    _cli()
