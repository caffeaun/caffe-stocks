#!/usr/bin/env python3
"""Claude mode — orchestrate ONE structural-change iteration.

Acquires the shared lock with priority (kills running train_mode if needed),
builds a prompt via prompt_builder.py, invokes `claude -p` with WebSearch /
WebFetch / Bash / Edit / Write tools, parses the structured JSON report
from Claude's output, records it in the feedback DB.

Hard wall-time: 30 min. After that the process is killed by `timeout`
in scripts/ml_loop.sh.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
sys.path.insert(0, str(BASE))

from scripts import feedback as fb

LOCK_PATH = BASE / 'models' / '.ml-loop.lock'
PROMPT_BUILDER = BASE / 'scripts' / 'prompt_builder.py'
CLAUDE_BIN = Path(os.path.expanduser('~/.local/bin/claude'))
LOG_DIR = BASE / 'logs'
HARD_TIMEOUT = 1700  # leave 100s headroom under the cron timeout 1800s


def acquire_lock_with_priority(path: Path):
    """Acquire the lock. If held by train_mode, kill it. Returns the fd."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Try once non-blocking
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, f'{os.getpid()} claude_mode {datetime.now().isoformat()}\n'.encode())
        os.fsync(fd)
        return fd
    except BlockingIOError:
        pass

    # Locked. Read holder info.
    os.lseek(fd, 0, 0)
    holder = os.read(fd, 1024).decode(errors='replace').strip()
    print(f'Lock held by: {holder!r} — preempting if it is train_mode')
    parts = holder.split()
    if len(parts) >= 2 and parts[1] == 'train_mode':
        try:
            held_pid = int(parts[0])
            os.kill(held_pid, signal.SIGTERM)
            print(f'  sent SIGTERM to PID {held_pid}, waiting up to 10s')
            for _ in range(20):
                time.sleep(0.5)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.lseek(fd, 0, 0)
                    os.ftruncate(fd, 0)
                    os.write(fd, f'{os.getpid()} claude_mode {datetime.now().isoformat()}\n'.encode())
                    os.fsync(fd)
                    print('  acquired after train_mode released')
                    return fd
                except BlockingIOError:
                    continue
            os.kill(held_pid, signal.SIGKILL)
            time.sleep(0.5)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.lseek(fd, 0, 0)
            os.ftruncate(fd, 0)
            os.write(fd, f'{os.getpid()} claude_mode {datetime.now().isoformat()}\n'.encode())
            return fd
        except (ValueError, ProcessLookupError):
            pass

    # Held by another claude (or unknown) — exit
    os.close(fd)
    print('Lock held by another claude_mode or unknown holder. Exiting.')
    sys.exit(0)


def release_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def parse_claude_report(stdout: str) -> Optional[dict]:
    """Extract the last ```json``` block from Claude's output and parse it.
    Falls back to the last bare JSON object containing 'gate_result'."""
    blocks = re.findall(r'```json\s*\n(.*?)\n\s*```', stdout, re.DOTALL)
    for block in reversed(blocks):
        try:
            data = json.loads(block)
            if 'gate_result' in data or 'gate_passed' in data:
                return data
        except json.JSONDecodeError:
            continue

    # Fallback: any object with a gate_result key
    for match in re.finditer(r'\{[^{}]*"gate_result"[^{}]*\{.*?\}\s*[^{}]*\}', stdout, re.DOTALL):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue

    return None


def telegram(text: str):
    """Best-effort Telegram notify via the existing telegram.conf."""
    conf_paths = [
        BASE.parent / 'ops' / 'telegram' / 'telegram.conf',
        Path.home() / 'kanoonth' / 'scripts' / 'telegram.conf',
    ]
    bot_token = chat_id = None
    for p in conf_paths:
        if p.exists():
            with open(p) as f:
                for line in f:
                    if line.strip().startswith('TELEGRAM_BOT_TOKEN='):
                        bot_token = line.split('=', 1)[1].strip().strip('"\'')
                    elif line.strip().startswith('TELEGRAM_CHAT_ID='):
                        chat_id = line.split('=', 1)[1].strip().strip('"\'')
            break
    if not (bot_token and chat_id):
        return
    # Send as plain text — Claude's lessons often contain stray * _ ` chars that
    # break Markdown parse mode (HTTP 400 entity parsing errors).
    try:
        import urllib.request, urllib.parse
        data = urllib.parse.urlencode({
            'chat_id': chat_id,
            'text': text[:4000],
            'disable_web_page_preview': 'true',
        }).encode()
        urllib.request.urlopen(
            f'https://api.telegram.org/bot{bot_token}/sendMessage',
            data=data, timeout=10)
    except Exception as e:
        print(f'Telegram alert failed: {e}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-prompt-bytes', type=int, default=65536)
    parser.add_argument('--dry-run', action='store_true',
                        help='Build prompt and exit; do not invoke claude')
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fb.init_db()

    streak = fb.consecutive_failures()
    if streak >= 5 and not args.dry_run:
        msg = f'AUTO-PAUSED: {streak} consecutive failures. `t resume-ml` to clear.'
        print(msg)
        telegram(f'⏸ ML loop paused — {streak} consecutive failures.')
        sys.exit(0)

    # Build prompt
    venv_py = BASE / 'venv' / 'bin' / 'python'
    prompt = subprocess.check_output(
        [str(venv_py), str(PROMPT_BUILDER)],
        cwd=str(BASE),
        env={**os.environ, 'PYTHONPATH': str(BASE)},
        timeout=30,
    ).decode()

    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    prompt_path = LOG_DIR / f'claude-prompt-{ts}.txt'
    prompt_path.write_text(prompt)
    print(f'Prompt written: {prompt_path} ({len(prompt)} chars)')

    if args.dry_run:
        print('--dry-run set — exiting without invoking claude')
        return

    if not CLAUDE_BIN.exists():
        print(f'ERROR: claude binary not found at {CLAUDE_BIN}', file=sys.stderr)
        sys.exit(2)

    # Acquire lock — preempt train_mode if needed
    fd = acquire_lock_with_priority(LOCK_PATH)

    try:
        # Telegram start alert
        telegram(f'🤖 ML claude_mode starting (streak={streak} fails)')

        env = os.environ.copy()
        env.pop('CLAUDECODE', None)
        env['PYTHONPATH'] = str(BASE)
        env['PATH'] = (f"{Path.home()}/.local/bin:"
                       f"{BASE}/venv/bin:" + env.get('PATH', ''))

        log_path = LOG_DIR / f'claude-mode-{ts}.log'
        started = datetime.now().isoformat()
        t0 = time.time()

        proc = subprocess.run(
            [str(CLAUDE_BIN), '-p',
             '--model', 'claude-opus-4-7',
             '--dangerously-skip-permissions',
             '--allowedTools', 'Bash Edit Write Read Glob Grep WebSearch WebFetch'],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=HARD_TIMEOUT,
            env=env,
            cwd=str(BASE),
        )
        elapsed = int(time.time() - t0)
        finished = datetime.now().isoformat()

        # Save full output
        log_path.write_text(
            f'EXIT: {proc.returncode}\nELAPSED: {elapsed}s\n\n'
            f'=== STDOUT ===\n{proc.stdout}\n\n=== STDERR ===\n{proc.stderr}\n')
        print(f'Claude exit={proc.returncode}, log={log_path}')

        if proc.returncode != 0:
            telegram(f'⚠️ Claude mode exit={proc.returncode}, see {log_path.name}')
            return

        # Parse structured report
        report = parse_claude_report(proc.stdout)
        if not report:
            telegram(f'⚠️ Claude mode produced no JSON report. See {log_path.name}')
            return

        gate_result = report.get('gate_result') or {
            'gate_passed': bool(report.get('gate_passed')),
            'windows_passed': report.get('windows_passed', 0),
            'windows_total': report.get('windows_total', 0),
            'results': [],
        }

        iter_id = fb.record_iteration(
            mode='claude',
            trainer=report.get('trainer', 'unknown'),
            hyperparams=report.get('hyperparams') or {},
            gate_result=gate_result,
            model_dir=str(BASE / 'models' / 'claude-mode' / ts),
            started_at=started,
            finished_at=finished,
            elapsed_seconds=elapsed,
            code_changes=report.get('code_changes'),
            hypothesis=report.get('hypothesis'),
            backbone=report.get('backbone') or None,
            lessons=report.get('lessons'),
        )

        # Optional: log a data request if Claude submitted one
        dr = report.get('data_request') or ''
        if dr.strip():
            fb.log_data_request(iter_id, dr.strip())

        # Telegram result
        flag = '✅' if gate_result.get('gate_passed') else '❌'
        wp = gate_result.get('windows_passed', 0)
        wt = gate_result.get('windows_total', 0)
        ann = (gate_result.get('avg_annualized_return')
               or report.get('avg_annualized_return') or 0)
        wr = (gate_result.get('avg_win_rate')
               or report.get('avg_win_rate') or 0)
        telegram(
            f'{flag} *Claude iter #{iter_id}* `{report.get("trainer", "?")}`\n'
            f'{report.get("hypothesis", "")[:200]}\n'
            f'wp={wp}/{wt}  ann={ann:+.1%}  wr={wr:.1%}  '
            f'({elapsed}s)\n'
            f'lessons: {(report.get("lessons") or "")[:300]}'
        )

        if dr.strip():
            telegram(f'💡 *ML data request* (iter #{iter_id})\n{dr[:1000]}')

    except subprocess.TimeoutExpired:
        telegram(f'⚠️ Claude mode timed out at {HARD_TIMEOUT}s')
    finally:
        release_lock(fd)


if __name__ == '__main__':
    main()
