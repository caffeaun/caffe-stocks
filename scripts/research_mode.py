#!/usr/bin/env python3
"""Research mode — daily research stage.

Invokes `claude -p` with the research-mode prompt; expects a single markdown
brief to be written under data/research_briefs/. Validates structure, commits
the brief, then auto-invokes scripts/ml_scaffold.py to do the mechanical work
(deps + scaffolds).

Wall-time: 60 min. Shorter than claude_mode (which gates) — research is
mostly web traffic + writing.
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

LOCK_PATH = BASE / 'models' / '.ml-loop.lock'
PROMPT_BUILDER = BASE / 'scripts' / 'research_prompt_builder.py'
SCAFFOLD = BASE / 'scripts' / 'ml_scaffold.py'
CLAUDE_BIN = Path(os.path.expanduser('~/.local/bin/claude'))
LOG_DIR = BASE / 'logs'
BRIEFS_DIR = BASE / 'data' / 'research_briefs'
LATEST_LINK = BRIEFS_DIR / 'latest.md'

HARD_TIMEOUT = 3300  # 55 min Claude wall, leave 5 min for scaffold


REQUIRED_HEADERS = [
    '## Registry-bias diagnosis',
    '## Recommended technique',
    '### Why now',
    '### Deps',
    '### Scaffold code',
    '### Registry registration',
    '### HP search space',
    '## Recommended next action',
    '## Sources',
]


def acquire_lock(path: Path) -> int:
    """Acquire shared lock. If held by train_mode, wait; if claude_mode, exit
    (claude_mode is more time-sensitive)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    for attempt in range(60):
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.lseek(fd, 0, 0)
            os.ftruncate(fd, 0)
            os.write(fd, f'{os.getpid()} research_mode '
                         f'{datetime.now().isoformat()}\n'.encode())
            os.fsync(fd)
            return fd
        except BlockingIOError:
            os.lseek(fd, 0, 0)
            holder = os.read(fd, 1024).decode(errors='replace').strip()
            parts = holder.split()
            if len(parts) >= 2 and parts[1] == 'claude_mode':
                print(f'Lock held by claude_mode — research mode yields. {holder!r}')
                os.close(fd)
                sys.exit(0)
            time.sleep(2)
    print('Lock contention after 120s, exiting.', file=sys.stderr)
    os.close(fd)
    sys.exit(0)


def release_lock(fd: int):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def parse_research_report(stdout: str) -> Optional[dict]:
    """Extract the last ```json``` block containing 'brief_path'."""
    blocks = re.findall(r'```json\s*\n(.*?)\n\s*```', stdout, re.DOTALL)
    for block in reversed(blocks):
        try:
            data = json.loads(block)
            if 'brief_path' in data:
                return data
        except json.JSONDecodeError:
            continue
    return None


def validate_brief(brief_path: Path) -> tuple[bool, list[str]]:
    """Confirm the brief has all required headers + parseable code blocks.
    Returns (ok, list_of_missing_or_problems)."""
    if not brief_path.exists():
        return False, [f'brief file does not exist: {brief_path}']
    text = brief_path.read_text(errors='replace')
    missing = [h for h in REQUIRED_HEADERS if h not in text]
    problems = list(missing)

    # Sanity: at least one ```python``` block and one ```bash``` block
    if '```python' not in text:
        problems.append('no ```python``` code block found')
    if '```bash' not in text:
        problems.append('no ```bash``` block (for pip install) found')

    return len(problems) == 0, problems


def update_latest_symlink(brief_path: Path):
    """latest.md mirrors today's brief. Use a copy (not symlink) so the
    file is committable as a stable path across days."""
    if brief_path.exists():
        LATEST_LINK.write_text(brief_path.read_text())


def commit_brief(brief_path: Path, technique: str) -> Optional[str]:
    """Auto-commit the brief + latest.md."""
    paths = [
        str(brief_path.relative_to(BASE)),
        str(LATEST_LINK.relative_to(BASE)),
    ]
    add = subprocess.run(
        ['git', '-C', str(BASE), 'add', '--'] + paths,
        capture_output=True, text=True,
    )
    if add.returncode != 0:
        print(f'commit: git add failed: {add.stderr.strip()}', file=sys.stderr)
        return None
    diff = subprocess.run(
        ['git', '-C', str(BASE), 'diff', '--cached', '--quiet'],
        capture_output=True,
    )
    if diff.returncode == 0:
        print('commit: nothing staged')
        return None
    msg = f'research: {datetime.now().strftime("%Y-%m-%d")} — {technique[:80]}'
    commit = subprocess.run(
        ['git', '-C', str(BASE), 'commit', '-m', msg],
        capture_output=True, text=True,
    )
    if commit.returncode != 0:
        print(f'commit: failed: {commit.stderr.strip()}', file=sys.stderr)
        return None
    sha = subprocess.run(
        ['git', '-C', str(BASE), 'rev-parse', '--short', 'HEAD'],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f'committed: {sha} - {msg}')
    return sha


def telegram(text: str):
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
    try:
        import urllib.request, urllib.parse
        data = urllib.parse.urlencode({
            'chat_id': chat_id, 'text': text[:4000],
            'disable_web_page_preview': 'true',
        }).encode()
        urllib.request.urlopen(
            f'https://api.telegram.org/bot{bot_token}/sendMessage',
            data=data, timeout=10)
    except Exception as e:
        print(f'Telegram alert failed: {e}', file=sys.stderr)


def invoke_scaffold():
    """Auto-invoke ml_scaffold.py. Its own failures are reported separately
    (Telegram in ml_scaffold itself); we just exec it and surface the exit
    code in research_mode's log."""
    if not SCAFFOLD.exists():
        print('ml_scaffold.py not present — skipping auto-scaffold')
        return
    venv_py = BASE / 'venv' / 'bin' / 'python'
    print(f'invoking {SCAFFOLD}')
    r = subprocess.run(
        [str(venv_py), str(SCAFFOLD)],
        cwd=str(BASE),
        env={**os.environ, 'PYTHONPATH': str(BASE)},
        timeout=600,
    )
    print(f'ml_scaffold exit={r.returncode}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Build prompt and exit; do not invoke claude')
    parser.add_argument('--skip-scaffold', action='store_true',
                        help='Skip auto-invocation of ml_scaffold.py (testing)')
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)

    venv_py = BASE / 'venv' / 'bin' / 'python'
    prompt = subprocess.check_output(
        [str(venv_py), str(PROMPT_BUILDER)],
        cwd=str(BASE),
        env={**os.environ, 'PYTHONPATH': str(BASE)},
        timeout=30,
    ).decode()

    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    prompt_path = LOG_DIR / f'research-prompt-{ts}.txt'
    prompt_path.write_text(prompt)
    print(f'Prompt: {prompt_path} ({len(prompt)} chars)')

    if args.dry_run:
        print('--dry-run set — exiting without invoking claude')
        return

    if not CLAUDE_BIN.exists():
        print(f'ERROR: claude binary not found at {CLAUDE_BIN}', file=sys.stderr)
        sys.exit(2)

    fd = acquire_lock(LOCK_PATH)
    try:
        telegram('🔬 ML research_mode starting')
        env = os.environ.copy()
        env.pop('CLAUDECODE', None)
        env['PYTHONPATH'] = str(BASE)
        env['PATH'] = (f"{Path.home()}/.local/bin:"
                       f"{BASE}/venv/bin:" + env.get('PATH', ''))

        log_path = LOG_DIR / f'research-mode-{ts}.log'
        t0 = time.time()
        proc = subprocess.run(
            [str(CLAUDE_BIN), '-p',
             '--model', 'claude-opus-4-7',
             '--effort', 'max',
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
        log_path.write_text(
            f'EXIT: {proc.returncode}\nELAPSED: {elapsed}s\n\n'
            f'=== STDOUT ===\n{proc.stdout}\n\n=== STDERR ===\n{proc.stderr}\n')
        print(f'Claude exit={proc.returncode}, log={log_path}')

        if proc.returncode != 0:
            telegram(f'⚠️ research_mode exit={proc.returncode}, see {log_path.name}')
            return

        report = parse_research_report(proc.stdout)
        if not report:
            telegram(f'⚠️ research_mode produced no JSON report. {log_path.name}')
            return

        brief_rel = report.get('brief_path', '')
        if not brief_rel:
            telegram('⚠️ research_mode JSON missing brief_path')
            return
        brief_path = BASE / brief_rel

        if report.get('registry_key') is None:
            telegram(f'ℹ️ research_mode: no credible candidate today. '
                     f'Brief: {brief_path.name}')
            update_latest_symlink(brief_path)
            commit_brief(brief_path, technique=report.get('technique', 'NONE'))
            return

        ok, problems = validate_brief(brief_path)
        if not ok:
            telegram(f'⚠️ research_mode brief malformed: {", ".join(problems[:3])}')
            print(f'Validation problems: {problems}', file=sys.stderr)
            return

        update_latest_symlink(brief_path)
        sha = commit_brief(brief_path, technique=report.get('technique', 'unnamed'))
        telegram(
            f'🔬 *Research brief* {datetime.now():%Y-%m-%d}\n'
            f'Technique: `{report.get("technique", "?")}`\n'
            f'Registry key: `{report.get("registry_key", "?")}`\n'
            f'Deps: {", ".join(report.get("deps", []))}\n'
            f'Sources: {report.get("sources_count", "?")} cited\n'
            f'git: {sha}\n'
            f'{(report.get("summary") or "")[:300]}'
        )

        if not args.skip_scaffold:
            invoke_scaffold()

    except subprocess.TimeoutExpired:
        telegram(f'⚠️ research_mode timed out at {HARD_TIMEOUT}s')
    finally:
        release_lock(fd)


if __name__ == '__main__':
    main()
