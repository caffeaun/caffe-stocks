#!/usr/bin/env python3
"""Claude mode — orchestrate ONE structural-change iteration.

Acquires the shared lock with priority (kills running train_mode if needed),
builds a prompt via prompt_builder.py, invokes `claude -p` with WebSearch /
WebFetch / Bash / Edit / Write tools, parses the structured JSON report
from Claude's output, records it in the feedback DB.

Hard wall-time: 90 min inner subprocess timeout (HARD_TIMEOUT below),
with a 120 min outer kill by `timeout` in scripts/ml_loop.sh — the
30 min gap lets the inner TimeoutExpired handler send a Telegram
alert and exit cleanly before the outer SIGTERM lands.
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

from scripts import active_case, feedback as fb
from scripts.return_gate import is_candidate, avg_annualized_return

LOCK_PATH = BASE / 'models' / '.ml-loop.lock'
PROMPT_BUILDER = BASE / 'scripts' / 'prompt_builder.py'
CLAUDE_BIN = Path(os.path.expanduser('~/.local/bin/claude'))
LOG_DIR = BASE / 'logs'
HARD_TIMEOUT = 5400  # 90 min inner subprocess; outer ml_loop.sh kill = 7200s (30 min headroom for Telegram alert + cleanup)


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


def commit_iteration(iter_id: int, trainer_name: str, gate_passed: bool,
                     hypothesis: Optional[str] = None) -> Optional[str]:
    """Commit claude_mode's source-edit changes after a successful iteration.

    The whole point: prior trainers stay in git history so future iterations
    can recover them via `git log -- models/trainers.py` even if a later run
    rewrites the file. Self-healing — no manual restore needed.

    Stages an explicit allowlist of paths (the surface area Claude is
    contracted to modify in prompt_builder.py). New helper modules under
    models/ are also caught via the *.py glob. Skips silently if there's
    nothing staged. Wrapped by the caller in try/except so a failure here
    never tanks the iteration record.

    Returns the short commit SHA on success, None otherwise.
    """
    paths = [
        'models/trainers.py',
        'models/search_spaces.py',
        'models/feature_eng.py',
        'models/labels.py',
        'models/sequence_loader.py',
        'models/active_case.json',
        'scripts/return_gate.py',
    ]
    # Also pick up any new .py files in models/ (a helper module Claude added)
    extra_py = [str(p.relative_to(BASE)) for p in (BASE / 'models').glob('*.py')
                if p.name not in ('__init__.py',) and str(p.relative_to(BASE)) not in paths]
    paths.extend(extra_py)

    add = subprocess.run(
        ['git', '-C', str(BASE), 'add', '--'] + paths,
        capture_output=True, text=True,
    )
    if add.returncode != 0:
        print(f'auto-commit: git add failed: {add.stderr.strip()}', file=sys.stderr)
        return None

    # Anything actually staged?
    diff = subprocess.run(
        ['git', '-C', str(BASE), 'diff', '--cached', '--quiet'],
        capture_output=True,
    )
    if diff.returncode == 0:
        print('auto-commit: no changes to commit')
        return None

    flag = 'PASS' if gate_passed else 'FAIL'
    msg_lines = [f'claude iter #{iter_id}: {trainer_name} ({flag})']
    if hypothesis:
        msg_lines.extend(['', hypothesis.strip()[:300]])
    msg = '\n'.join(msg_lines)

    commit = subprocess.run(
        ['git', '-C', str(BASE), 'commit', '-m', msg],
        capture_output=True, text=True,
    )
    if commit.returncode != 0:
        print(f'auto-commit: git commit failed: {commit.stderr.strip()}', file=sys.stderr)
        return None

    sha = subprocess.run(
        ['git', '-C', str(BASE), 'rev-parse', '--short', 'HEAD'],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f'auto-commit: {sha} - {msg_lines[0]}')
    return sha


def detect_brief_deviation(report: dict) -> Optional[str]:
    """Decide if claude's report deviates from the active research brief.

    Returns a deviation reason string when the iteration should be rejected
    (revert changes, no active_case update). Returns None when the report
    is compliant — either matches the brief, legitimately marks it
    exhausted, or no brief exists.

    Compliance rules (mirroring §6.B.0 in the prompt):
      - No brief on disk → compliant (full discretion).
      - Brief present and trainer == registry_key → compliant.
      - Brief present, trainer != registry_key, AND `exhausted_brief: true`
        AND DB confirms ≥2 prior claude attempts of registry_key with
        best wp < 3 → compliant.
      - Anything else → deviation.

    The exhaustion check is duplicated from prompt_builder._brief_directive
    (intentional — keep the enforcement self-contained).
    """
    brief_path = BASE / 'data' / 'research_briefs' / 'latest.md'
    if not brief_path.exists():
        return None
    text = brief_path.read_text(errors='replace')
    m = re.search(
        r'### Registry registration\s*\n```python\s*\n(.*?)\n\s*```',
        text, re.DOTALL,
    )
    if not m:
        return None  # unparseable brief — fail open
    m2 = re.search(r"['\"](\w+)['\"]\s*:\s*\w+", m.group(1))
    if not m2:
        return None
    registry_key = m2.group(1)

    picked = report.get('trainer', '')

    # Phase 3C archive check: any pick of an archived trainer is a
    # deviation regardless of brief alignment.
    try:
        from models.trainers_archive import is_archived, archive_notice
        if is_archived(picked):
            return f'archived trainer pick: {archive_notice(picked)}'
    except Exception:
        pass

    if picked == registry_key:
        return None  # following the order

    # Trainer differs — check if exhaustion claim is legitimate.
    # Phase 3A bumped thresholds to (≥5 attempts, best wp <5) + AMP
    # (best wp ≥4 forces ablation, no pivot allowed regardless of attempts).
    claims_exhausted = bool(report.get('exhausted_brief'))
    fb.init_db()
    with fb.get_conn() as conn:
        rows = conn.execute(
            "SELECT windows_passed FROM iterations "
            "WHERE mode='claude' AND trainer = ? AND contaminated = 0 "
            "ORDER BY id DESC LIMIT 10",
            (registry_key,),
        ).fetchall()
    attempts = len(rows)
    best_wp = max((r['windows_passed'] for r in rows), default=0)
    amp_locked = best_wp >= 4
    db_exhausted = attempts >= 5 and best_wp < 5

    if amp_locked:
        return (
            f'trainer={picked!r} != brief.registry_key={registry_key!r}; '
            f'AMP locked (best_wp={best_wp} ≥4 — ablation required, pivot '
            f'forbidden)'
        )

    if claims_exhausted and db_exhausted:
        return None  # legitimate pivot — brief is dead

    return (
        f'trainer={picked!r} != brief.registry_key={registry_key!r}; '
        f'exhausted_brief={claims_exhausted}, DB attempts={attempts}, '
        f'best_wp={best_wp} (needs ≥5 attempts AND best wp<5)'
    )


def revert_claude_edits(paths: list[str]) -> bool:
    """Discard claude's working-tree changes to allowlisted files via
    `git checkout HEAD --`. Surgical — doesn't touch unrelated paths.
    Returns True on success."""
    r = subprocess.run(
        ['git', '-C', str(BASE), 'checkout', 'HEAD', '--'] + paths,
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f'revert: git checkout failed: {r.stderr.strip()}', file=sys.stderr)
        return False
    return True


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

    streak = fb.consecutive_failures()  # informational only; no longer gates the run

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

        # Authoritatively re-evaluate the model-level gate from the per-window
        # data. Claude's reported gate_passed is advisory only — we own the
        # decision to mark something a candidate.
        results = gate_result.get('results') or []
        if results:
            prior_best_ann = fb.best_candidate_ann_return()
            gate_result['gate_passed'] = is_candidate(results, prior_best_ann)
            gate_result['avg_annualized_return'] = avg_annualized_return(results)
            gate_result['prior_best_ann_return'] = prior_best_ann

        # Detect deviation from the research-brief order BEFORE recording
        # active_case or committing. If claude went off-script:
        #  - record the iteration (truth-keeping; the gate result is still
        #    useful evidence for tomorrow's research_mode)
        #  - revert claude's code changes (don't pollute the registry)
        #  - skip active_case update (train mode keeps sweeping prior case)
        #  - Telegram alert (operator visibility)
        deviation_reason = detect_brief_deviation(report)

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

        if deviation_reason:
            # Revert the working-tree changes claude made (same allowlist
            # as commit_iteration) so the next 3-hourly run starts clean
            # and the deviant trainer doesn't pollute the registry.
            paths_to_revert = [
                'models/trainers.py',
                'models/search_spaces.py',
                'models/feature_eng.py',
                'models/labels.py',
                'models/sequence_loader.py',
                'models/active_case.json',
                'scripts/return_gate.py',
            ]
            revert_ok = revert_claude_edits(paths_to_revert)
            print(f'DEVIATION DETECTED: {deviation_reason}')
            print(f'reverted={revert_ok}; active_case NOT updated; no commit')
            telegram(
                f'⛔ *Claude iter #{iter_id} DEVIATED*\n'
                f'`{report.get("trainer", "?")}` (gate ignored)\n'
                f'Reason: {deviation_reason}\n'
                f'Reverted code changes={"yes" if revert_ok else "FAILED"}; '
                f'active_case kept; brief still active for next run.\n'
                f'See {log_path.name}'
            )
            # Skip active_case update and auto-commit for deviations.
            return

        # Designate the next experimental case for train mode to sweep within.
        # Falls back to inferring from report['trainer'] + defaults when the
        # report doesn't include an explicit active_case block.
        try:
            new_case = active_case.from_report(report, claude_iter_id=iter_id)
            if new_case is not None:
                ac_path = active_case.write(new_case)
                print(f'active_case → {new_case.get("trainer")} '
                      f'(written to {ac_path})')
            else:
                print('active_case: skipped (report had no trainer field)')
        except Exception as e:
            # Never let a bad active_case write tank the iteration record.
            print(f'active_case write failed: {e!r}', file=sys.stderr)

        # Auto-commit source changes so prior trainers stay in git history
        # — future iterations can recover any class via `git log` / `git show`
        # even if a later run rewrites trainers.py.
        commit_sha = None
        try:
            commit_sha = commit_iteration(
                iter_id=iter_id,
                trainer_name=report.get('trainer', 'unknown'),
                gate_passed=bool(gate_result.get('gate_passed')),
                hypothesis=report.get('hypothesis'),
            )
        except Exception as e:
            print(f'auto-commit failed: {e!r}', file=sys.stderr)

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
        sha_line = f'\ngit: {commit_sha}' if commit_sha else ''
        telegram(
            f'{flag} *Claude iter #{iter_id}* `{report.get("trainer", "?")}`{sha_line}\n'
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
