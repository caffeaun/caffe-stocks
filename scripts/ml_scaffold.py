#!/usr/bin/env python3
"""ml_scaffold — deterministic post-script for research_mode briefs.

Reads data/research_briefs/latest.md and:
  1. Runs the `pip install ...` line from `### Deps`
  2. Downloads weights via the command in `### Weights` (if any)
  3. Appends the `### Scaffold code` Python block to models/trainers.py
  4. Appends the `### Registry registration` line to the TRAINERS dict
  5. Appends the `### HP search space` block to models/search_spaces.py
  6. Validates: `python -c "from models.trainers import TRAINERS; assert 'X' in TRAINERS"`
  7. Auto-commits as `scaffold: <technique>`

Everything is idempotent. Re-running on the same brief is a no-op if the
registry key already exists.

Wall-time: 10 min (mostly pip install).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
LATEST_BRIEF = BASE / 'data' / 'research_briefs' / 'latest.md'
TRAINERS_PY = BASE / 'models' / 'trainers.py'
SEARCH_SPACES_PY = BASE / 'models' / 'search_spaces.py'
VENV_PIP = BASE / 'venv' / 'bin' / 'pip'
VENV_PY = BASE / 'venv' / 'bin' / 'python'

# Trainers dict sentinel — the line we anchor inserts to. Confirmed
# in models/trainers.py via `grep -n '^TRAINERS = {'`.
TRAINERS_ANCHOR = re.compile(r'^TRAINERS\s*=\s*\{\s*$', re.MULTILINE)


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


def _block_after_header(text: str, header: str, fence: str) -> Optional[str]:
    """Find the first fenced code block of language `fence` AFTER `header`.
    Returns the inner contents (without fences) or None."""
    idx = text.find(header)
    if idx < 0:
        return None
    rest = text[idx + len(header):]
    pattern = re.compile(rf'```{fence}\s*\n(.*?)\n\s*```', re.DOTALL)
    m = pattern.search(rest)
    return m.group(1) if m else None


def _find_section_text(text: str, header: str, next_headers: list[str]) -> Optional[str]:
    """Plain text between `header` and the next `## ` / `### ` header."""
    idx = text.find(header)
    if idx < 0:
        return None
    rest = text[idx + len(header):]
    end = len(rest)
    for h in next_headers + ['\n## ', '\n### ']:
        i = rest.find(h)
        if i >= 0 and i < end:
            end = i
    return rest[:end].strip()


def parse_brief(text: str) -> dict:
    """Extract the seven fields the scaffold needs from a brief."""
    out: dict = {}

    # Technique name — first line after "## Recommended technique:"
    m = re.search(r'^## Recommended technique:\s*(.+?)$', text, re.MULTILINE)
    out['technique'] = m.group(1).strip() if m else 'unnamed'

    if out['technique'].upper() == 'NONE':
        out['skip'] = True
        return out
    out['skip'] = False

    # Deps — bash block right after "### Deps"
    deps_block = _block_after_header(text, '### Deps', 'bash')
    if deps_block:
        pip_lines = [l.strip() for l in deps_block.splitlines()
                      if l.strip().startswith('pip install')]
        out['pip_command'] = pip_lines[0] if pip_lines else None
    else:
        out['pip_command'] = None

    # Weights — optional, free-text section
    weights_text = _find_section_text(text, '### Weights',
                                       ['### Scaffold code'])
    out['weights_command'] = None
    if weights_text:
        # Take the first non-comment, non-empty line; skip "N/A" lines
        for line in weights_text.splitlines():
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('<') or 'N/A' in s.upper():
                continue
            # If it's inside a code fence, strip the fence
            if s.startswith('```'):
                continue
            out['weights_command'] = s
            break

    # Scaffold code — python block after "### Scaffold code"
    out['scaffold_code'] = _block_after_header(text, '### Scaffold code', 'python')

    # Registry registration — python block after "### Registry registration"
    out['registry_line'] = _block_after_header(text, '### Registry registration', 'python')

    # HP search space — python block after "### HP search space"
    out['search_space_block'] = _block_after_header(text, '### HP search space', 'python')

    # Registry key — extract from registry_line ('key': ClassName,)
    if out['registry_line']:
        m = re.search(r"['\"](\w+)['\"]\s*:\s*\w+Trainer", out['registry_line'])
        out['registry_key'] = m.group(1) if m else None
    else:
        out['registry_key'] = None

    return out


def already_registered(registry_key: str) -> bool:
    """Quick check: does trainers.py already contain this registry key?"""
    if not registry_key:
        return False
    text = TRAINERS_PY.read_text()
    return bool(re.search(rf"['\"]\s*{re.escape(registry_key)}\s*['\"]\s*:",
                          text))


def run_pip(pip_command: str) -> bool:
    """Execute pip install using the venv pip. Returns True on success."""
    if not pip_command:
        print('No pip command found in brief; skipping install step.')
        return True
    parts = pip_command.split()
    if parts[:2] != ['pip', 'install']:
        print(f'Unexpected pip command shape: {pip_command!r}', file=sys.stderr)
        return False
    cmd = [str(VENV_PIP)] + parts[1:]
    print(f'$ {" ".join(cmd)}')
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    sys.stdout.write(r.stdout[-2000:])
    sys.stderr.write(r.stderr[-2000:])
    if r.returncode != 0:
        print(f'pip install FAILED (exit {r.returncode})', file=sys.stderr)
        return False
    print('pip install OK')
    return True


def run_weights(weights_command: str) -> bool:
    """Best-effort weight download. Non-fatal on failure — the trainer may
    fall back to lazy on-demand download at training time."""
    if not weights_command:
        return True
    print(f'$ {weights_command}')
    r = subprocess.run(weights_command, shell=True, capture_output=True,
                        text=True, timeout=1200)
    sys.stdout.write(r.stdout[-2000:])
    sys.stderr.write(r.stderr[-2000:])
    if r.returncode != 0:
        print(f'WARN: weights download exited {r.returncode}; trainer may '
              f'need to lazy-load at training time.', file=sys.stderr)
        return False
    return True


def insert_scaffold(scaffold_code: str, registry_line: str) -> bool:
    """Append the scaffold code BEFORE the TRAINERS dict line; append the
    registry line inside the dict, before the closing '}'. Idempotent via
    already_registered()."""
    text = TRAINERS_PY.read_text()

    # Anchor: the `TRAINERS = {` line
    m = TRAINERS_ANCHOR.search(text)
    if not m:
        print('FATAL: TRAINERS_ANCHOR not found in trainers.py', file=sys.stderr)
        return False
    anchor_pos = m.start()

    # Insert scaffold_code just BEFORE the TRAINERS dict (with separating newlines)
    new_text = (text[:anchor_pos].rstrip() + '\n\n\n'
                + scaffold_code.strip() + '\n\n\n'
                + text[anchor_pos:])

    # Insert registry_line into TRAINERS dict. Re-anchor on the modified
    # text using the same line-anchored regex — scaffold code routinely
    # mentions "TRAINERS" in comments (e.g., "append before TRAINERS dict"),
    # so a plain substring search after anchor_pos would find the comment
    # first and corrupt an unrelated dict literal inside the class body.
    m2 = TRAINERS_ANCHOR.search(new_text)
    if not m2:
        print('FATAL: TRAINERS_ANCHOR not findable after scaffold insert',
              file=sys.stderr)
        return False
    dict_open = new_text.find('{', m2.start())
    if dict_open < 0:
        print('FATAL: could not find TRAINERS dict opening brace', file=sys.stderr)
        return False
    depth = 0
    dict_close = -1
    for i in range(dict_open, len(new_text)):
        ch = new_text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                dict_close = i
                break
    if dict_close < 0:
        print('FATAL: unbalanced braces in TRAINERS dict', file=sys.stderr)
        return False

    # Insert the registry line just before the closing brace, on its own line
    indent = '    '
    reg = registry_line.strip()
    # Strip trailing comma if present so we can normalize
    reg = reg.rstrip(',').strip()
    insertion = f'{indent}{reg},\n'
    new_text = new_text[:dict_close] + insertion + new_text[dict_close:]

    TRAINERS_PY.write_text(new_text)
    print(f'Wrote {TRAINERS_PY}: +{len(insertion)+len(scaffold_code)} chars')
    return True


def insert_search_space(search_space_block: str, registry_key: str) -> bool:
    """Append the HP search-space block to SEARCH_SPACES dict. The block
    from the brief is the dict entry (key + value). We find the closing
    `}` of SEARCH_SPACES and insert before it."""
    text = SEARCH_SPACES_PY.read_text()

    # Anchor: SEARCH_SPACES: dict[str, dict] = { or SEARCH_SPACES = {
    m = re.search(r'^SEARCH_SPACES\s*[:=]\s*.*?\{\s*$', text, re.MULTILINE)
    if not m:
        print('WARN: SEARCH_SPACES not found in search_spaces.py — skipping',
              file=sys.stderr)
        return False
    dict_open = m.end() - 1  # position of '{'

    # Find matching '}'
    depth = 0
    dict_close = -1
    for i in range(dict_open, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                dict_close = i
                break
    if dict_close < 0:
        print('WARN: unbalanced braces in SEARCH_SPACES — skipping', file=sys.stderr)
        return False

    # Idempotency: skip if 'registry_key' already a key in the dict
    if re.search(rf"['\"]\s*{re.escape(registry_key)}\s*['\"]\s*:",
                 text[dict_open:dict_close]):
        print(f'SEARCH_SPACES already has {registry_key!r} — skipping insert')
        return True

    indent = '    '
    block = search_space_block.strip().rstrip(',')
    # Make sure the trailing comma is present
    insertion = f'{indent}{block},\n'
    new_text = text[:dict_close] + insertion + text[dict_close:]
    SEARCH_SPACES_PY.write_text(new_text)
    print(f'Wrote {SEARCH_SPACES_PY}: +{len(insertion)} chars')
    return True


def validate_import(registry_key: str) -> bool:
    """Try to import models.trainers and confirm the new key is present."""
    cmd = [str(VENV_PY), '-c',
           f"import sys; sys.path.insert(0, '{BASE}'); "
           f"from models.trainers import TRAINERS; "
           f"assert '{registry_key}' in TRAINERS, list(TRAINERS); "
           f"print('OK: {registry_key} is registered')"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode == 0


def commit_scaffold(technique: str, registry_key: str) -> Optional[str]:
    paths = ['models/trainers.py', 'models/search_spaces.py']
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
        print('commit: nothing staged (maybe scaffold was already in place)')
        return None
    msg = f'scaffold: {registry_key} ({technique[:60]})'
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--brief', type=str, default=str(LATEST_BRIEF))
    parser.add_argument('--dry-run', action='store_true',
                        help='Parse only; do not install/write/commit')
    args = parser.parse_args()

    brief_path = Path(args.brief)
    if not brief_path.exists():
        print(f'No brief found at {brief_path}', file=sys.stderr)
        sys.exit(1)

    text = brief_path.read_text(errors='replace')
    parsed = parse_brief(text)
    print(f'Parsed brief: technique={parsed.get("technique")!r}, '
          f'registry_key={parsed.get("registry_key")!r}')

    if parsed.get('skip'):
        print('Brief recommends no candidate (technique=NONE). Nothing to do.')
        telegram(f'ℹ️ ml_scaffold: no-op (brief recommends no candidate)')
        return

    if not parsed.get('registry_key'):
        print('Could not extract registry key from brief. Aborting.',
              file=sys.stderr)
        telegram('⚠️ ml_scaffold: brief missing registry key, aborting')
        sys.exit(2)

    if already_registered(parsed['registry_key']):
        print(f'{parsed["registry_key"]!r} already in TRAINERS — scaffold idempotent no-op.')
        telegram(f'ℹ️ ml_scaffold: {parsed["registry_key"]} already registered')
        return

    if args.dry_run:
        print('--dry-run: stopping after parse')
        print(f'  pip:     {parsed.get("pip_command")!r}')
        print(f'  weights: {parsed.get("weights_command")!r}')
        print(f'  scaffold code chars: '
              f'{len(parsed.get("scaffold_code") or "")}')
        print(f'  registry line: {parsed.get("registry_line")!r}')
        return

    # 1. pip install
    if not run_pip(parsed.get('pip_command') or ''):
        telegram(f'⚠️ ml_scaffold: pip install failed for {parsed["registry_key"]}')
        sys.exit(3)

    # 2. weights (best-effort)
    run_weights(parsed.get('weights_command') or '')

    # 3. scaffold code + registry line
    if not parsed.get('scaffold_code') or not parsed.get('registry_line'):
        print('Missing scaffold_code or registry_line — aborting.', file=sys.stderr)
        telegram(f'⚠️ ml_scaffold: brief missing scaffold/registry code')
        sys.exit(4)

    # Snapshot the working-tree content of the files we're about to edit
    # so we can restore on validate failure WITHOUT touching concurrent
    # uncommitted claude_mode work elsewhere. `git checkout --` is too
    # blunt — it discards every uncommitted change in the file, including
    # in-flight trainer additions from a concurrent iteration.
    pre_trainers = TRAINERS_PY.read_text()
    pre_search = SEARCH_SPACES_PY.read_text() if SEARCH_SPACES_PY.exists() else None

    if not insert_scaffold(parsed['scaffold_code'], parsed['registry_line']):
        telegram(f'⚠️ ml_scaffold: trainers.py edit failed')
        sys.exit(5)

    # 4. search space (non-fatal on failure)
    if parsed.get('search_space_block'):
        insert_search_space(parsed['search_space_block'], parsed['registry_key'])

    # 5. validate import
    if not validate_import(parsed['registry_key']):
        # Surgical restore from in-memory snapshot — preserves whatever
        # uncommitted state was in the working tree before we touched it.
        TRAINERS_PY.write_text(pre_trainers)
        if pre_search is not None:
            SEARCH_SPACES_PY.write_text(pre_search)
        telegram(f'⚠️ ml_scaffold: {parsed["registry_key"]} failed import; restored snapshot')
        sys.exit(6)

    # 6. commit
    sha = commit_scaffold(parsed['technique'], parsed['registry_key'])
    telegram(
        f'🛠️ *Scaffold installed*\n'
        f'Trainer: `{parsed["registry_key"]}`\n'
        f'Technique: {parsed["technique"]}\n'
        f'git: {sha}\n'
        f'claude_mode will see it on its next 21:00 run.'
    )


if __name__ == '__main__':
    main()
