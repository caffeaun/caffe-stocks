#!/usr/bin/env python3
"""Extract structured feedback from Claude ML improvement logs and gate output.

Reads the Claude log for a JSON report block, merges with gate decision,
and appends to ml-feedback.json.
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime


FEEDBACK_FILE = os.path.expanduser('~/projects/caffe-stocks/data/ml-feedback.json')


def extract_json_from_log(log_path):
    """Extract the structured JSON report block from Claude's log output."""
    if not os.path.isfile(log_path):
        return {}

    with open(log_path) as f:
        content = f.read()

    # Look for JSON block in ```json ... ``` fences
    json_blocks = re.findall(r'```json\s*\n(.*?)\n\s*```', content, re.DOTALL)

    for block in reversed(json_blocks):  # last one is most likely the report
        try:
            data = json.loads(block)
            # Validate it has expected fields
            if any(k in data for k in ['strategy_category', 'strategy_detail', 'what_changed']):
                return data
        except json.JSONDecodeError:
            continue

    # Fallback: try to find JSON object with strategy fields anywhere in text
    for match in re.finditer(r'\{[^{}]*"strategy_category"[^{}]*\}', content, re.DOTALL):
        try:
            data = json.loads(match.group())
            return data
        except json.JSONDecodeError:
            continue

    return {}


def extract_improvement_note(log_path):
    """Extract a human-readable improvement note from the log."""
    if not os.path.isfile(log_path):
        return 'ML improvement attempt'

    with open(log_path) as f:
        content = f.read()

    # Try structured JSON first
    json_data = extract_json_from_log(log_path)
    if json_data.get('strategy_detail'):
        return json_data['strategy_detail']

    # Fallback: scan for common patterns
    for marker in ['## What I Changed', '## Change', '### Change', 'I changed', 'I modified']:
        idx = content.find(marker)
        if idx >= 0:
            line = content[idx:idx + 300].split('\n')[1] if '\n' in content[idx:idx + 300] else ''
            line = line.strip().strip('-').strip()
            if line:
                return line[:150]

    return 'ML improvement attempt'


def parse_gate_output(gate_output_path):
    """Parse model_gate.py JSON output."""
    if not gate_output_path or not os.path.isfile(gate_output_path):
        return {}

    with open(gate_output_path) as f:
        content = f.read()

    # Gate outputs JSON, possibly with log lines before it
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    # Try full content
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


def build_feedback_entry(claude_log, gate_output_path=None, gate_exit=None):
    """Build a complete feedback entry from log and gate output."""
    json_report = extract_json_from_log(claude_log)
    gate_data = parse_gate_output(gate_output_path) if gate_output_path else {}

    # Determine decision
    if gate_exit is not None:
        decision = 'promoted' if gate_exit == 0 else 'rejected'
    elif gate_data.get('decision'):
        decision = gate_data['decision']
    else:
        decision = 'unknown'

    entry = {
        'id': None,  # filled by caller
        'timestamp': datetime.now().isoformat(),
        'strategy_category': json_report.get('strategy_category', 'other'),
        'strategy_detail': json_report.get('strategy_detail',
                                            extract_improvement_note(claude_log)),
        'hypothesis': json_report.get('hypothesis', ''),
        'what_changed': json_report.get('what_changed', ''),
        'lessons': json_report.get('lessons', ''),
        'candidate_backtest': gate_data.get('candidate', {}).get('backtest', {}),
        'production_backtest': gate_data.get('production', {}).get('backtest', {}),
        'candidate_metrics': {
            'precision': gate_data.get('candidate', {}).get('precision', 0),
            'accuracy': gate_data.get('candidate', {}).get('accuracy', 0),
            'recall': gate_data.get('candidate', {}).get('recall', 0),
        },
        'decision': decision,
        'reason': gate_data.get('reason', ''),
    }

    # Capture walk-forward aggregate if present
    wf = gate_data.get('walk_forward')
    if wf and wf.get('aggregate'):
        entry['walk_forward'] = {
            'passed': wf.get('passed', False),
            'avg_win_rate': wf['aggregate'].get('avg_win_rate', 0),
            'wr_std': wf['aggregate'].get('wr_std', 0),
            'total_trades': wf['aggregate'].get('total_trades', 0),
            'valid_splits': wf['aggregate'].get('valid_splits', 0),
        }

    return entry


def load_feedback():
    """Load existing feedback entries."""
    if not os.path.isfile(FEEDBACK_FILE):
        return []
    with open(FEEDBACK_FILE) as f:
        return json.load(f)


def save_feedback(entries):
    """Save feedback entries with sequential IDs."""
    for i, entry in enumerate(entries, 1):
        entry['id'] = i
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(entries, f, indent=2)


def append_feedback(entry):
    """Append a new feedback entry and save."""
    entries = load_feedback()
    entry['id'] = len(entries) + 1
    entries.append(entry)
    save_feedback(entries)
    return entry


def seed_from_improvements(improvements_path):
    """Seed feedback from existing ml-improvements.json (one-time migration)."""
    if not os.path.isfile(improvements_path):
        print(f"No improvements file at {improvements_path}")
        return

    with open(improvements_path) as f:
        improvements = json.load(f)

    entries = load_feedback()
    existing_timestamps = {e.get('timestamp') for e in entries}

    added = 0
    for imp in improvements:
        ts = imp.get('timestamp', '')
        if ts in existing_timestamps:
            continue

        entry = {
            'id': None,
            'timestamp': ts,
            'strategy_category': 'other',
            'strategy_detail': imp.get('improvement_note', 'ML improvement attempt'),
            'hypothesis': '',
            'what_changed': imp.get('improvement_note', ''),
            'lessons': '',
            'candidate_backtest': imp.get('candidate', {}).get('backtest', {}),
            'production_backtest': imp.get('production', {}).get('backtest', {}),
            'candidate_metrics': {
                'precision': imp.get('candidate', {}).get('precision', 0),
                'accuracy': imp.get('candidate', {}).get('accuracy', 0),
                'recall': imp.get('candidate', {}).get('recall', 0),
            },
            'decision': imp.get('decision', 'unknown'),
            'reason': imp.get('reason', ''),
        }
        entries.append(entry)
        added += 1

    if added:
        save_feedback(entries)
        print(f"Seeded {added} entries from {improvements_path}")
    else:
        print("No new entries to seed")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command')

    # Extract from a completed run
    ext = sub.add_parser('extract', help='Extract feedback from Claude log + gate output')
    ext.add_argument('--claude-log', required=True, help='Path to Claude improvement log')
    ext.add_argument('--gate-output', help='Path to gate JSON output file')
    ext.add_argument('--gate-exit', type=int, help='Gate exit code (0=promoted, 1=rejected)')

    # Seed from old improvements
    seed = sub.add_parser('seed', help='Seed feedback from ml-improvements.json')
    seed.add_argument('--improvements', required=True, help='Path to ml-improvements.json')

    # Extract just the improvement note (for Telegram)
    note = sub.add_parser('extract-note', help='Extract improvement note from Claude log')
    note.add_argument('--claude-log', required=True)

    # Show current feedback
    sub.add_parser('show', help='Show current feedback entries')

    args = parser.parse_args()

    if args.command == 'extract':
        entry = build_feedback_entry(args.claude_log, args.gate_output, args.gate_exit)
        result = append_feedback(entry)
        print(json.dumps(result, indent=2))

    elif args.command == 'seed':
        seed_from_improvements(args.improvements)

    elif args.command == 'extract-note':
        print(extract_improvement_note(args.claude_log))

    elif args.command == 'show':
        entries = load_feedback()
        print(json.dumps(entries, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
