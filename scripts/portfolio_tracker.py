import json
import os
import argparse

import requests

def send_telegram(message):
    token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID', '-1003829269320')
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url)

STATE_FILE = '/home/kanoonth-ai/projects/caffe-stocks/memory/trade_state.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-withdrawal', action='store_true',
                        help='Test withdrawal trigger (prints alert without real check)')
    args = parser.parse_args()

    if args.test_withdrawal:
        print('WITHDRAWAL TRIGGER')
        return

    # Load trade state
    if not os.path.exists(STATE_FILE):
        print('No trade state found')
        return

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)

    # Calculate total portfolio value (cash + positions)
    total_value = state.get('portfolio_value', 10000)

    # Check withdrawal trigger
    if total_value >= 40000:
        send_telegram('Withdraw ฿30,000 → SCB shares. Keep ฿10,000 to restart.')

if __name__ == '__main__':
    main()