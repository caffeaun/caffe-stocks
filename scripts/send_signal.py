import json
import os
import requests
import datetime
import argparse

# Load environment variables
ENV_PATH = '/home/kanoonth-ai/projects/caffe-stocks/config/.env'
env = {}
with open(ENV_PATH) as f:
    for line in f:
        if '=' in line:
            key, value = line.strip().split('=', 1)
            env[key] = value

TELEGRAM_BOT_TOKEN = env.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = env.get('TELEGRAM_CHAT_ID')

# Load signal data
SIGNAL_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/latest_signal.json'
with open(SIGNAL_PATH) as f:
    signal = json.load(f)

# Load portfolio state
TRADE_STATE_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/trade_state.json'
with open(TRADE_STATE_PATH) as f:
    trade_state = json.load(f)
    portfolio_value = trade_state.get('portfolio_value', 0)
    cash = trade_state.get('cash', 0)
    positions = trade_state.get('positions', 0)
    distance = trade_state.get('distance', 0)

# Generate signal ID with required hyphens
SIGNAL_ID = f'SIG-{datetime.datetime.now().strftime("%Y%m%d")}-{signal["symbol"]}'

# Calculate derived values
stop_loss = signal['limit_price'] * 0.97
target = signal['limit_price'] * 1.15
risk = portfolio_value * 0.01

# Format message according to spec
MESSAGE = (
    f"📊 NEW SIGNAL — {SIGNAL_ID}\n"
    f"Stock: {signal['symbol']}\n"
    f"Action: BUY\n"
    f"Limit Price: ฿{signal['limit_price']:.2f}\n"
    f"Quantity: {signal['quantity']} shares\n"
    f"Stop-Loss: ฿{stop_loss:.2f} (−3.0%)\n"
    f"Target: ฿{target:.2f} (+15.0%)\n"
    f"Risk: ฿{risk:.2f} (1.0% of ฿{portfolio_value:.2f})\n\n"
    f"💰 PORTFOLIO: Cash ฿{cash:.2f} | Positions: {positions} | To ฿40K: ฿{distance:.2f}\n"
    f"Reply: T ACCEPT or T SKIP"
)


def send_to_telegram(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        print(f'Error sending message: {response.text}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Print message instead of sending')
    args = parser.parse_args()

    if args.dry_run:
        print(f'DRY RUN: Would send to chat {TELEGRAM_CHAT_ID}\n{MESSAGE}')
        return

    print(f'Sending signal to chat {TELEGRAM_CHAT_ID}...')
    send_to_telegram(MESSAGE)
    print('Signal sent successfully')

if __name__ == '__main__':
    main()