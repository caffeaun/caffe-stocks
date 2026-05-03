import logging
import os
from datetime import datetime, timedelta
import telegram

# Configure logging
logging.basicConfig(
    filename='/home/kanoonth-ai/projects/caffe-stocks/logs/workflow_test.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load Telegram config from kanoonth (single source of truth)
def _load_telegram_conf():
    conf_path = os.path.expanduser('~/kanoonth/scripts/telegram.conf')
    conf = {}
    if os.path.exists(conf_path):
        with open(conf_path) as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    conf[key] = value
    return conf

_tg = _load_telegram_conf()
BOT_TOKEN = _tg.get('TELEGRAM_BOT_TOKEN', '')
CHAT_ID = _tg.get('TELEGRAM_CHAT_ID', '')
bot = telegram.Bot(token=BOT_TOKEN)

# Mock portfolio state
portfolio = {
    'value': 10000.0,  # THB
    'positions': {}
}


def calculate_position_size(risk_percent, max_position_percent, portfolio_value):
    risk = portfolio_value * risk_percent / 100
    position_size = risk / 0.03  # 3% stop loss
    max_size = portfolio_value * max_position_percent / 100
    return min(position_size, max_size)


def send_signal(stock, price, limit_price, quantity):
    global state
    state = 'SIGNAL_SENT'
    message = f"Signal: {stock} at {limit_price} (limit), quantity {quantity} | Portfolio: {portfolio['value']:.2f} THB"
    bot.send_message(chat_id=CHAT_ID, text=message)
    logging.info(f"Signal sent: {stock} at {limit_price}, quantity {quantity}")


def handle_response(response, stock, fill_price):
    global state
    if response == 'ACCEPT':
        risk_percent = 1.0
        max_position_percent = 40.0
        position_size = calculate_position_size(risk_percent, max_position_percent, portfolio['value'])
        num_shares = position_size / fill_price
        stop_loss = fill_price * 0.97
        target = fill_price * 1.15
        portfolio['positions'][stock] = {
            'entry_price': fill_price,
            'stop_loss': stop_loss,
            'target': target,
            'entry_date': datetime.now(),
            'num_shares': num_shares
        }
        state = 'POSITION_OPEN'
        logging.info(f"Position opened: {stock} at {fill_price}, SL: {stop_loss}, Target: {target}")
    elif response == 'SKIP':
        state = 'IDLE'
        logging.info("Signal skipped by user")
    else:
        logging.warning(f"Unknown response: {response}")


def check_exit_conditions(stock, current_price):
    if stock not in portfolio['positions']:
        return None
    pos = portfolio['positions'][stock]
    # Check stop-loss
    if current_price <= pos['stop_loss']:
        return 'STOP_LOSS'
    # Check target
    if current_price >= pos['target']:
        return 'TARGET'
    # Check max hold (10 days)
    if datetime.now() - pos['entry_date'] >= timedelta(days=10):
        return 'MAX_HOLD'
    # Check trailing stop (after +7%)
    if current_price >= pos['entry_price'] * 1.07:
        peak = max(pos['entry_price'], current_price)
        trailing_stop = pos['entry_price'] + (peak - pos['entry_price']) * 0.5
        if current_price <= trailing_stop:
            return 'TRAILING_STOP'
    return None


# Test execution
if __name__ == '__main__':
    # Simulate full workflow
    stock = 'COM7'
    signal_price = 45.50
    limit_price = 45.50
    quantity = 100
    send_signal(stock, signal_price, limit_price, quantity)
    
    # Simulate user ACCEPT response
    fill_price = 45.50
    handle_response('ACCEPT', stock, fill_price)
    
    # Simulate market price (10% up)
    current_price = 50.05
    exit_reason = check_exit_conditions(stock, current_price)
    if exit_reason:
        logging.info(f"Exit reason: {exit_reason}")
    else:
        logging.info("No exit conditions met")

    # Verify position sizing
    pos = portfolio['positions'][stock]
    expected_size = 10000 * 0.01 / 0.03  # 3333.33 THB
    assert abs(pos['num_shares'] * fill_price - expected_size) < 1.0, \
        f"Position size mismatch: {pos['num_shares'] * fill_price:.2f} != {expected_size:.2f}"
    
    # Verify SL from fill
    assert abs(pos['stop_loss'] - fill_price * 0.97) < 0.01, \
        f"SL mismatch: {pos['stop_loss']} != {fill_price * 0.97}"
    
    logging.info("Full workflow test completed successfully")