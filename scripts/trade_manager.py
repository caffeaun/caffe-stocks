import json
import os
import sys
from datetime import datetime

STATE_FILE = '/home/kanoonth-ai/projects/caffe-stocks/memory/trade_state.json'

# Required fields per state (fields that must be non-null)
REQUIRED_FIELDS = {
    'IDLE': [],
    'SIGNAL_SENT': ['signal_id'],
    'ORDER_PENDING': ['signal_id', 'symbol'],
    'POSITION_OPEN': ['symbol', 'fill_price', 'stop_loss', 'target', 'entry_date', 'peak_price'],
    'SELL_PENDING': ['symbol', 'fill_price', 'stop_loss', 'target', 'entry_date'],
}


class TradeStateMachine:
    def __init__(self):
        self.state = self.load_state()

    def load_state(self):
        if not os.path.exists(STATE_FILE):
            return {
                'state': 'IDLE',
                'portfolio_value': 10000.0,
                'cash': 10000.0
            }
        with open(STATE_FILE, 'r') as f:
            return json.load(f)

    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def validate_state(self):
        """Check state consistency. Returns list of error strings (empty = valid)."""
        errors = []
        current = self.state.get('state', 'UNKNOWN')
        if current not in REQUIRED_FIELDS:
            errors.append(f"Unknown state: {current}")
            return errors
        for field in REQUIRED_FIELDS[current]:
            if self.state.get(field) is None:
                errors.append(f"State is {current} but '{field}' is null/missing")
        return errors

    def reset_state(self):
        """Reset state to IDLE, preserving portfolio_value and cash."""
        portfolio_value = self.state.get('portfolio_value', 10000.0)
        cash = self.state.get('cash', portfolio_value)
        self.state = {
            'state': 'IDLE',
            'portfolio_value': portfolio_value,
            'cash': cash,
        }
        self.save_state()
        print(f'State reset to IDLE (portfolio_value={portfolio_value}, cash={cash})')

    def repair_state(self):
        """Fix missing required fields with sensible defaults. Does not change state."""
        current = self.state.get('state', 'IDLE')
        if current not in REQUIRED_FIELDS:
            print(f'Unknown state "{current}" — use --reset instead')
            return
        fill_price = self.state.get('fill_price')
        defaults = {
            'symbol': 'UNKNOWN',
            'signal_id': 'REPAIRED',
            'fill_price': fill_price,
            'stop_loss': fill_price * 0.97 if fill_price else None,
            'target': fill_price * 1.15 if fill_price else None,
            'entry_date': datetime.now().strftime('%Y-%m-%d'),
            'peak_price': fill_price,
            'trailing_active': False,
        }
        repaired = []
        for field in REQUIRED_FIELDS[current]:
            if self.state.get(field) is None:
                if defaults.get(field) is not None:
                    self.state[field] = defaults[field]
                    repaired.append(f'{field}={defaults[field]}')
                else:
                    print(f'Cannot repair field "{field}" without fill_price — use --reset')
                    return
        if repaired:
            self.save_state()
            print(f'Repaired state ({current}): ' + ', '.join(repaired))
        else:
            print(f'State ({current}) is already consistent — nothing to repair')

    def transition_state(self, new_state, fill_price=None, symbol=None):
        # Valid transitions
        transitions = {
            'IDLE': ['SIGNAL_SENT'],
            'SIGNAL_SENT': ['ORDER_PENDING', 'IDLE'],
            'ORDER_PENDING': ['POSITION_OPEN', 'IDLE'],
            'POSITION_OPEN': ['SELL_PENDING'],
            'SELL_PENDING': ['IDLE']
        }

        current = self.state['state']
        if new_state not in transitions.get(current, []):
            raise ValueError(f'Invalid transition from {current} to {new_state}')

        # Update stop_loss/target on FILLED (ORDER_PENDING → POSITION_OPEN)
        if current == 'ORDER_PENDING' and new_state == 'POSITION_OPEN' and fill_price:
            self.state['stop_loss'] = fill_price * 0.97
            self.state['target'] = fill_price * 1.15
            self.state['fill_price'] = fill_price
            self.state['peak_price'] = fill_price
            self.state['entry_date'] = datetime.now().strftime('%Y-%m-%d')
            if symbol:
                self.state['symbol'] = symbol

        self.state['state'] = new_state
        self.save_state()
        print(f'✅ State transitioned: {current} → {new_state}')

    def check_trailing_stop(self, current_price):
        """Return True if trailing stop triggered. Updates peak_price if new high."""
        if self.state.get('state') != 'POSITION_OPEN':
            return False
        fill_price = self.state.get('fill_price')
        if fill_price is None:
            return False
        # Trailing stop activates after +7%
        entry_price = fill_price
        if current_price >= entry_price * 1.07:
            peak_price = self.state.get('peak_price', entry_price)
            if current_price > peak_price:
                self.state['peak_price'] = current_price
                peak_price = current_price
                self.save_state()
            trailing_stop = entry_price + (peak_price - entry_price) * 0.5
            return current_price <= trailing_stop
        return False

    def check_exit_conditions(self, current_price):
        """Check exit conditions in order. Returns exit reason string or None."""
        if self.state.get('state') != 'POSITION_OPEN':
            return None
        fill_price = self.state.get('fill_price')
        if fill_price is None:
            return None
        # 1. Stop-loss
        if current_price <= fill_price * 0.97:
            return 'stop_loss'
        # 2. Target (+15%)
        if current_price >= fill_price * 1.15:
            return 'target'
        # 3. Trailing stop
        if self.check_trailing_stop(current_price):
            return 'trailing_stop'
        # 4. 10-day max hold
        entry_date = self.state.get('entry_date')
        if entry_date:
            entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
            days_held = (datetime.now() - entry_dt).days
            if days_held >= 10:
                return 'max_hold'
        return None

    def is_signal_allowed(self):
        return self.state['state'] == 'IDLE'

def run_tests():
    # Reset state to IDLE before running tests to avoid stale state issues
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        state['state'] = 'IDLE'
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

    # Test 1: Basic transition
    tm = TradeStateMachine()
    tm.transition_state('SIGNAL_SENT')
    assert tm.state['state'] == 'SIGNAL_SENT', 'State not updated'
    
    # Test 2: FILLED transition with price
    tm.transition_state('ORDER_PENDING')
    tm.transition_state('POSITION_OPEN', fill_price=100.0)
    assert abs(tm.state['stop_loss'] - 97.0) < 1e-9, 'Stop loss calculation incorrect'
    assert abs(tm.state['target'] - 115.0) < 1e-9, 'Target calculation incorrect'
    
    # Test 3: Reject signal when not IDLE
    assert not tm.is_signal_allowed(), 'Signal allowed when not IDLE'
    
    print('✅ All tests passed!')


def main():
    if '--test' in sys.argv:
        run_tests()
    elif '--validate' in sys.argv:
        tm = TradeStateMachine()
        errors = tm.validate_state()
        if errors:
            print(f'State validation FAILED ({len(errors)} issue(s)):')
            for e in errors:
                print(f'  - {e}')
            sys.exit(1)
        else:
            print(f'State is consistent: {tm.state["state"]}')
    elif '--reset' in sys.argv:
        tm = TradeStateMachine()
        tm.reset_state()
    elif '--repair' in sys.argv:
        tm = TradeStateMachine()
        tm.repair_state()
        errors = tm.validate_state()
        if errors:
            print('Repair incomplete — remaining issues:')
            for e in errors:
                print(f'  - {e}')
            sys.exit(1)
    else:
        print('Usage: trade_manager.py [--test|--validate|--reset|--repair]')
        print('  --test     Run unit tests')
        print('  --validate Check state consistency')
        print('  --reset    Reset state to IDLE (preserves portfolio_value/cash)')
        print('  --repair   Fix missing required fields with defaults')

if __name__ == '__main__':
    main()