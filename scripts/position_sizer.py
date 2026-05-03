import argparse

def calculate_position_size(portfolio_value, price, first_ten=False):
    risk_amount = portfolio_value * 0.01  # 1% risk
    qty_by_risk = int(risk_amount / (price * 0.03))  # 3% stop-loss
    qty_by_cap = int((portfolio_value * 0.40) / price)  # 40% max position
    qty = min(qty_by_risk, qty_by_cap)
    if first_ten:
        qty = int(qty / 2)
    return qty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--portfolio', type=float, required=True, help='Portfolio value')
    parser.add_argument('--price', type=float, required=True, help='Stock price')
    parser.add_argument('--first-ten', action='store_true', help='Halve position size for first 10 trades')
    args = parser.parse_args()
    
    quantity = calculate_position_size(args.portfolio, args.price, args.first_ten)
    print(quantity)

if __name__ == '__main__':
    main()