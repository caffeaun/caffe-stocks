#!/usr/bin/env python3
"""Compare Port A (silent auto-accept) vs Port B (Telegram interactive) paper trading results."""

import sqlite3
import os
import pandas as pd

DB_A = os.path.expanduser("~/projects/caffe-stocks/data/paper-silent.db")
DB_B = os.path.expanduser("~/projects/caffe-stocks/data/paper-live.db")

COMMISSION_PCT = 0.001  # 0.1% per trade assumption


def load_trades(db_path):
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM trades", conn)
        conn.close()
        return df
    except Exception:
        return None


def calc_max_drawdown(portfolio_series):
    if portfolio_series.empty:
        return 0.0
    peak = portfolio_series.cummax()
    drawdown = (portfolio_series - peak) / peak * 100
    return abs(drawdown.min())


def compute_stats(df):
    if df is None or df.empty:
        return None

    total = len(df)
    wins = df[df["pnl_pct"] > 0]
    losses = df[df["pnl_pct"] <= 0]

    win_rate = len(wins) / total if total > 0 else 0.0
    avg_win = wins["pnl_pct"].mean() if not wins.empty else 0.0
    avg_loss = losses["pnl_pct"].mean() if not losses.empty else 0.0
    max_dd = calc_max_drawdown(df["portfolio_after"]) if "portfolio_after" in df.columns else 0.0
    ev = (win_rate * avg_win) + ((1 - win_rate) * avg_loss) - (COMMISSION_PCT * 100)

    return {
        "total_trades": total,
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "max_drawdown_pct": max_dd,
        "ev_per_trade": ev,
    }


def check_go_live(stats, port):
    if stats is None:
        return {}
    criteria = {}
    if port == "A":
        criteria["trades >= 50"] = stats["total_trades"] >= 50
        criteria["WR >= 50%"] = stats["win_rate"] >= 0.50
        criteria["EV > 0%"] = stats["ev_per_trade"] > 0
        criteria["Drawdown < 15%"] = stats["max_drawdown_pct"] < 15
    else:
        criteria["trades >= 20"] = stats["total_trades"] >= 20
        criteria["WR >= 50%"] = stats["win_rate"] >= 0.50
        criteria["EV > 0%"] = stats["ev_per_trade"] > 0
        criteria["Drawdown < 15%"] = stats["max_drawdown_pct"] < 15
    return criteria


def main():
    df_a = load_trades(DB_A)
    df_b = load_trades(DB_B)

    if df_a is None or df_a.empty:
        print("No trades to compare for Port A")
    if df_b is None or df_b.empty:
        print("No trades to compare for Port B")

    if (df_a is None or df_a.empty) and (df_b is None or df_b.empty):
        return

    stats_a = compute_stats(df_a)
    stats_b = compute_stats(df_b)

    print("=" * 65)
    print(f"{'Metric':<30} {'Port A (Silent)':>16} {'Port B (Telegram)':>16}")
    print("=" * 65)

    metrics = [
        ("Total Trades", "total_trades", "{:.0f}"),
        ("Win Rate", "win_rate", "{:.1%}"),
        ("Avg Win %", "avg_win_pct", "{:.2f}%"),
        ("Avg Loss %", "avg_loss_pct", "{:.2f}%"),
        ("Max Drawdown %", "max_drawdown_pct", "{:.2f}%"),
        ("EV per Trade", "ev_per_trade", "{:.2f}%"),
    ]

    for label, key, fmt in metrics:
        val_a = fmt.format(stats_a[key]) if stats_a else "N/A"
        val_b = fmt.format(stats_b[key]) if stats_b else "N/A"
        print(f"{label:<30} {val_a:>16} {val_b:>16}")

    print("=" * 65)

    print("\nGo-Live Criteria:")
    print("-" * 40)

    for port, stats in [("A", stats_a), ("B", stats_b)]:
        criteria = check_go_live(stats, port)
        if not criteria:
            print(f"Port {port}: No data")
            continue
        all_pass = all(criteria.values())
        print(f"Port {port} ({'READY' if all_pass else 'NOT READY'}):")
        for criterion, passed in criteria.items():
            print(f"  {'PASS' if passed else 'FAIL'} {criterion}")


if __name__ == "__main__":
    main()
