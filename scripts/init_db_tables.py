import sqlite3
import os

DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/candles.db'


def create_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create signals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            lstm_score REAL,
            rsi REAL,
            atr REAL,
            volume_ratio REAL,
            action TEXT
        )
    ''')

    # Create trades table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            entry_date TEXT,
            exit_date TEXT,
            entry_price REAL,
            exit_price REAL,
            quantity INTEGER,
            pnl_pct REAL,
            outcome TEXT
        )
    ''')

    # Create performance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            portfolio_value REAL,
            win_rate REAL,
            total_trades INTEGER
        )
    ''')

    conn.commit()
    conn.close()
    print('Tables created successfully')

if __name__ == '__main__':
    create_tables()