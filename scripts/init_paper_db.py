import sqlite3
import os

def init_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute('''CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        entry_date TEXT NOT NULL,
        entry_price REAL NOT NULL,
        exit_date TEXT,
        exit_price REAL,
        exit_reason TEXT,
        quantity INTEGER NOT NULL,
        pnl_pct REAL,
        pnl_amount REAL,
        portfolio_after REAL,
        commission REAL DEFAULT 50
    )''')
    conn.commit()
    conn.close()
    print(f"Initialized: {db_path}")

def main():
    init_db('/home/kanoonth-ai/projects/caffe-stocks/data/paper-silent.db')
    init_db('/home/kanoonth-ai/projects/caffe-stocks/data/paper-live.db')

if __name__ == '__main__':
    main()