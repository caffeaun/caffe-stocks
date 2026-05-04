#!/home/kanoonth-ai/projects/caffe-stocks/venv/bin/python
import sqlite3
import random
from datetime import datetime, timedelta

db_path = '/home/kanoonth-ai/projects/caffe-stocks/data/paper-live.db'

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Insert 20 fake trades for Port B
for i in range(20):
    trade_id = f'port_b_{i+1}'
    symbol = random.choice(['COM7', 'JMART', 'SAWAD', 'MTC'])
    price = round(random.uniform(40, 55), 2)
    quantity = random.randint(5, 25)
    timestamp = (datetime.now() - timedelta(days=random.randint(1, 14))).strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute('''
        INSERT INTO trades (signal_id, symbol, entry_price, quantity, entry_date)
        VALUES (?, ?, ?, ?, ?)
    ''', (trade_id, symbol, price, quantity, timestamp))

conn.commit()
conn.close()

print('✅ 20 fake trades inserted for Port B (paper-live.db)')