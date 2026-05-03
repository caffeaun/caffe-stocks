import sqlite3
import csv
import os
from datetime import datetime, timedelta

# Database path
DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/candles.db'

# Export path
EXPORT_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/training_data.csv'

def compute_label(symbol, close_price, date):
    """Compute binary label: 1 if price rises 15% then drops 3% within 10 days"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Query next 10 days of data
    cursor.execute('''
        SELECT close FROM candles 
        WHERE symbol = ? AND date > ? 
        ORDER BY date
        LIMIT 10
    ''', (symbol, date))
    
    prices = [row[0] for row in cursor.fetchall()]
    if not prices:
        return 0
    
    # Check for 15% rise followed by 3% drop
    peak = close_price * 1.15
    for p in prices:
        if p >= peak:
            # Check if drops 3% from peak within window
            if p * 0.97 <= close_price:
                return 1
    return 0


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Fetch all candles data
    cursor.execute('SELECT * FROM candles')
    rows = cursor.fetchall()
    
    # Column names (from table structure)
    columns = [desc[0] for desc in cursor.description]
    
    # Add label column
    columns.append('label')
    
    # Compute labels and prepare data
    data = []
    positive_count = 0
    negative_count = 0
    
    for row in rows:
        row_dict = dict(zip(columns, row))
        
        # Compute label (simplified for verification)
        label = compute_label(row_dict['symbol'], row_dict['close'], row_dict['date'])
        
        # Update counts
        if label == 1:
            positive_count += 1
        else:
            negative_count += 1
        
        # Add label to row
        row_dict['label'] = label
        data.append(row_dict)

    # Write to CSV
    with open(EXPORT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)

    # Log summary
    total = len(data)
    print(f'Exported {total} rows to {EXPORT_PATH}')
    print(f'Positive labels (1): {positive_count} | Negative labels (0): {negative_count}')

if __name__ == '__main__':
    main()