import json
import sqlite3
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define keyword lists (case-insensitive)
positive_keywords = ['profit', 'growth', 'dividend', 'record', 'upgrade']
negative_keywords = ['loss', 'downgrade', 'lawsuit', 'debt', 'default']

db_path = '/home/kanoonth-ai/projects/caffe-stocks/data/candles.db'

# Connect to SQLite
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create sentiment table if not exists
cursor.execute('''
CREATE TABLE IF NOT EXISTS sentiment (
    timestamp TEXT,
    symbol TEXT,
    score REAL,
    headline TEXT,
    UNIQUE(headline, timestamp)
)
''')
conn.commit()

# Read headlines from JSON file
headlines_file = '/home/kanoonth-ai/projects/caffe-stocks/data/headlines.json'
try:
    with open(headlines_file, 'r') as f:
        headlines = json.load(f)
except FileNotFoundError:
    logging.error(f'Headlines file not found: {headlines_file}')
    conn.close()
    exit()

# Get already-stored headlines to avoid duplicates
cursor.execute('SELECT headline, timestamp FROM sentiment')
existing = {(row[0], row[1]) for row in cursor.fetchall()}

# Process each headline
inserted = 0
skipped = 0
for entry in headlines:
    headline = entry['headline']
    symbol = entry['symbol']
    date = entry['date']

    if (headline, date) in existing:
        skipped += 1
        continue

    # Clean and split headline into words
    words = [re.sub(r'[^\w]', '', word.lower()) for word in headline.split()]

    # Count positive/negative words
    pos_count = sum(1 for word in words if word in positive_keywords)
    neg_count = sum(1 for word in words if word in negative_keywords)
    total_words = len(words)

    # Calculate score (clamped to [-1, 1])
    score = 0
    if total_words > 0:
        score = (pos_count - neg_count) / total_words
        score = max(-1, min(1, score))

    # Insert into database
    cursor.execute('''
    INSERT OR IGNORE INTO sentiment (timestamp, symbol, score, headline)
    VALUES (?, ?, ?, ?)
    ''', (date, symbol, score, headline))
    inserted += 1

conn.commit()
logging.info(f'Processed {len(headlines)} headlines: {inserted} inserted, {skipped} skipped (already stored)')
conn.close()
logging.info('Sentiment scoring completed.')
