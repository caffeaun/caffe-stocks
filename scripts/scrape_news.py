import requests
import json
import os
import sys
from datetime import datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

sys.path.insert(0, os.path.dirname(__file__))
from scrape_alert import alert_scrape_error

# Bangkok Post business RSS feed (more reliable than HTML scraping)
RSS_URL = 'https://www.bangkokpost.com/rss/data/business.xml'

# SET symbols tracked in candles DB (without .BK suffix for headline matching)
SET_SYMBOLS = [
    'ADVANC', 'AWC', 'BDMS', 'COM7', 'CPALL', 'CPN',
    'MINT', 'PTT', 'SCC', 'GULF', 'JMART', 'SAWAD',
    'MTC', 'JMT', 'TIDLOR',
]


def scrape_news():
    """Fetch Thai financial headlines from Bangkok Post RSS and extract symbols."""
    try:
        response = requests.get(RSS_URL, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        channel = root.find('channel')
        if channel is None:
            return []

        headlines = []
        for item in channel.findall('item'):
            title_el = item.find('title')
            pub_el = item.find('pubDate')
            desc_el = item.find('description')

            if title_el is None:
                continue

            title = title_el.text.strip() if title_el.text else ''
            description = desc_el.text.strip() if desc_el is not None and desc_el.text else ''
            combined_text = title + ' ' + description

            # Parse date
            date = datetime.now().strftime('%Y-%m-%d')
            if pub_el is not None and pub_el.text:
                try:
                    date = parsedate_to_datetime(pub_el.text).strftime('%Y-%m-%d')
                except Exception:
                    pass

            # Match symbol from title/description
            symbol = 'UNKNOWN'
            for s in SET_SYMBOLS:
                if s in combined_text.upper():
                    symbol = s
                    break

            headlines.append({
                'symbol': symbol,
                'headline': title,
                'date': date,
            })

        return headlines
    except Exception as e:
        print(f'Error fetching news: {e}')
        alert_scrape_error("scrape_news.py", f"Bangkok Post RSS failed: {e}")
        return []


def main():
    file_path = '/home/kanoonth-ai/projects/caffe-stocks/data/headlines.json'

    # Load existing data
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = []

    # Deduplicate by headline text (not symbol — multiple headlines per symbol are valid)
    existing_headlines = {h['headline'] for h in data}

    new_headlines = scrape_news()
    added = 0
    for headline in new_headlines:
        if headline['headline'] not in existing_headlines:
            data.append(headline)
            existing_headlines.add(headline['headline'])
            added += 1

    # Save back to file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f'Added {added} new headlines. Total: {len(data)}')


if __name__ == '__main__':
    main()
