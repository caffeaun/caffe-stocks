"""Shared Telegram alert for web scraping scripts.

Usage:
    from scrape_alert import alert_scrape_error
    alert_scrape_error("scrape_settrade.py", "COM7: API returned 403")
"""
import os
import requests

CONF_PATH = os.path.expanduser("~/kanoonth/scripts/telegram.conf")

_BOT_TOKEN = ""
_CHAT_ID = ""

if os.path.isfile(CONF_PATH):
    with open(CONF_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith("TELEGRAM_BOT_TOKEN="):
                _BOT_TOKEN = line.split("=", 1)[1].strip('"')
            elif line.startswith("TELEGRAM_CHAT_ID="):
                _CHAT_ID = line.split("=", 1)[1].strip('"')


def alert_scrape_error(script_name, message):
    """Send a scrape failure alert to Telegram."""
    if not _BOT_TOKEN or not _CHAT_ID:
        print(f"[alert] No Telegram config — {script_name}: {message}")
        return
    text = f"[Scrape Alert] {script_name}\n{message}"
    try:
        requests.post(
            f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage",
            data={"chat_id": _CHAT_ID, "text": text, "disable_web_page_preview": "true"},
            timeout=10,
        )
    except Exception as e:
        print(f"[alert] Send failed: {e}")
