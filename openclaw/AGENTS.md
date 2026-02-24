# OpenClaw Agent Profiles

## Agent 1: Data Scientist
**Role:** Model training, backtesting, and feature engineering
**Schedule:** Weekly retrain on Saturday, daily feature refresh at 06:00 ICT
**Responsibilities:**
- Run LSTM model training and backtesting pipelines
- Manage feature engineering (technical indicators, sentiment scores)
- Track model performance metrics and trigger retraining
- Evaluate and select utility LLM (Qwen3 8B or 14B)

## Agent 2: Signal Scanner
**Role:** Market scanning and signal generation
**Schedule:** Daily at 17:00 ICT (30 min after SET closes at 16:30)
**Responsibilities:**
- Scan SET mid-cap universe (COM7, JMART, SAWAD, MTC, JMT, TIDLOR, etc.) after market close
- Scrape closing data from Streaming app via browser automation (CDP relay)
- Apply LSTM predictions + technical filters (RSI, MACD, volume surge)
- Send buy/sell signals via Telegram with stock code, limit price, and quantity
- Include portfolio status in every signal (cash, positions, distance to ฿40K)
- Filter for stocks with ATR > 3.0% of price (Tier A only)

## Agent 3: Operations Reporter
**Role:** Trade lifecycle management and reporting
**Schedule:** Telegram poll every 5 min during market hours; daily close check at 17:15 ICT
**Responsibilities:**
- Manage 5-state machine: IDLE → SIGNAL_SENT → ORDER_PENDING → POSITION_OPEN → SELL_PENDING
- Process Telegram replies (ACCEPT, SKIP, FILLED, NOTFILLED, PARTIAL, SOLD)
- Recalculate stop-loss and target from ACTUAL fill price (not signal price)
- Track portfolio value and trigger ฿40,000 withdrawal alert
- Log every trade outcome to MEMORY.md for self-improvement loop
- Generate daily and weekly performance reports
