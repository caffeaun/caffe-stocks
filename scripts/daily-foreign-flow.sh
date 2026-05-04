#!/bin/bash
# Fetch daily foreign investor flow data from SET API
# Run Mon-Fri after market close (17:30 ICT)
cd /home/kanoonth-ai/projects/caffe-stocks
/home/kanoonth-ai/projects/caffe-stocks/venv/bin/python scripts/fetch_market_data.py --only foreign 2>&1 | tee -a /home/kanoonth-ai/projects/caffe-stocks/logs/foreign-flow.log
