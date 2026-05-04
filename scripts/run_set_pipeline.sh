#!/bin/bash
SCRIPTS_DIR="/home/kanoonth-ai/projects/caffe-stocks/scripts"
/home/kanoonth-ai/projects/caffe-stocks/venv/bin/python "$SCRIPTS_DIR/fetch_ohlcv.py"
/home/kanoonth-ai/projects/caffe-stocks/venv/bin/python "$SCRIPTS_DIR/compute_indicators.py"
/home/kanoonth-ai/projects/caffe-stocks/venv/bin/python "$SCRIPTS_DIR/score_signals.py"
