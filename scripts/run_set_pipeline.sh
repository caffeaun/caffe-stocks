#!/bin/bash
SCRIPTS_DIR="/home/kanoonth-ai/projects/caffe-stocks/scripts"
PYTHON="/home/kanoonth-ai/projects/caffe-stocks/venv/bin/python"
export PYTHONPATH="/home/kanoonth-ai/projects/caffe-stocks"

"$PYTHON" "$SCRIPTS_DIR/fetch_multi_source.py" --mode daily
"$PYTHON" "$SCRIPTS_DIR/compute_indicators.py"
"$PYTHON" "$SCRIPTS_DIR/score_signals.py"
