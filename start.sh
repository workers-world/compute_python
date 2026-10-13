#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -d venv ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

echo "=== strategy-optimizer starting on :8002 ==="
exec uvicorn main:app --host 0.0.0.0 --port 8002 --log-level info
