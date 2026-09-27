#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
python3 -m playwright install chromium
python3 -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
