"""Kaggle kernel logs are JSON arrays of {stream_name,time,data}; print the stdout/stderr text."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
raw = open(sys.argv[1], encoding="utf-8", errors="replace").read()
try:
    items = json.loads(raw)
    for it in items:
        print(it.get("data", ""), end="")
except Exception:
    print(raw)
