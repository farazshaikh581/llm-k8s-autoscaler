"""Simple web service for autoscaling experiments. Returns after a configurable compute delay."""

import hashlib
import os
import time
from flask import Flask, jsonify, request

app = Flask(__name__)

WORK_MS = int(os.environ.get("WORK_MS", "5"))

@app.route("/")
def handle():
    t0 = time.monotonic()
    # simulate CPU work
    data = b"x" * 1024
    for _ in range(WORK_MS * 50):
        data = hashlib.sha256(data).digest()
    elapsed_ms = (time.monotonic() - t0) * 1000
    return jsonify({"status": "ok", "work_ms": round(elapsed_ms, 1)})

@app.route("/health")
def health():
    return "ok"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
