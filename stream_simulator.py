"""
Simulate a live feed: field officers' mobile app submits 2026 kharif farm records as they are collected.

Writes small JSON-lines files into data/stream/incoming/ every few seconds. The Spark Structured
Streaming job (python -m bda.streaming) picks them up as they arrive.

Usage:  python stream_simulator.py                         (30 batches of 1,000 records, every 3 s)
        python stream_simulator.py --batches 10 --interval 0  (dump quickly, for testing)
"""
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "raw" / "stream_source_2026.csv"
INCOMING = ROOT / "data" / "stream" / "incoming"


def main(batches: int, size: int, interval: float, reset: bool = False) -> None:
    src = pd.read_csv(SOURCE)
    INCOMING.mkdir(parents=True, exist_ok=True)
    if reset:                                                    # start the feed again from batch 0
        for f in INCOMING.glob("*.json"):
            f.unlink()
        print(f"cleared {INCOMING}")
    done = len(list(INCOMING.glob("batch_*.json")))
    if done >= len(src) // size:
        print("All batches have already been sent. Use --reset to replay the feed from the start.")
        return
    for b in range(done, min(done + batches, len(src) // size)):
        chunk = src.iloc[b * size:(b + 1) * size].copy()
        chunk["event_time"] = datetime.now().isoformat(timespec="seconds")
        tmp = INCOMING / f".tmp_batch_{b:04d}.json"
        chunk.to_json(tmp, orient="records", lines=True)
        os.replace(tmp, INCOMING / f"batch_{b:04d}.json")       # atomic: Spark never sees half a file
        print(f"{datetime.now():%H:%M:%S}  wrote batch_{b:04d}.json ({len(chunk)} records)")
        if interval:
            time.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--reset", action="store_true", help="delete previous batches and replay from the start")
    a = ap.parse_args()
    main(a.batches, a.size, a.interval, a.reset)
