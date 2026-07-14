#!/usr/bin/env python3
"""Build a real-world workload trace from Wikipedia request rates.

Fetches hourly pageview counts for a Wikimedia project from the public
Pageviews REST API and maps them into the RPS trace format the autoscaler
harness consumes (a per-step integer numpy array, same layout as
extract_workload_traces.py). One control step replays one hour of real
traffic, so the trace carries genuine diurnal + weekly structure and real
event spikes -- no synthetic spike injection.

Scaling matches trace_cpu.npy (norm * 2000 + 200, clipped [50, 4000]) so the
Wikipedia and Alibaba-CPU workloads are directly load-comparable on the
testbed.

Usage:
  python extract_wikipedia_trace.py                       # defaults below
  python extract_wikipedia_trace.py --start 2024030100 --end 2024050100
"""

import argparse
import json
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent / "traces"
API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/"
       "{project}/all-access/user/hourly/{start}/{end}")
# a contactable UA is required by the Wikimedia API policy
UA = "gym-sfu-autoscaler-research/1.0 (farazshaikh581@gmail.com)"
RPS_AMPLITUDE = 2000   # matches trace_cpu scaling
RPS_BASELINE = 200
RPS_CLIP = (50, 4000)


def _fetch_window(project: str, start: str, end: str) -> list:
    url = API.format(project=project, start=start, end=end)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)["items"]
        except Exception as e:
            if attempt == 3:
                raise
            print(f"  retry {attempt+1} ({e})")
            time.sleep(3 * (attempt + 1))
    return []


def fetch_hourly(project: str, start: str, end: str) -> np.ndarray:
    """Fetch hourly views over [start, end), chunked by month for reliability."""
    fmt = "%Y%m%d%H"
    t0 = datetime.strptime(start, fmt)
    t1 = datetime.strptime(end, fmt)
    items = []
    cur = t0
    while cur < t1:
        nxt = min(cur + timedelta(days=30), t1)
        print(f"  fetching {cur:%Y-%m-%d} -> {nxt:%Y-%m-%d}")
        items += _fetch_window(project, cur.strftime(fmt), nxt.strftime(fmt))
        cur = nxt
        time.sleep(0.5)

    # de-duplicate on timestamp (chunk boundaries overlap by one hour)
    by_ts = {it["timestamp"]: it["views"] for it in items}
    ts_sorted = sorted(by_ts)
    views = np.array([by_ts[t] for t in ts_sorted], dtype=float)
    # guard against occasional zero/missing hours: replace with local median
    if (views <= 0).any():
        med = np.median(views[views > 0])
        views[views <= 0] = med
    print(f"  got {len(views)} hourly points, "
          f"views/hr [{views.min():.0f}, {views.max():.0f}]")
    return views


def to_rps(views: np.ndarray) -> np.ndarray:
    norm = (views - views.min()) / (views.max() - views.min() + 1e-9)
    rps = norm * RPS_AMPLITUDE + RPS_BASELINE
    return np.clip(rps, *RPS_CLIP).astype(int)


def main():
    ap = argparse.ArgumentParser(description="Wikipedia -> RPS workload trace")
    ap.add_argument("--project", default="en.wikipedia.org")
    ap.add_argument("--start", default="2024030100", help="YYYYMMDDHH (inclusive)")
    ap.add_argument("--end", default="2024050100", help="YYYYMMDDHH (exclusive)")
    ap.add_argument("--out", default=str(OUT_DIR / "trace_wiki.npy"))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Wikipedia trace: {args.project} {args.start} -> {args.end}")
    views = fetch_hourly(args.project, args.start, args.end)
    rps = to_rps(views)
    np.save(args.out, rps)
    print(f"\nSaved {args.out}: {len(rps)} steps, "
          f"RPS [{rps.min()}, {rps.max()}], mean={rps.mean():.0f}, std={rps.std():.0f}")
    print(f"First 240 steps (a 10-day slice): RPS [{rps[:240].min()}, {rps[:240].max()}], "
          f"mean={rps[:240].mean():.0f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 3.2))
        ax.fill_between(range(len(rps)), rps, alpha=0.3)
        ax.plot(rps, linewidth=0.7)
        ax.axvspan(0, 240, color="tab:orange", alpha=0.12, label="first 240-step slice")
        ax.set_xlabel("Step (1 step = 1 hour of real traffic)")
        ax.set_ylabel("RPS")
        ax.set_title(f"Wikipedia Workload Trace ({args.project}, hourly, "
                     f"{args.start}-{args.end})")
        ax.legend(loc="upper right"); ax.grid(True, alpha=0.2)
        plt.tight_layout()
        p = OUT_DIR / "trace_wiki.png"
        plt.savefig(p, dpi=150); plt.close()
        print(f"Plot: {p}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
