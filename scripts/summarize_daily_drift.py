# ***************************************************************************
#  Replay every day-file in a 1-second data directory through ONE continuous
#  SensorHealth engine and tabulate the incidents opened per day, so detector
#  behavior can be eyeballed across many days instead of one CLI run at a time.
#
#    python scripts/summarize_daily_drift.py
#    python scripts/summarize_daily_drift.py --limit 10 -o /tmp/drift.csv
#    python scripts/summarize_daily_drift.py --config docs/health-config.json
# ***************************************************************************

import argparse
import collections
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

from safe.health import SensorHealth
from safe.loader import replay_csv

# The day-files are 1-second data; profiles must know the cadence.
DEFAULT_CONFIGURATION = {"profiles": {"expected_interval_seconds": 1}}


def _day_label(file_path):
    name = os.path.basename(file_path)
    # valo_node_01_20250614_20250615.csv.gz -> 20250614
    parts = name.replace(".csv.gz", "").replace(".csv", "").split("_")
    return parts[-2] if len(parts) >= 2 else name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="mintsXU4/data/valo_node_01_1s",
                        help="Directory of day-files (default: %(default)s)")
    parser.add_argument("--pattern", default="*.csv.gz",
                        help="Glob pattern for day-files (default: %(default)s)")
    parser.add_argument("--config", type=Path,
                        help="SensorHealth JSON configuration (default: 1-second cadence profiles)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N day-files (sorted by name)")
    parser.add_argument("-o", "--output", default="mintsXU4/output/drift_day_summary.csv",
                        help="Where to write the per-day CSV (default: %(default)s)")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    configuration = (json.loads(args.config.read_text(encoding="utf-8")) if args.config
                     else DEFAULT_CONFIGURATION)

    files = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"No files matched {args.data_dir}/{args.pattern}", file=sys.stderr)
        return 1

    counts = collections.Counter()

    def on_event(action, event):
        if action == "opened":
            counts[event["category"]] += 1

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    engine = SensorHealth(**configuration, on_event=on_event)
    rows = []
    failed = False
    for i, file_path in enumerate(files, 1):
        day = _day_label(file_path)
        counts.clear()
        t0 = time.time()
        if replay_csv(file_path, engine=engine) is None:
            # The engine may hold partial state; later days would not be comparable.
            print(f"[{i}/{len(files)}] {day}: FAILED; stopping")
            failed = True
            break
        total = sum(counts.values())
        print(f"[{i}/{len(files)}] {day}: {total} incident(s) opened in {time.time() - t0:.1f}s "
              f"({dict(counts)})")
        rows.append({"day": day, "total_incidents": total, **counts})

    if not rows:
        return 1
    df = pd.DataFrame(rows).fillna(0)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} row(s) to {args.output}")
    print(df["total_incidents"].agg(["mean", "min", "max", "count"]))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
