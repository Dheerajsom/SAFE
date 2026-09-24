# ***************************************************************************
#  SAFE — command-line interface
#
#    safe health  <csv>            replay CSVs into correlated health incidents
#                                  (`safe stream` is an alias)
#    safe periods <csv> -o <dir>   period-over-period analysis + plots
# ***************************************************************************

import argparse
import glob
import logging
import os
import sys
import json
from pathlib import Path


def _expand_csv_args(paths):
    """Expand directories into their sorted *.csv / *.csv.gz files."""
    files = []
    for path in paths:
        if os.path.isdir(path):
            found = sorted(glob.glob(os.path.join(path, "*.csv.gz")) +
                           glob.glob(os.path.join(path, "*.csv")))
            if not found:
                print(f"No .csv/.csv.gz files found in directory: {path}", file=sys.stderr)
                return None
            files.extend(found)
        else:
            files.append(path)
    return files


def _add_periods_parser(subparsers):
    p = subparsers.add_parser(
        "periods", help="Run period-over-period drift analysis and write CSVs + plots")
    p.add_argument("csv", help="Path to the long-format CSV export")
    p.add_argument("-o", "--output", required=True, help="Output directory for period_*.csv")
    p.add_argument("--alpha", type=float, default=0.01,
                   help="Significance level for the drift tests (default: 0.01)")
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="safe",
        description="SAFE - Sensor Analysis and Failure Evaluation for MINTS air-quality nodes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_periods_parser(subparsers)
    health = subparsers.add_parser("health", aliases=["stream"],
                                   help="Replay CSVs into correlated health incidents")
    health.add_argument("csv", nargs="+",
                        help="One or more long-format CSV/CSV.GZ paths, or a directory of them "
                             "(replayed in sorted order through a single continuous engine)")
    health.add_argument("--metric", action="append", dest="metrics",
                        help="Restrict processing to this metric (repeatable)")
    health.add_argument("--merge", action="store_true",
                        help="Inputs cover the same period (e.g. one export per PM bin); merge them "
                             "into one stream so cross-metric rules see every field")
    health.add_argument("--config", type=Path, help="JSON profiles, model rules, and sensor metadata")
    health.add_argument("--state-in", type=Path, help="Resume a compatible saved health state")
    health.add_argument("--state-out", type=Path, help="Atomically save state after successful replay")
    health.add_argument("--tick-until", help="Explicit observation end (ISO UTC) for silence detection")
    health.add_argument("--event-update-interval", type=float, default=1800,
                        help="Minimum seconds between repeated JSONL incident updates; 0 exports every observation")
    health.add_argument("-o", "--output", type=Path, default=Path("mintsXU4/output/health"))

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        return _run(args)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        logger = logging.getLogger(__name__)
        logger.error("%s", exc)
        return 1


def _run(args):
    if args.command in ("health", "stream"):
        from safe.health import ENGINE_VERSION, SensorHealth
        from safe.loader import replay_csv, replay_csvs

        if args.config and args.state_in:
            raise ValueError("saved state already contains configuration; omit --config when resuming")
        if not 0 <= args.event_update_interval < float("inf"):
            raise ValueError("event-update-interval must be finite and nonnegative")
        configuration = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        files = _expand_csv_args(args.csv)
        if files is None:
            return 1
        args.output.mkdir(parents=True, exist_ok=True)
        event_path = args.output / "events.jsonl"
        notification_path = args.output / "notifications.jsonl"
        # Streams retain lifecycle evidence beyond bounded in-memory history.
        with event_path.open("a", encoding="utf-8") as events, notification_path.open("a", encoding="utf-8") as notices:
            exported = {}

            def on_event(action, event):
                signature = (event["category"], event["severity"], tuple(sorted(event["evidence"])))
                previous = exported.get(event["id"])
                if (action == "updated" and previous is not None and previous[1] == signature
                        and event["last_seen_at"] - previous[0] < args.event_update_interval):
                    return
                events.write(json.dumps({"action": action, "event": event}, allow_nan=False) + "\n")
                events.flush()
                if action == "closed":
                    exported.pop(event["id"], None)
                else:
                    exported[event["id"]] = (event["last_seen_at"], signature)

            def on_notification(event):
                notices.write(json.dumps(event, allow_nan=False) + "\n")
                notices.flush()

            callbacks = dict(on_event=on_event, on_notification=on_notification)
            engine = (SensorHealth.load_state(args.state_in, **callbacks) if args.state_in else
                      SensorHealth(**configuration, **callbacks))
            replayed = (replay_csv(files, engine=engine, metrics=args.metrics) if args.merge else
                        replay_csvs(files, engine=engine, metrics=args.metrics))
            if replayed is None:
                return 1
            if args.tick_until:
                engine.tick(args.tick_until)
            if args.state_out:
                engine.save_state(args.state_out)
            summary = {"engine_version": ENGINE_VERSION, "configuration": engine.configuration(),
                       "total_incidents": engine.incidents.total_opened,
                       "total_notifications": engine.incidents.total_notifications,
                       "active_incidents": len(engine.incidents.active),
                       "retained_incidents": engine.events}
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{engine.incidents.total_opened} incident(s); {engine.incidents.total_notifications} notification(s). {args.output}")
        return 0

    if args.command == "periods":
        from safe.periods import run_period_analysis

        ok = run_period_analysis(args.csv, args.output, p_alpha=args.alpha,
                                 make_plots=not args.no_plots)
        return 0 if ok else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
