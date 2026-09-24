"""Evaluate SensorHealth on fixed, separate synthetic splits."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe.health_evaluation import write_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation_output" / "health")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--require-targets", action="store_true", help="Exit 2 if a holdout target fails")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
        report = write_comparison(args.output, configuration=config)
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(args.output / "comparison.md")
    return 2 if args.require_targets and not all(report["splits"]["holdout"]["acceptance"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
