"""Measure SensorHealth accuracy on the labeled synthetic PM dataset."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe.synthetic_pm import DEFAULT_DIRECTORY, evaluate, loader_check, write_report

MODES = {"single-sensor": False, "with-references": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation_output" / "synthetic_pm")
    parser.add_argument("--mode", choices=[*MODES, "both"], default="both",
                        help="single-sensor matches `safe health`; with-references feeds the other "
                             "co-located nodes as peers (default: both)")
    args = parser.parse_args(argv)
    modes = list(MODES) if args.mode == "both" else [args.mode]
    try:
        check = loader_check(args.data_dir)
        reports = {}
        for mode in modes:
            print(f"Evaluating {mode}...", flush=True)
            reports[mode] = {**evaluate(args.data_dir, MODES[mode]), "loader": check}
        write_report(reports, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(args.output / "accuracy.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
