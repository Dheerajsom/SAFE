"""Measure SensorHealth accuracy on the labeled synthetic PM and PC datasets."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe.synthetic_pm import DIRECTORIES, evaluate, loader_check, write_report

MODES = {"single-sensor": False, "with-references": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["pm", "pc", "both"], default="both",
                        help="Dataset to evaluate: PM mass bins, PC count bins, or both (default)")
    parser.add_argument("--data-dir", type=Path, help="Dataset directory (only with a single --family)")
    parser.add_argument("--output", type=Path,
                        help="Report directory (default: evaluation_output/synthetic_<family>)")
    parser.add_argument("--mode", choices=[*MODES, "both"], default="both",
                        help="single-sensor matches `safe health`; with-references feeds the other "
                             "co-located nodes as peers (default: both)")
    args = parser.parse_args(argv)
    families = ["pm", "pc"] if args.family == "both" else [args.family]
    if len(families) > 1 and (args.data_dir or args.output):
        parser.error("--data-dir and --output need a single --family")
    modes = list(MODES) if args.mode == "both" else [args.mode]
    for family in families:
        data_dir = args.data_dir or DIRECTORIES[family]
        output = args.output or ROOT / "evaluation_output" / f"synthetic_{family}"
        try:
            check = loader_check(data_dir, family)
            reports = {}
            for mode in modes:
                print(f"Evaluating {family} {mode}...", flush=True)
                reports[mode] = {**evaluate(data_dir, MODES[mode], family), "loader": check}
            write_report(reports, output, family)
        except (ValueError, OSError) as exc:
            parser.exit(1, f"Evaluation failed: {exc}\n")
        print(output / "accuracy.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
