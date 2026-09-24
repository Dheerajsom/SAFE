"""Write (or verify) the seeded synthetic PM dataset: one CSV per PM bin plus labels."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe.synthetic_pm import DEFAULT_DIRECTORY, DEFAULT_SEED, dataset_files, generate, write_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 unless the files in --output match the generator byte for byte")
    args = parser.parse_args(argv)
    if args.check:
        stale = [name for name, data in dataset_files(generate(args.seed)).items()
                 if not (args.output / name).is_file() or (args.output / name).read_bytes() != data]
        if stale:
            print("Out of date: " + ", ".join(stale), file=sys.stderr)
            return 1
        print(f"{args.output} matches the generator")
        return 0
    for name in write_dataset(args.output, args.seed):
        print(args.output / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
