"""Write (or verify) the seeded synthetic PM and PC datasets: one CSV per bin plus labels."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe.synthetic_pm import DEFAULT_SEED, DIRECTORIES, dataset_files, file_content, generate, write_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pm-output", type=Path, default=DIRECTORIES["pm"])
    parser.add_argument("--pc-output", type=Path, default=DIRECTORIES["pc"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 unless the files in both outputs match the generator "
                             "(gzip files are compared decompressed)")
    args = parser.parse_args(argv)
    directories = {"pm": args.pm_output, "pc": args.pc_output}
    if args.check:
        dataset = generate(args.seed)
        stale = [str(directory / name) for family, directory in directories.items()
                 for name, data in dataset_files(dataset, family).items()
                 if not (directory / name).is_file()
                 or file_content(name, (directory / name).read_bytes()) != file_content(name, data)]
        if stale:
            print("Out of date: " + ", ".join(stale), file=sys.stderr)
            return 1
        print(" and ".join(str(d) for d in directories.values()) + " match the generator")
        return 0
    for path in write_dataset(directories, args.seed):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
