"""Create deterministic, task-filtered CSV lists from a structure database."""

import argparse
import csv
import os
from pathlib import Path
import random


def discover_samples(database):
    samples = set()
    for poscar in Path(database).rglob("POSCAR"):
        if poscar.is_file():
            root = poscar.parent.parent if poscar.parent.name.upper() == "POSCAR" else poscar.parent
            samples.add(root.resolve())
    return sorted(samples)


def split_samples(samples, seed):
    samples = sorted(set(samples))
    if len(samples) < 3:
        raise ValueError("At least three structures are needed for train/validation/test lists.")
    random.Random(seed).shuffle(samples)
    n_train = min(max(1, int(0.70 * len(samples))), len(samples) - 2)
    n_val = max(1, int(0.15 * len(samples)))
    return samples[:n_train], samples[n_train:n_train+n_val], samples[n_train+n_val:]


def write_list(path, samples, column="files"):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([column])
        for sample in samples:
            writer.writerow([Path(os.path.relpath(sample, path.parent)).as_posix()])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("database"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    samples = discover_samples(args.database)
    try:
        train, validation, test = split_samples(samples, args.seed)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    requirements = {"all": (), "energy": ("energy", "forces", "stress"), "dos": ("dos", "VB_CB")}
    for task, files in requirements.items():
        for name, rows in (("Train", train), ("Val", validation)):
            selected = [row for row in rows if all((row / item).is_file() for item in files)]
            write_list(args.output / f"{name}_{task}.csv", selected)
            print(f"{name}_{task}: {len(selected)}")
    write_list(args.output / "predict.csv", test, "file_loc_test")
    print(f"Held-out structures: {len(test)}; lists: {args.output.resolve()}")


if __name__ == "__main__":
    main()
