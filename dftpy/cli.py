"""Command-line entry points for the original ML-DFT models."""

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path

from .paths import read_file_list, require_files, structure_folder
from .config import FINGERPRINT_DEFAULTS


def _positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return result


def _positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Must be positive and finite.")
    return result


def _fingerprint_arguments(parser):
    for name, value in FINGERPRINT_DEFAULTS.items():
        value_type = _positive_int if name == "num_gamma" else _positive_float
        parser.add_argument("--" + name.replace("_", "-"), type=value_type, default=value)
    parser.add_argument("--padding-multiplier", type=_positive_float, default=1.0)
    parser.add_argument("--scaler-dir", type=Path,
                        help="Directory containing Scale_model_C/H/N/O.joblib.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="dftpy", description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    check = commands.add_parser("check-data", help="Check input file paths without model dependencies.")
    check.add_argument("--list", dest="file_list", required=True, type=Path)
    check.add_argument("--column", default="files")
    check.add_argument("--task", choices=("structure", "chg", "energy", "dos", "all"), default="structure")

    train = commands.add_parser("train", help="Train charge, energy, or DOS models.")
    train.add_argument("--task", required=True, choices=("chg", "energy", "dos", "all"))
    train.add_argument("--train-list", required=True, type=Path)
    train.add_argument("--val-list", required=True, type=Path)
    train.add_argument("--output-dir", type=Path, default=Path("runs/train"))
    train.add_argument("--epochs", type=_positive_int, default=100)
    train.add_argument("--batch-size", type=_positive_int, default=64)
    train.add_argument("--learning-rate", type=_positive_float, default=1e-4)
    train.add_argument("--patience", type=_positive_int, default=20)
    _fingerprint_arguments(train)

    infer = commands.add_parser("infer", help="Predict using existing trained checkpoints.")
    infer.add_argument("--infer-list", required=True, type=Path)
    infer.add_argument("--output-dir", type=Path, default=Path("results"))
    infer.add_argument("--predict-chg", action="store_true")
    infer.add_argument("--predict-energy", action="store_true")
    infer.add_argument("--predict-dos", action="store_true")
    infer.add_argument("--chg-checkpoint", type=Path, default=Path("best_chg.pth"))
    infer.add_argument("--energy-checkpoint", type=Path, default=Path("newEmodel.pth"))
    infer.add_argument("--dos-checkpoint", type=Path, default=Path("best_dos.pth"))
    infer.add_argument("--plot-dos", action="store_true")
    _fingerprint_arguments(infer)

    args = parser.parse_args(argv)
    if args.mode != "check-data":
        if args.padding_multiplier < 1:
            parser.error("padding-multiplier must be at least 1.")
        if args.narrowest_gaussian > args.widest_gaussian:
            parser.error("narrowest-gaussian must not exceed widest-gaussian.")
    if args.mode == "infer" and not any((args.predict_chg, args.predict_energy, args.predict_dos)):
        parser.error("Select at least one --predict-* option.")
    return args


def _folders(csv_path, column):
    return [str(structure_folder(path)) for path in read_file_list(csv_path, column)]


def check_inputs(folders, task):
    required = []
    if task in ("chg", "all"):
        required.extend(f"Coef_{element}.npy" for element in "CHNO")
    if task in ("energy", "all"):
        required.extend(("energy", "forces", "stress"))
    if task in ("dos", "all"):
        required.extend(("dos", "VB_CB"))
    require_files(folders, required)


@contextmanager
def _output_directory(path, args):
    previous = Path.cwd()
    target = path.resolve()
    target.mkdir(parents=True, exist_ok=False)
    with (target / "run.json").open("w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, indent=2, default=str)
        stream.write("\n")
    try:
        os.chdir(target)
        yield target
    finally:
        os.chdir(previous)


def _train(args):
    train = _folders(args.train_list, "files")
    validation = _folders(args.val_list, "files")
    if set(train) & set(validation):
        raise ValueError("Training and validation lists contain overlapping structures.")
    check_inputs(train + validation, args.task)
    from .data_io import get_max_atom_count
    padding = math.ceil(get_max_atom_count(train + validation) * args.padding_multiplier)
    with _output_directory(args.output_dir, args) as output:
        if args.task in ("chg", "all"):
            from .chg import train_chg_model
            train_chg_model(train, validation, padding, args)
        if args.task in ("energy", "all"):
            from .energy import train_energy_model
            train_energy_model(train, validation, None, padding, args)
        if args.task in ("dos", "all"):
            from .dos import train_dos_model
            train_dos_model(train, validation, padding, args)
    print(f"Run files: {output}")


def _infer(args):
    folders = _folders(args.infer_list, "file_loc_test")
    for task in ("chg", "energy", "dos"):
        if getattr(args, f"predict_{task}"):
            checkpoint = getattr(args, f"{task}_checkpoint").resolve(strict=True)
            setattr(args, f"{task}_checkpoint", checkpoint)
    from .data_io import get_max_atom_count, save_charges, save_dos, save_energy
    padding = math.ceil(get_max_atom_count(folders) * args.padding_multiplier)
    chg_model = energy_model = dos_model = None
    if args.predict_chg:
        from .chg import infer_charges, load_pretrained_chg_model
        chg_model = load_pretrained_chg_model(args.chg_checkpoint, padding)
    if args.predict_energy:
        from .energy import infer_energy, load_pretrained_energy_model
        energy_model = load_pretrained_energy_model(args.energy_checkpoint, padding, 369, 369, 369, 369)
    if args.predict_dos:
        from .dos import infer_dos, load_pretrained_dos_model
        dos_model = load_pretrained_dos_model(args.dos_checkpoint, padding)

    with _output_directory(args.output_dir, args) as output:
        for index, folder in enumerate(folders, 1):
            name = f"{index:04d}_{Path(folder).name}"
            print(f"Predicting {folder}")
            if args.predict_chg:
                values = infer_charges(folder, chg_model, padding, args)
                save_charges(values, str(output / f"charges_{name}.txt"))
            if args.predict_energy:
                energy, forces, stress = infer_energy(folder, None, energy_model, padding, args)
                save_energy(energy, forces, stress, str(output / f"energy_{name}.txt"), per_atom=True)
            if args.predict_dos:
                grid, values, vb, cb, gap, _ = infer_dos(folder, None, dos_model, padding, args)
                save_dos(grid, values, vb, cb, gap, str(output / f"dos_{name}.txt"))
    print(f"Predictions: {output}")


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.mode == "check-data":
            folders = _folders(args.file_list, args.column)
            check_inputs(folders, args.task)
            print(f"Checked {len(folders)} structures for task '{args.task}'.")
            return
        for key, value in vars(args).items():
            if isinstance(value, Path):
                setattr(args, key, value.expanduser().resolve())
        if args.mode == "train":
            _train(args)
        else:
            _infer(args)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
