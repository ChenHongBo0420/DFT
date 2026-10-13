"""Fit per-element MaxAbsScaler files using a training list."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dftpy.config import FINGERPRINT_DEFAULTS
from dftpy.paths import read_file_list, require_files, structure_folder


def main(argv=None, default_prefix="Scale_model"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path, help="New directory for generated scalers.")
    for key, value in FINGERPRINT_DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args(argv)
    import math
    if any(not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0
           for key in FINGERPRINT_DEFAULTS):
        parser.error("Fingerprint parameters must be positive and finite.")
    if args.narrowest_gaussian > args.widest_gaussian:
        parser.error("narrowest-gaussian must not exceed widest-gaussian.")
    if args.out_dir.exists():
        parser.error("Choose a new output directory; existing scalers are not overwritten.")
    folders = [str(structure_folder(path)) for path in read_file_list(args.train_csv)]
    require_files(folders, ("energy", "forces", "stress"))

    import joblib
    from sklearn.preprocessing import MaxAbsScaler
    from dftpy.data_io import get_max_atom_count, get_efp_data, pad_efp_data

    padding = get_max_atom_count(folders)
    _, forces, _, fingerprints, basis, _, _, elements = get_efp_data(folders, args)
    packed = pad_efp_data(elements, fingerprints, forces, basis, padding)
    fitted = {}
    for element, block in zip("CHNO", packed[4:8]):
        flat = block.reshape(-1, block.shape[-1])
        fitted[element] = MaxAbsScaler().fit(flat)
        print(f"{element}: fitted {flat.shape[1]} features from {flat.shape[0]} padded rows.")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for element, scaler in fitted.items():
        joblib.dump(scaler, args.out_dir / f"{default_prefix}_{element}.joblib")
    metadata = {"training_list": str(args.train_csv.resolve()), "samples": len(folders),
                "padding_size": padding, "filename_prefix": default_prefix,
                "fingerprints": {key: getattr(args, key) for key in FINGERPRINT_DEFAULTS},
                "feature_counts": {key: int(value.n_features_in_) for key, value in fitted.items()}}
    (args.out_dir / "scaler_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Scalers saved to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
