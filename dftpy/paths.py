"""Dependency-free path handling shared by the CLI and research modules."""

import csv
from pathlib import Path


def read_file_list(csv_path, col="files"):
    """Read paths relative to the CSV location, preserving the listed order."""
    path = Path(csv_path).expanduser().resolve(strict=True)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or col not in reader.fieldnames:
            raise ValueError(f"{path}: expected a CSV column named {col!r}.")
        folders = []
        for row in reader:
            value = (row.get(col) or "").strip()
            if not value:
                continue
            folder = Path(value).expanduser()
            if not folder.is_absolute():
                folder = path.parent / folder
            folders.append(str(folder.resolve()))
    if not folders:
        raise ValueError(f"{path}: no structure paths in column {col!r}.")
    return folders


def find_poscar(location):
    """Accept a structure directory, a POSCAR file, or POSCAR/POSCAR."""
    path = Path(location).expanduser().resolve()
    if path.is_file() and path.name.upper() == "POSCAR":
        return path
    for candidate in (path / "POSCAR", path / "POSCAR" / "POSCAR"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No POSCAR file at or below {path}.")


def structure_folder(location):
    """Return the sample root containing POSCAR and reference property files."""
    poscar = find_poscar(location)
    return poscar.parent.parent if poscar.parent.name.upper() == "POSCAR" else poscar.parent


def require_files(folders, names):
    missing = [str(Path(folder) / name) for folder in folders
               for name in names if not (Path(folder) / name).is_file()]
    if missing:
        raise FileNotFoundError("Required input files are missing:\n" + "\n".join(missing))


def scaler_paths(directory=None):
    base = Path(directory) if directory is not None else Path(__file__).parent / "scalers"
    return tuple(str(base / f"Scale_model_{element}.joblib") for element in "CHNO")
