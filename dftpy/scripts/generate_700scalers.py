"""Compatibility entry point retaining the historical Scale700 filename prefix."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dftpy.scripts.generate_scalers import main

if __name__ == "__main__":
    main(default_prefix="Scale700")
