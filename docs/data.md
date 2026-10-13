# Data Formats

## CSV Lists

Training and validation lists have one `files` column. Prediction lists have one `file_loc_test` column. Paths may be absolute or relative to the CSV file's directory. Quoted paths containing commas are supported.

For a list stored in `DFT_CSVs/`:

```csv
files
../database/chains/R1000001
../database/molecules/benzene
```

A path may name a sample directory, a POSCAR file, or a nested `POSCAR/POSCAR` layout. The shared resolver locates the structure and its sample root. Reference properties belong in the sample root.

## Per-Structure Files

| File | Used by | Layout |
| --- | --- | --- |
| `POSCAR` | All tasks | VASP structure file |
| `energy` | Energy training | One total reference-energy value |
| `forces` | Energy training | One three-component force row per atom |
| `stress` | Energy training | Six stress components |
| `dos` | DOS training | 341 DOS values on the model's -33 to 1 eV grid |
| `VB_CB` | DOS training | Two band-edge values |
| `Coef_C.npy`, `Coef_N.npy`, `Coef_O.npy` | Charge training | Per-element target arrays with 340 columns |
| `Coef_H.npy` | Charge training | Per-hydrogen target array with 208 columns |

Provide arrays for all four elements for charge training; an absent element can use an empty array with the corresponding column count. Charge targets are required explicitly and are not replaced by zero labels. The included structure examples contain energy/force/stress data, with DOS/band-edge files for chains and molecules. Charge coefficient targets are supplied separately by the user.

The element-specific packing follows the original C, H, N, O grouping. Prepare structures, fingerprints, force rows, and coefficient rows in that same order. The model modules operate on these four elements.

## Existing Splits

`DFT_CSVs/` and `DFT_CSVs_v2/` are separate historical split collections, not duplicate files to combine. Their row order and assignments are preserved; only the old Colab-specific path prefix was replaced with a portable relative prefix.

For new splits, `scripts/build_splits.py` assigns each structure to a single global train, validation, or test group before task-specific filtering. The default seed is 42. An empty task subset means that split does not have enough examples with that task's reference files.
