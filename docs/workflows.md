# Workflows and Compatibility

## Components

`fp.py` constructs radial, dipole, and quadrupole fingerprints and local basis vectors. `data_io.py` pads per-element blocks and reads reference properties. The three model modules retain the existing element-specific PyTorch networks and training losses.

The command-line interface selects a task and delegates model operations to those modules. Path discovery and CSV parsing live in `paths.py`, so `--help` and `check-data` work without importing PyTorch or pymatgen.

## Fingerprint Settings

The shared defaults in `dftpy/config.py` are taken from the original `inp_params.py`:

| Option | Default |
| --- | --- |
| `--grid-spacing` | 0.7 |
| `--cut-off-rad` | 5.0 |
| `--widest-gaussian` | 6.0 |
| `--narrowest-gaussian` | 0.5 |
| `--num-gamma` | 18 |

The old CLI used different defaults (1.0, 6.0, 2.0, 0.5, 10). Pass those values explicitly when using a checkpoint trained with that configuration. Use identical settings for scaler fitting, training, and prediction. The original 360-feature padding/truncation convention is retained for model inputs. `inp_params.py` remains as a historical reference; editing it does not change CLI settings.

## Outputs

The CLI requires a new output directory and stores its settings in `run.json`. Training keeps the original checkpoint filenames, but places them inside that directory. Inference accepts explicit checkpoint locations and gives prediction tables numbered structure-based names. Input structures and coefficient targets are not overwritten by the CLI.

Energy training retains the original absolute-energy-per-atom target convention. The inference adapter retains its negative per-atom prediction convention and uses the same normalized fingerprint-plus-basis representation as training. Forces are concatenated in C/H/N/O order. DOS inference builds inputs without reading reference DOS or band-edge targets.

The charge-model definitions and charge integration formula are retained from the research implementation. Repository checks cover interfaces and file handling; numerical validation belongs to the corresponding research workflow.

## Scalers

`generate_scalers.py` uses the energy-data preparation path and fits one MaxAbsScaler per element on training rows, including the original padding convention. It records actual feature counts in `scaler_config.json`. `generate_700scalers.py` is a compatibility wrapper using the historical `Scale700` filename prefix; that prefix is not an assertion of the generated feature count. The main models load `Scale_model_*.joblib` filenames.

Joblib serializations depend on their Python/scikit-learn environment. For a new experiment, fit scalers in the environment used for that experiment and retain them alongside its checkpoint. The four existing scaler files are included in built distributions and have not been refitted during repository organization.

## Organization Changes

- Kept existing database content and both historical CSV split collections.
- Centralized CSV/POSCAR handling and fingerprint defaults.
- Added explicit checkpoint, scaler, and output-directory options.
- Removed the redundant charge-inference pass that wrote into dataset directories before DOS training.
- Connected energy CLI options to its preparation and optimizer settings.
- Kept scaler scripts import-safe and replaced hard-coded Drive paths with arguments.
- Cleared notebook outputs and replaced monolithic Colab setup cells with separate steps.
- Removed empty placeholder files and commented-out duplicate DOS implementations.

Prior source versions remain available in Git history.
