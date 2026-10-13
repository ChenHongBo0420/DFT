# ML-DFT

PyTorch research code for learning charge-related coefficients, energy, forces, stress, and density of states (DOS) from atom-centered fingerprints of C/H/N/O structures.

Created by Hongbo Chen (Jushua Chen). This repository includes model definitions, training and prediction commands, example structure data, and data-preparation tools.

## Layout

| Path | Purpose |
| --- | --- |
| `dftpy/chg.py` | Element-specific charge/coefficient networks |
| `dftpy/energy.py` | Energy, force, and stress model |
| `dftpy/dos.py` | DOS and band-edge model |
| `dftpy/fp.py` | Atom-centered fingerprints and normalization |
| `dftpy/data_io.py`, `dftpy/paths.py` | Input formats, structure paths, and padding |
| `dftpy/config.py` | Shared fingerprint defaults |
| `dftpy/scalers/` | Four existing per-element scaler files |
| `dftpy/scripts/` | Scaler-generation commands |
| `database/` | 19 example structures and their reference property files |
| `DFT_CSVs/`, `DFT_CSVs_v2/` | Two preserved example split collections |
| `scripts/build_splits.py` | Deterministic split generation |
| `Colab/` | Data and scaler preparation notebooks |
| `tests/` | Fast path, CLI, and repository tests |

## Install

Use Python 3.10 or newer. Clone this repository and install the local checkout in a dedicated environment:

```bash
git clone https://github.com/ChenHongBo0420/DFT.git
cd DFT
python -m pip install -e .
```

The Python package and command retain the historical name `dftpy`. Install **this checkout** using the command above. Choose a PyTorch build appropriate for your CPU or CUDA environment. Optional plotting and Jupyter dependencies:

```bash
python -m pip install -e ".[plot,notebooks]"
```

## Check the Example Data

These commands use only the Python standard library and do not load models or run training:

```bash
python -m dftpy check-data --list DFT_CSVs/Train_all.csv
python -m dftpy check-data --list DFT_CSVs/Train_energy.csv --task energy
python -m dftpy check-data --list DFT_CSVs/Train_dos.csv --task dos
```

CSV paths are resolved relative to the CSV file, not the current working directory. Both existing CSV collections retain their original sample assignments. See [data formats](docs/data.md).

## Prepare Splits and Scalers

To create a new seeded split collection:

```bash
python scripts/build_splits.py --database database --output runs/splits --seed 42
```

This first splits structures into train/validation/test groups, then filters each group by available energy or DOS labels. Use the generated task-specific lists only when their train and validation subsets are nonempty.

Fit new scalers from a training list with energy, force, and stress files:

```bash
python -m dftpy.scripts.generate_scalers --train-csv DFT_CSVs/Train_energy.csv --out-dir runs/scalers
```

The command saves `Scale_model_C.joblib`, `Scale_model_H.joblib`, `Scale_model_N.joblib`, `Scale_model_O.joblib`, and `scaler_config.json`. Pass the same scaler directory during training and inference. Existing bundled scaler files remain unchanged. Only load joblib files from a source you trust.

## Train

Energy, force, and stress:

```bash
python -m dftpy train --task energy --train-list DFT_CSVs/Train_energy.csv --val-list DFT_CSVs/Val_energy.csv --scaler-dir runs/scalers --output-dir runs/energy
```

Density of states and band edges:

```bash
python -m dftpy train --task dos --train-list DFT_CSVs/Train_dos.csv --val-list DFT_CSVs/Val_dos.csv --scaler-dir runs/scalers --output-dir runs/dos
```

For charge training, prepare the four per-element `Coef_*.npy` target arrays described in [data formats](docs/data.md), then use `--task chg`. `--task all` expects every selected structure to have all three tasks' reference inputs. Energy and DOS training use their own fingerprint inputs and do not require a charge-model checkpoint.

Common options include `--epochs`, `--batch-size`, `--learning-rate`, `--patience`, `--scaler-dir`, and the fingerprint settings listed by `python -m dftpy train --help`.

Each run uses a new output directory. It records the command settings in `run.json` and saves the task-specific best-validation weights as `best_chg.pth`, `newEmodel.pth`, or `best_dos.pth`.

## Predict

After training, use the corresponding checkpoint and the same fingerprint/scaler settings:

```bash
python -m dftpy infer --infer-list DFT_CSVs/predict.csv --predict-energy --energy-checkpoint runs/energy/newEmodel.pth --scaler-dir runs/scalers --output-dir runs/energy-predictions
python -m dftpy infer --infer-list DFT_CSVs/predict.csv --predict-dos --dos-checkpoint runs/dos/best_dos.pth --scaler-dir runs/scalers --output-dir runs/dos-predictions
```

Charge prediction uses `--predict-chg --chg-checkpoint PATH`. Combine prediction flags when using compatible checkpoints. `--plot-dos` enables DOS plots with the optional plotting dependency. New structures need POSCAR files; energy/DOS reference targets are used during training, not inference. Trained model checkpoints are supplied by your own runs and are not bundled with the repository.

## Notes and Tests

The original network architectures and bundled datasets are retained. Shared CLI fingerprint defaults now follow the original `inp_params.py` settings. See [workflow and compatibility notes](docs/workflows.md) for parameter and output conventions.

```bash
python -m unittest discover -s tests -v
```

These tests cover repository plumbing without installing ML dependencies or running model training. They do not measure prediction accuracy.

## License

Code is provided under the [MIT License](LICENSE). Original example data and serialized scalers are retained from this repository; their provenance is described in [database/README.md](database/README.md) and [dftpy/scalers/README.md](dftpy/scalers/README.md).
