# Preparation Notebooks

- `Data.ipynb`: locate or clone the repository and build deterministic CSV lists.
- `joblib.ipynb`: fit per-element scalers and create a downloadable archive.

Run cells in order. In Colab, the setup cell clones into `/content/DFT` only when the checkout is absent. Locally, launch Jupyter from inside the repository. Generated files go to `runs/`, without deleting or remounting Drive directories. Each preparation command expects a new output directory.

Install the local package before scaler generation. Notebook outputs and execution counts are cleared in Git.
