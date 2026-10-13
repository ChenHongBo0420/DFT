# Bundled Scalers

This directory contains the four original `Scale_model_C/H/N/O.joblib` files from this repository. They are preserved byte-for-byte and included in the Python package.

To fit scalers for a new run, use the [scaler-generation command](../../README.md#prepare-splits-and-scalers) and pass the generated directory with `--scaler-dir`. Generated files are kept outside the source directory by default in the documented workflow.

Load joblib files only from trusted sources. This cleanup does not infer fitting-data provenance or refit these existing serializations.
