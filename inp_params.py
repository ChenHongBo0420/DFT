# inp_params.py

"""
Configuration file for DFTpy.
Contains default hyperparameters and settings for training and inference.
"""

# ----------------------------
# Control flags for training
# ----------------------------
train_e = False        # Whether to train the energy model
ert_epochs = 50        # Number of epochs to train the energy model
ert_batch_size = 10    # Batch size for energy training
ert_patience = 10      # Early-stopping patience for energy model

train_dos = False      # Whether to train the DOS model
drt_epochs = 50        # Number of epochs to train the DOS model
drt_batch_size = 10    # Batch size for DOS training
drt_patience = 10      # Early-stopping patience for DOS model

new_weights_e = False      # If True, load new weights for energy model instead of default
new_weights_dos = False    # If True, load new weights for DOS model instead of default

# ----------------------------
# Control flags for testing/inference
# ----------------------------
test_chg = True       # Whether to predict charges during inference
test_e = True         # Whether to predict energy during inference
test_dos = True       # Whether to predict DOS during inference

# ----------------------------
# Fingerprinting parameters
# ----------------------------
fp_file = "fp"             # Filename or prefix for fingerprinting (placeholder)
cut_off_rad = 5.0          # Cutoff radius for fingerprint generation
batch_size_fp = 10000      # Batch size used when computing fingerprints
widest_gaussian = 6.0      # Widest Gaussian width (sigma)
narrowest_gaussian = 0.5   # Narrowest Gaussian width (sigma)
num_gamma = 18             # Number of gamma values (i.e., number of fingerprints per atom)

# ----------------------------
# Visualization / output settings
# ----------------------------
plot_dos = True     # Whether to generate DOS plots during inference
tot_chg = False     # If True, include core charge when computing total charge
comp_chg = False    # If True, compute charge comparison metrics
write_chg = True    # If True, write predicted charge to output files
ref_chg = False     # If True, read reference charge data for comparison

# ----------------------------
# Grid settings (for DOS/charge grid)
# ----------------------------
grid_spacing = 0.7  # Spacing between grid points when sampling the charge/DOS
