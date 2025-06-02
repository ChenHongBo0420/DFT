# -*- coding: utf-8 -*-
"""dftpy.chg
~~~~~~~~~~~~~~~~

PyTorch implementation of the atomic–charge network used in **DFTpy**.
This version removes the mandatory dependency on *CHGCAR* files during
training so that structures without charge–density files can still be
used.  In addition, the stale imports of ``fp_chg_norm`` and
``fp_norm``—which no longer exist in *dftpy.fp*—have been dropped so that
an ``ImportError`` is not raised when the package is imported.

Key modifications compared with the original script
---------------------------------------------------
1. **prepare_coef_data()** – the call to :pyfunc:`chg_ref` has been
   removed because the returned values were never used later on.  This
   eliminates the *FileNotFoundError* that appeared when a directory did
   not contain a *CHGCAR* file.
2. Super‑fluous imports of ``fp_chg_norm`` and ``fp_norm`` have been
   deleted.
3. General tidying: PEP 8 compliant spacing, typing hints, and doc‑strings
   in English for better readability.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pymatgen.io.vasp.outputs import Chgcar, Poscar
from torch.utils.data import DataLoader, Dataset

from .data_io import chg_data
from .fp import fp_atom

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Utility helpers
# =============================================================================

def chg_ref(folder: str | Path, vol: float, supercell) -> Tuple[np.ndarray, np.ndarray, Sequence[int]]:
    """Read *CHGCAR* and return the flattened charge density.

    Notes
    -----
    Although this helper is still provided (because it is referenced by
    :pyfunc:`infer_charges`), it is **no longer** required during model
    training.
    """
    chgcar = Chgcar.from_file(Path(folder) / "CHGCAR")

    density = chgcar.data["total"].flatten(order="F") / vol
    centering = [0.5 - 1 / 2] * 3

    # Build fractional grids then convert to Cartesian coordinates.
    xg = np.array(chgcar.get_axis_grid(0)) / supercell.lattice.a + centering[0]
    yg = np.array(chgcar.get_axis_grid(1)) / supercell.lattice.b + centering[1]
    zg = np.array(chgcar.get_axis_grid(2)) / supercell.lattice.c + centering[2]
    xx, yy, zz = np.meshgrid(xg, yg, zg, indexing="ij")

    pts = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T  # (N, 3)
    coords = pts @ supercell.lattice.matrix                 # fractional → Cartesian
    num_pts = [len(xg), len(yg), len(zg)]

    return coords, density, num_pts

# =============================================================================
# Neural‑network blocks
# =============================================================================

class _SingleAtomChargeNetCNO(nn.Module):
    """360‑dim fingerprint → 340 charge‑expansion coefficients (C/N/O)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(360, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, 340),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B*P, 360)
        out = self.net(x)
        exp_part = torch.abs(out[:, :93])    # enforce positive exponents
        coef_part = out[:, 93:]
        return torch.cat([exp_part, coef_part], dim=1)


class _SingleAtomChargeNetH(nn.Module):
    """360‑dim fingerprint → 208 charge‑expansion coefficients (H)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(360, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, 208),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B*P, 360)
        out = self.net(x)
        exp_part = torch.abs(out[:, :58])
        coef_part = out[:, 58:]
        return torch.cat([exp_part, coef_part], dim=1)


class ChargeModel(nn.Module):
    """Four‑branch network producing atomic charge coefficients."""

    def __init__(self, padding_size: int):
        super().__init__()
        self.padding_size = padding_size
        self.netC = _SingleAtomChargeNetCNO().to(DEVICE)
        self.netH = _SingleAtomChargeNetH().to(DEVICE)
        self.netN = _SingleAtomChargeNetCNO().to(DEVICE)
        self.netO = _SingleAtomChargeNetCNO().to(DEVICE)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        X_C: torch.Tensor,
        X_H: torch.Tensor,
        X_N: torch.Tensor,
        X_O: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, P, _ = X_C.shape
        C_flat, H_flat = X_C.view(B * P, 360), X_H.view(B * P, 360)
        N_flat, O_flat = X_N.view(B * P, 360), X_O.view(B * P, 360)

        outC = self.netC(C_flat).view(B, P, -1)
        outH = self.netH(H_flat).view(B, P, -1)
        outN = self.netN(N_flat).view(B, P, -1)
        outO = self.netO(O_flat).view(B, P, -1)
        return outC, outH, outN, outO

# =============================================================================
# Convenience wrappers
# =============================================================================

def init_chgmod(padding_size: int) -> ChargeModel:
    """Return an un‑trained :class:`ChargeModel`."""
    return ChargeModel(padding_size)


def load_pretrained_chg_model(checkpoint_path: str | Path, padding_size: int) -> ChargeModel:
    """Load weights and put the model into *eval* mode."""
    model = init_chgmod(padding_size).to(DEVICE)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ChargeModel 权重不存在: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model

# =============================================================================
# Dataset & Data‑loader
# =============================================================================

class ChargeDataset(Dataset):
    """Simple *torch.utils.data.Dataset* wrapper holding features & labels."""

    def __init__(
        self,
        X_C: np.ndarray,
        X_H: np.ndarray,
        X_N: np.ndarray,
        X_O: np.ndarray,
        Coef_C: np.ndarray,
        Coef_H: np.ndarray,
        Coef_N: np.ndarray,
        Coef_O: np.ndarray,
    ) -> None:
        self.X_C = torch.from_numpy(X_C).float()
        self.X_H = torch.from_numpy(X_H).float()
        self.X_N = torch.from_numpy(X_N).float()
        self.X_O = torch.from_numpy(X_O).float()
        self.Coef_C = torch.from_numpy(Coef_C).float()
        self.Coef_H = torch.from_numpy(Coef_H).float()
        self.Coef_N = torch.from_numpy(Coef_N).float()
        self.Coef_O = torch.from_numpy(Coef_O).float()

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return self.X_C.shape[0]

    def __getitem__(self, idx: int):  # noqa: D401
        return (
            self.X_C[idx], self.X_H[idx], self.X_N[idx], self.X_O[idx],
            self.Coef_C[idx], self.Coef_H[idx], self.Coef_N[idx], self.Coef_O[idx],
        )

# =============================================================================
# Training loop
# =============================================================================

_MSE = nn.MSELoss()

def _pad_coef(arr: np.ndarray, target_rows: int) -> np.ndarray:
    """Pad coefficient matrix to *target_rows* × *dim*."""
    if arr.size == 0:
        return np.zeros((target_rows, arr.shape[1] if arr.ndim == 2 else arr.size), dtype=np.float32)
    padded = np.zeros((target_rows, arr.shape[1]), dtype=np.float32)
    n = min(arr.shape[0], target_rows)
    padded[:n] = arr[:n]
    return padded


def train_chg_model(
    train_folders: List[str],
    val_folders: List[str],
    padding_size: int,
    args,
):
    """Train *ChargeModel* and save the best checkpoint as *best_chg.pth*."""

    def prepare_coef_data(folders: Sequence[str]):
        X_C_l, X_H_l, X_N_l, X_O_l = [], [], [], []
        C_l, H_l, N_l, O_l = [], [], [], []

        for folder in folders:
            folder = Path(folder)
            poscar = Poscar.from_file(folder / "POSCAR")
            struct = poscar.structure

            dset, basis_mat, _, _, at_elem = fp_atom(
                struct,
                args.grid_spacing,
                args.cut_off_rad,
                args.widest_gaussian,
                args.narrowest_gaussian,
                args.num_gamma,
            )

            X_3D1, X_3D2, X_3D3, X_3D4, *_ = chg_data(
                dset, basis_mat, at_elem[0], at_elem[1], at_elem[2], at_elem[3], padding_size
            )
            X_C_l.append(X_3D1), X_H_l.append(X_3D2)
            X_N_l.append(X_3D3), X_O_l.append(X_3D4)

            # --------------------------------------------------------------
            # Ground‑truth coefficients (pre‑computed & stored as .npy)
            # --------------------------------------------------------------
            coef_C = np.load(folder / "Coef_C.npy") if (folder / "Coef_C.npy").exists() else np.zeros((0, 340))
            coef_H = np.load(folder / "Coef_H.npy") if (folder / "Coef_H.npy").exists() else np.zeros((0, 208))
            coef_N = np.load(folder / "Coef_N.npy")
