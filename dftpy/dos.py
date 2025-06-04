# dftpy/dos.py – revised version
# -----------------------------------------------------------------------------
#  Machine‑learning prediction of DOS curves & band edges for C/H/N/O materials
#  The code has been cleaned up to fix shape mismatches (341 points instead of
#  343), duplicate variable names, and incorrect calls to `dos_mask`.
# -----------------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List

from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Poscar

from .fp import fp_chg_norm
from .data_io import chg_data, dos_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
DOS_POINTS = 341  # number of energy bins (‑33 → 1 eV step 0.1 eV)
ENERGY_GRID = np.linspace(-33.0, 1.0, DOS_POINTS, dtype=np.float32)

# ---------------------------------------------------------------------------
#  Helpers for reading reference data
# ---------------------------------------------------------------------------

def _read_dos(folder: str, total_elec: float) -> Tuple[np.ndarray, float, float]:
    """Read *dos* + *VB_CB* files and return (norm_dos, VB, CB)."""
    dos_vals = np.loadtxt(Path(folder) / "dos", dtype=np.float32)
    if dos_vals.size != DOS_POINTS:
        raise ValueError(
            f"Expected {DOS_POINTS} DOS points but got {dos_vals.size} in {folder}/dos"
        )
    norm_dos = dos_vals / total_elec

    vb_cb = np.loadtxt(Path(folder) / "VB_CB", dtype=np.float32)
    VB, CB = abs(vb_cb[0]), abs(vb_cb[1])
    return norm_dos, VB, CB


# ---------------------------------------------------------------------------
#  Single‑atom subnetworks
# ---------------------------------------------------------------------------
class _DOSSubNetCNO(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        hidden = 600
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, DOS_POINTS),
        )
        self.conv = nn.Conv1d(1, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B*P, feat)
        h = self.mlp(x).view(-1, 1, DOS_POINTS)          # (B*P,1,341)
        h = self.conv(h).mean(dim=1)                     # (B*P,341)
        return h


class _DOSSubNetH(_DOSSubNetCNO):
    pass  # identical architecture; only input_dim differs when instantiated


# ---------------------------------------------------------------------------
#  Combined DOS model
# ---------------------------------------------------------------------------
class DOSModel(nn.Module):
    def __init__(self, padding_size: int):
        super().__init__()
        self.padding_size = padding_size

        self.netC = _DOSSubNetCNO(360 + 340)
        self.netH = _DOSSubNetH(360 + 208)
        self.netN = _DOSSubNetCNO(360 + 340)
        self.netO = _DOSSubNetCNO(360 + 340)

        # global head predicting [‑VB, ‑CB]
        self.band_head = nn.Sequential(
            nn.Linear(DOS_POINTS, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 2)
        )

    # ---------------------------------------------------------------------
    def forward(
        self,
        X_C: torch.Tensor,
        X_H: torch.Tensor,
        X_N: torch.Tensor,
        X_O: torch.Tensor,
        total_elec: torch.Tensor,
        C_m: torch.Tensor,
        H_m: torch.Tensor,
        N_m: torch.Tensor,
        O_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        X_*         : (B,P,feat)
        total_elec  : (B,1)
        *_m (masks) : (B,P,DOS_POINTS)
        Returns      norm_dos (B,DOS_POINTS), vbcb (B,2)
        """
        B, P, _ = X_C.shape

        def _flatten(t: torch.Tensor) -> torch.Tensor:
            return t.view(B * P, -1)

        flatC = self.netC(_flatten(X_C)).view(B, P, DOS_POINTS)
        flatH = self.netH(_flatten(X_H)).view(B, P, DOS_POINTS)
        flatN = self.netN(_flatten(X_N)).view(B, P, DOS_POINTS)
        flatO = self.netO(_flatten(X_O)).view(B, P, DOS_POINTS)

        # apply masks & sum over atoms ------------------------------------
        total_dos = (
            (flatC * C_m).sum(dim=1)
            + (flatH * H_m).sum(dim=1)
            + (flatN * N_m).sum(dim=1)
            + (flatO * O_m).sum(dim=1)
        )  # (B, DOS_POINTS)

        norm_dos = total_dos / total_elec  # broadcasting (B,1)
        vbcb = self.band_head(norm_dos)    # (B,2)
        return norm_dos, vbcb


# ---------------------------------------------------------------------------
#  Loss function
# ---------------------------------------------------------------------------
class DOSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_dos, true_dos, pred_vbcb, true_vbcb):
        return 1000 * self.mse(pred_dos, true_dos) + self.mse(pred_vbcb, true_vbcb)


# ---------------------------------------------------------------------------
#  Utility – build scaler paths once
# ---------------------------------------------------------------------------
SCALER_PATHS = tuple((Path(__file__).with_suffix("").parent / "scalers" / f"Scale_model_{e}.joblib") for e in ("C", "H", "N", "O"))


# ---------------------------------------------------------------------------
#  Dataset wrapper
# ---------------------------------------------------------------------------
class DOSDataset(Dataset):
    def __init__(self, data: Tuple[np.ndarray, ...]):
        (
            self.X_C,
            self.X_H,
            self.X_N,
            self.X_O,
            self.total_e,
            self.C_m,
            self.H_m,
            self.N_m,
            self.O_m,
            self.Prop_dos,
            self.VB_CB,
        ) = [torch.from_numpy(t).float() for t in data]

    def __len__(self):
        return self.X_C.shape[0]

    def __getitem__(self, idx):
        return (
            self.X_C[idx],
            self.X_H[idx],
            self.X_N[idx],
            self.X_O[idx],
            self.total_e[idx],
            self.C_m[idx],
            self.H_m[idx],
            self.N_m[idx],
            self.O_m[idx],
            self.Prop_dos[idx],
            self.VB_CB[idx],
        )


# ---------------------------------------------------------------------------
#  Training utilities
# ---------------------------------------------------------------------------

def _prepare_single(folder: str, padding_size: int, args) -> Tuple[np.ndarray, ...]:
    """Prepare one structure for DOS training / inference."""

    # -------- read structure & element counts ----------------------------
    struct = Poscar.from_file(Path(folder) / "POSCAR").structure
    elem_dict = {"C": 0, "H": 0, "N": 0, "O": 0}
    for s in struct:
        sym = s.specie.symbol
        if sym in elem_dict:
            elem_dict[sym] += 1
    at_elem = [elem_dict[e] for e in ("C", "H", "N", "O")]

    # -------- raw fingerprints (4 × 3-D) ---------------------------------
    from .fp import fp_atom  # local import to avoid circular deps

    dset, basis_mat, *_ = fp_atom(
        struct,
        args.grid_spacing,
        args.cut_off_rad,
        args.widest_gaussian,
        args.narrowest_gaussian,
        args.num_gamma,
    )

    (
        X_3D1, X_3D2, X_3D3, X_3D4,  # fingerprints
        _, _, _, _,                  # unused grads
        C_m_1, H_m_1, N_m_1, O_m_1,  # masks
    ) = chg_data(dset, basis_mat, *at_elem, padding_size)

    # -------- charge-derived coefficients --------------------------------
    coef_paths = [Path(folder) / f"Coef_{e}.npy" for e in ("C", "H", "N", "O")]
    if not all(p.exists() for p in coef_paths):
        raise FileNotFoundError("Missing Coef_*.npy — run charge→coef export first.")
    Coef_C, Coef_H, Coef_N, Coef_O = [np.load(p).astype(np.float32) for p in coef_paths]

    # -------- concatenate & normalise ------------------------------------
    X_C_aug, X_H_aug, X_N_aug, X_O_aug = fp_chg_norm(
        X_3D1, X_3D2, X_3D3, X_3D4,        # 4 × (P, feat_fp)
        Coef_C, Coef_H, Coef_N, Coef_O,    # 4 × (P, coeff_dim) – same 2-D dims
        padding_size,
        SCALER_PATHS,
    )

    # -------- masks (DOS-specific) ---------------------------------------
    C_d, H_d, N_d, O_d = dos_mask(C_m_1, H_m_1, N_m_1, O_m_1, padding_size)

    # -------- ground-truth DOS & band edges ------------------------------
    elec_per_atom = {6: 4, 1: 1, 7: 5, 8: 6}
    total_elec = np.array([elec_per_atom[z] for z in struct.atomic_numbers], dtype=np.float32).sum()
    Prop_dos, VB, CB = _read_dos(folder, total_elec)
    VB_CB = np.array([-VB, -CB], dtype=np.float32)

    # -------- return (each item shape matches fp_chg_norm expectations) ---
    return (
        X_C_aug, X_H_aug, X_N_aug, X_O_aug,              # 4 × (P, feat_all)
        np.array([total_elec], dtype=np.float32),        # (1,)
        C_d, H_d, N_d, O_d,                             # 4 × (P, DOS_POINTS)
        Prop_dos, VB_CB,
    )

# ---------------------------------------------------------------------------
#  Public helpers (train / load / infer)
# ---------------------------------------------------------------------------

def init_DOSmod(padding_size: int) -> DOSModel:
    return DOSModel(padding_size).to(DEVICE)


def load_pretrained_dos_model(path: str, padding_size: int) -> DOSModel:
    model = init_DOSmod(padding_size)
    if not Path(path).exists():
        raise FileNotFoundError(f"Trained DOS model not found: {path}")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# Training ---------------------------------------------------------------

def train_dos_model(train_folders: List[str], val_folders: List[str], padding_size: int, args):
    """Train DOS model and write *best_dos.pth* in cwd."""

    def prepare(f_list: List[str]):
        return tuple(np.stack(t) for t in zip(*[_prepare_single(f, padding_size, args) for f in f_list]))

    train_data = prepare(train_folders)
    val_data = prepare(val_folders)

    train_loader = DataLoader(DOSDataset(train_data), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DOSDataset(val_data), batch_size=args.batch_size, shuffle=False)

    model = init_DOSmod(padding_size)
    criterion = DOSLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val, best_epoch = float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        # ---- train -----------------------------------------------------
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            (
                X_C, X_H, X_N, X_O, tot_e, C_m, H_m, N_m, O_m, tgt_dos, tgt_vbcb,
            ) = [b.to(DEVICE) for b in batch]

            optimizer.zero_grad()
            pred_dos, pred_vbcb = model(X_C, X_H, X_N, X_O, tot_e, C_m, H_m, N_m, O_m)
            loss = criterion(pred_dos, tgt_dos, pred_vbcb, tgt_vbcb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()

        # ---- validation ----------------------------------------------
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                (
                    X_C, X_H, X_N, X_O, tot_e, C_m, H_m, N_m, O_m, tgt_dos, tgt_vbcb,
                ) = [b.to(DEVICE) for b in batch]
                pred_dos, pred_vbcb = model(X_C, X_H, X_N, X_O, tot_e, C_m, H_m, N_m, O_m)
                vl_loss += criterion(pred_dos, tgt_dos, pred_vbcb, tgt_vbcb).item()

        tr_loss /= len(train_loader)
        vl_loss /= len(val_loader)
        print(f"Epoch {epoch}: train {tr_loss:.6f}  val {vl_loss:.6f}")

        if vl_loss < best_val:
            best_val, best_epoch = vl_loss, epoch
            torch.save(model.state_dict(), "best_dos.pth")
        elif epoch - best_epoch >= args.patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete – best val {best_val:.6f} at epoch {best_epoch}")


# Inference --------------------------------------------------------------

def infer_dos(folder: str, chg_model: nn.Module, dos_model: DOSModel, padding_size: int, args):
    """Predict DOS curve and band edges for a single POSCAR folder."""

    # Compute augmented fingerprints & masks (re‑using private helper)
    (
        X_C, X_H, X_N, X_O, tot_e, C_m, H_m, N_m, O_m, _, _,
    ) = _prepare_single(folder, padding_size, args)

    X_C_t = torch.from_numpy(X_C[np.newaxis, :, :]).float().to(DEVICE)
    X_H_t = torch.from_numpy(X_H[np.newaxis, :, :]).float().to(DEVICE)
    X_N_t = torch.from_numpy(X_N[np.newaxis, :, :]).float().to(DEVICE)
    X_O_t = torch.from_numpy(X_O[np.newaxis, :, :]).float().to(DEVICE)
    tot_e_t = torch.from_numpy(tot_e.reshape(1, 1)).float().to(DEVICE)
    C_m_t = torch.from_numpy(C_m[np.newaxis, :, :]).float().to(DEVICE)
    H_m_t = torch.from_numpy(H_m[np.newaxis, :, :]).float().to(DEVICE)
    N_m_t = torch.from_numpy(N_m[np.newaxis, :, :]).float().to(DEVICE)
    O_m_t = torch.from_numpy(O_m[np.newaxis, :, :]).float().to(DEVICE)

    dos_model.eval()
    with torch.no_grad():
        norm_dos_t, vbcb_t = dos_model(X_C_t, X_H_t, X_N_t, X_O_t, tot_e_t, C_m_t, H_m_t, N_m_t, O_m_t)

    norm_dos = norm_dos_t.cpu().numpy().squeeze(0)
    vbcb = vbcb_t.cpu().numpy().squeeze(0)
    VB, CB = -vbcb[0], -vbcb[1]
    BG = CB - VB

    if args.plot_dos:
        import matplotlib.pyplot as plt
        plt.plot(ENERGY_GRID, norm_dos)
        plt.axvline(VB, color="b", linestyle=":")
        plt.axvline(CB, color="g")
        plt.axvline(0.0, color="k", linestyle="--")
        plt.xlabel("Energy (eV)")
        plt.ylabel("DOS (arb. unit)")
        plt.tight_layout()
        plt.savefig(Path("dos_plot_" + Path(folder).name + ".png"), dpi=300)
        plt.close()

    return ENERGY_GRID, norm_dos, VB, CB, BG, None
