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
            nn.Linear(hidden, hidden),   nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden),   nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden),   nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, DOS_POINTS),
        )
        self.conv = nn.Conv1d(1, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*P, input_dim)
        h = self.mlp(x).view(-1, 1, DOS_POINTS)  # → (B*P, 1, 341)
        h = self.conv(h).mean(dim=1)             # → (B*P, 341)
        return h

# class _DOSSubNetCNO(nn.Module):
#     """
#     Single-atom subnetwork for C, N, and O.
#     输入维度应为 (360 + DOS_POINTS)，即指纹 360 维 + 掩码 341 维 = 701 维。
#     """

#     def __init__(self, input_dim: int):
#         super().__init__()
#         hidden = 600

#         # 定义一个不包含 Dropout 的 MLP，使用 LayerNorm 代替 BatchNorm
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden),   # index 0
#             nn.LayerNorm(hidden),           # index 1
#             nn.ReLU(),                      # index 2

#             nn.Linear(hidden, hidden),      # index 3
#             nn.LayerNorm(hidden),           # index 4
#             nn.ReLU(),                      # index 5

#             nn.Linear(hidden, hidden),      # index 6
#             nn.LayerNorm(hidden),           # index 7
#             nn.ReLU(),                      # index 8

#             nn.Linear(hidden, hidden),      # index 9
#             nn.LayerNorm(hidden),           # index 10
#             nn.ReLU(),                      # index 11

#             nn.Linear(hidden, DOS_POINTS)   # index 12
#         )

#         self.conv = nn.Conv1d(1, 3, kernel_size=3, padding=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         参数:
#           x: torch.Tensor, 形状 (B*P, input_dim)
#              其中 B 是 batch size, P 是 padding_size, input_dim = 701

#         返回:
#           torch.Tensor, 形状 (B*P, DOS_POINTS)，即每个原子在 341 个能量点上的 DOS 预测
#         """
#         # 1) 先做第一层 Linear
#         h0 = self.mlp[0](x)              # (B*P, hidden)
#         # 2) 再做第一层 LayerNorm
#         h1 = self.mlp[1](h0)             # (B*P, hidden)

#         # 调试打印
#         print("Linear 输出 mean/std =", h0.mean().item(), h0.std().item())
#         print("LayerNorm 输出 mean/std =", h1.mean().item(), h1.std().item())

#         # 3) 然后把结果依次传给剩余层
#         h = h1
#         for layer in list(self.mlp.children())[2:]:  # index 从 2 开始到末尾
#             h = layer(h)  # 依次经过 ReLU, Linear, LayerNorm, ReLU, ... 最终到 Linear(hidden→341)

#         # 4) 将 (B*P, 341) reshape 为 (B*P, 1, 341)，再做 Conv1d 并对通道求平均
#         h = h.view(-1, 1, DOS_POINTS)   # (B*P, 1, 341)
#         h = self.conv(h).mean(dim=1)     # → (B*P, 341)

#         return h
        
class _DOSSubNetH(_DOSSubNetCNO):
    pass  # identical architecture; only input_dim differs when instantiated


# ---------------------------------------------------------------------------
#  Combined DOS model
# ---------------------------------------------------------------------------
class DOSModel(nn.Module):
    def __init__(self, padding_size: int):
        super().__init__()
        self.padding_size = padding_size

        # 每个子网的输入维度 = 360（fingerprint） + DOS_POINTS（mask）
        in_dim_C = 360 + DOS_POINTS  # C: 701
        in_dim_H = 360 + DOS_POINTS  # H: 701
        in_dim_N = 360 + DOS_POINTS  # N: 701
        in_dim_O = 360 + DOS_POINTS  # O: 701

        self.netC = _DOSSubNetCNO(in_dim_C)
        self.netH = _DOSSubNetH(in_dim_H)
        self.netN = _DOSSubNetCNO(in_dim_N)
        self.netO = _DOSSubNetCNO(in_dim_O)

        # 全局 head 用于预测 [-VB, -CB]
        self.band_head = nn.Sequential(
            nn.Linear(DOS_POINTS, 100), nn.ReLU(),
            nn.Linear(100, 100),       nn.ReLU(),
            nn.Linear(100, 2)
        )

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
        """
        X_*       : (B, P, 360)         （360 维原子指纹）
        *_m       : (B, P, DOS_POINTS)  （341 维掩码）
        total_elec: (B, 1)
        返回     norm_dos (B, 341), vbcb (B, 2)
        """
        B, P, _ = X_C.shape

        def _flatten(t: torch.Tensor) -> torch.Tensor:
            # 把 (B, P, feat) → (B*P, feat)
            return t.view(B * P, -1)

        # 先把 fingerprint 和 mask 在最后一维拼接 → (B, P, 701)
        catC = torch.cat([X_C, C_m], dim=-1)  # (B, P, 701)
        catH = torch.cat([X_H, H_m], dim=-1)
        catN = torch.cat([X_N, N_m], dim=-1)
        catO = torch.cat([X_O, O_m], dim=-1)

        # 再 flatten, 送入子网
        flatC = self.netC(_flatten(catC)).view(B, P, DOS_POINTS)  # (B, P, 341)
        flatH = self.netH(_flatten(catH)).view(B, P, DOS_POINTS)
        flatN = self.netN(_flatten(catN)).view(B, P, DOS_POINTS)
        flatO = self.netO(_flatten(catO)).view(B, P, DOS_POINTS)

        # 对每个原子在每个能量点的预测乘以对应掩码，再 sum over atoms
        total_dos = (
            (flatC * C_m).sum(dim=1) +
            (flatH * H_m).sum(dim=1) +
            (flatN * N_m).sum(dim=1) +
            (flatO * O_m).sum(dim=1)
        )  # (B, 341)

        # 归一化，预测带边
        norm_dos = total_dos / total_elec  # (B, 341) / (B, 1)
        vbcb = self.band_head(norm_dos)    # (B, 2)

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

# def _prepare_single(folder: str, padding_size: int, args) -> Tuple[np.ndarray, ...]:
#     """Prepare one structure for DOS training / inference."""

#     # -------- read structure & element counts ----------------------------
#     struct = Poscar.from_file(Path(folder) / "POSCAR").structure
#     elem_dict = {"C": 0, "H": 0, "N": 0, "O": 0}
#     for s in struct:
#         sym = s.specie.symbol
#         if sym in elem_dict:
#             elem_dict[sym] += 1
#     at_elem = [elem_dict[e] for e in ("C", "H", "N", "O")]

#     # -------- raw fingerprints (4 × 3-D) ---------------------------------
#     from .fp import fp_atom  # local import to avoid circular deps

#     dset, basis_mat, *_ = fp_atom(
#         struct,
#         args.grid_spacing,
#         args.cut_off_rad,
#         args.widest_gaussian,
#         args.narrowest_gaussian,
#         args.num_gamma,
#     )

#     (
#         X_3D1, X_3D2, X_3D3, X_3D4,  # fingerprints
#         _, _, _, _,                  # unused grads
#         C_m_1, H_m_1, N_m_1, O_m_1,  # masks
#     ) = chg_data(dset, basis_mat, *at_elem, padding_size)

#     # -------- charge-derived coefficients --------------------------------
#     coef_paths = [Path(folder) / f"Coef_{e}.npy" for e in ("C", "H", "N", "O")]
#     if not all(p.exists() for p in coef_paths):
#         raise FileNotFoundError("Missing Coef_*.npy — run charge→coef export first.")
#     Coef_C, Coef_H, Coef_N, Coef_O = [np.load(p).astype(np.float32) for p in coef_paths]

#     # -------- concatenate & normalise ------------------------------------
#     X_C_aug, X_H_aug, X_N_aug, X_O_aug = fp_chg_norm(
#         X_3D1, X_3D2, X_3D3, X_3D4,        # 4 × (P, feat_fp)
#         Coef_C, Coef_H, Coef_N, Coef_O,    # 4 × (P, coeff_dim) – same 2-D dims
#         padding_size,
#         SCALER_PATHS,
#     )

#     # -------- masks (DOS-specific) ---------------------------------------
#     C_d, H_d, N_d, O_d = dos_mask(C_m_1, H_m_1, N_m_1, O_m_1, padding_size)

#     # -------- ground-truth DOS & band edges ------------------------------
#     elec_per_atom = {6: 4, 1: 1, 7: 5, 8: 6}
#     total_elec = np.array([elec_per_atom[z] for z in struct.atomic_numbers], dtype=np.float32).sum()
#     Prop_dos, VB, CB = _read_dos(folder, total_elec)
#     VB_CB = np.array([-VB, -CB], dtype=np.float32)

#     # -------- return (each item shape matches fp_chg_norm expectations) ---
#     return (
#         X_C_aug, X_H_aug, X_N_aug, X_O_aug,              # 4 × (P, feat_all)
#         np.array([total_elec], dtype=np.float32),        # (1,)
#         C_d, H_d, N_d, O_d,                             # 4 × (P, DOS_POINTS)
#         Prop_dos, VB_CB,
# )

# def _prepare_single(folder: str, padding_size: int, args) -> Tuple[np.ndarray, ...]:
#     """
#     Prepare one structure for DOS training/inference, WITHOUT any normalization.

#     Returns:
#       X_C_aug, X_H_aug, X_N_aug, X_O_aug : 四个 np.ndarray，形状均为 (padding_size, 360)
#       total_elec                         : np.ndarray，形状 (1,)
#       C_d, H_d, N_d, O_d                 : 四个 np.ndarray，形状均为 (padding_size, DOS_POINTS)
#       Prop_dos                           : np.ndarray，形状 (DOS_POINTS,)
#       VB_CB                              : np.ndarray，形状 (2,)
#     """

#     # -------- read structure & element counts ----------------------------
#     struct = Poscar.from_file(Path(folder) / "POSCAR").structure
#     elem_dict = {"C": 0, "H": 0, "N": 0, "O": 0}
#     for s in struct:
#         sym = s.specie.symbol
#         if sym in elem_dict:
#             elem_dict[sym] += 1
#     at_elem = [elem_dict[e] for e in ("C", "H", "N", "O")]

#     # -------- raw fingerprints (4 × (padding_size, feat_fp_raw)) -----------
#     from .fp import fp_atom  # 避免循环依赖
#     from .data_io import chg_data, dos_mask

#     dset, basis_mat, *_ = fp_atom(
#         struct,
#         args.grid_spacing,
#         args.cut_off_rad,
#         args.widest_gaussian,
#         args.narrowest_gaussian,
#         args.num_gamma,
#     )

#     # chg_data 返回：
#     #   X_3D1..X_3D4  → 可能形状 (1, padding_size, feat_raw) 或 (padding_size, feat_raw)
#     #   _, _, _, _    → 占位（grad 信息，不使用）
#     #   C_m_1..O_m_1  → 可能形状 (1, padding_size, DOS_POINTS) 或 (padding_size, DOS_POINTS)
#     X_3D1, X_3D2, X_3D3, X_3D4, _, _, _, _, C_m_1, H_m_1, N_m_1, O_m_1 = chg_data(
#         dset, basis_mat, *at_elem, padding_size
#     )

#     # -------- 把 X_3D* squeeze 到 (padding_size, feat_raw) ----------------
#     def _squeeze_if_needed(arr: np.ndarray) -> np.ndarray:
#         """若 arr.shape = (1, P, D) 就 squeeze 为 (P, D)，否则假定已是 (P, D)。"""
#         if arr.ndim == 3 and arr.shape[0] == 1:
#             return arr[0]
#         return arr

#     X_3D1 = _squeeze_if_needed(X_3D1)
#     X_3D2 = _squeeze_if_needed(X_3D2)
#     X_3D3 = _squeeze_if_needed(X_3D3)
#     X_3D4 = _squeeze_if_needed(X_3D4)

#     # -------- masks 输入 dos_mask 前也要 squeeze --------------------------
#     C_m_1 = _squeeze_if_needed(C_m_1)
#     H_m_1 = _squeeze_if_needed(H_m_1)
#     N_m_1 = _squeeze_if_needed(N_m_1)
#     O_m_1 = _squeeze_if_needed(O_m_1)

#     # -------- 工具：pad_or_trunc，使 (P, D_raw) → (P, 360) -------------
#     def _pad_or_trunc_feat(arr: np.ndarray, target_dim: int = 360) -> np.ndarray:
#         """
#         输入 arr.shape = (P, D)
#         - 若 D ≥ target_dim，则 arr[:, :target_dim]
#         - 若 D <  target_dim，则在最后维度补零到 (P, target_dim)
#         输出 (P, target_dim)
#         """
#         P, D = arr.shape
#         assert P == padding_size, f"Expected {padding_size} rows, got {P}"
#         if D == target_dim:
#             return arr.copy()
#         if D > target_dim:
#             return arr[:, :target_dim]
#         # D < target_dim，补零
#         pad_width = target_dim - D
#         pad_block = np.zeros((P, pad_width), dtype=arr.dtype)
#         return np.concatenate([arr, pad_block], axis=-1)

#     # -------- 对每个元素的 raw fingerprint 做 pad_or_trunc --------
#     Xc_pad = _pad_or_trunc_feat(X_3D1, 360)  # (padding_size, 360)
#     Xh_pad = _pad_or_trunc_feat(X_3D2, 360)
#     Xn_pad = _pad_or_trunc_feat(X_3D3, 360)
#     Xo_pad = _pad_or_trunc_feat(X_3D4, 360)

#     # 此时 X_*_pad 已经是 (padding_size, 360)，不再添加 batch 维度
#     X_C_aug = Xc_pad
#     X_H_aug = Xh_pad
#     X_N_aug = Xn_pad
#     X_O_aug = Xo_pad

#     # -------- masks (DOS-specific) ---------------------------------------
#     C_d, H_d, N_d, O_d = dos_mask(C_m_1, H_m_1, N_m_1, O_m_1, padding_size)
#     # dos_mask 返回可能是 (1, P, 341)，需要再 squeeze 一次
#     C_d = _squeeze_if_needed(C_d)  # 现在应为 (P, 341)
#     H_d = _squeeze_if_needed(H_d)
#     N_d = _squeeze_if_needed(N_d)
#     O_d = _squeeze_if_needed(O_d)

#     # 再次确认掩码形状为 (padding_size, DOS_POINTS)
#     assert C_d.shape == (padding_size, DOS_POINTS), f"C_d.shape={C_d.shape}"
#     assert H_d.shape == (padding_size, DOS_POINTS), f"H_d.shape={H_d.shape}"
#     assert N_d.shape == (padding_size, DOS_POINTS), f"N_d.shape={N_d.shape}"
#     assert O_d.shape == (padding_size, DOS_POINTS), f"O_d.shape={O_d.shape}"

#     # -------- ground-truth DOS & band edges ------------------------------
#     elec_per_atom = {6: 4, 1: 1, 7: 5, 8: 6}
#     total_elec = np.array(
#         [elec_per_atom[z] for z in struct.atomic_numbers], dtype=np.float32
#     ).sum()
#     Prop_dos, VB, CB = _read_dos(folder, total_elec)

#     # 确保 Prop_dos 形状为 (DOS_POINTS,)
#     assert Prop_dos.shape == (DOS_POINTS,), f"Prop_dos.shape={Prop_dos.shape}"
#     VB_CB = np.array([-VB, -CB], dtype=np.float32)

#     # -------- 返回所有内容 -----------------------------------------------
#     return (
#         X_C_aug, X_H_aug, X_N_aug, X_O_aug,          # 四个 np.ndarray, 形状 (padding_size, 360)
#         np.array([total_elec], dtype=np.float32),    # (1,)
#         C_d, H_d, N_d, O_d,                          # 四个 np.ndarray, 形状 (padding_size, 341)
#         Prop_dos,                                    # (341,)
#         VB_CB                                        # (2,)
#     )


def _prepare_single(folder: str, padding_size: int, args) -> Tuple[np.ndarray, ...]:
    """
    Prepare one structure for DOS training/inference, using fp_norm for MaxAbsScaler normalization.

    Returns:
      X_C_aug, X_H_aug, X_N_aug, X_O_aug : 四个 np.ndarray，形状均为 (padding_size, 360)
      total_elec                         : np.ndarray，形状 (1,)
      C_d, H_d, N_d, O_d                 : 四个 np.ndarray，形状均为 (padding_size, DOS_POINTS)
      Prop_dos                           : np.ndarray，形状 (DOS_POINTS,)
      VB_CB                              : np.ndarray，形状 (2,)
    """
    # -------- read structure & element counts ----------------------------
    struct = Poscar.from_file(Path(folder) / "POSCAR").structure
    elem_dict = {"C": 0, "H": 0, "N": 0, "O": 0}
    for s in struct:
        sym = s.specie.symbol
        if sym in elem_dict:
            elem_dict[sym] += 1
    at_elem = [elem_dict[e] for e in ("C", "H", "N", "O")]

    # -------- raw fingerprints (4 × (padding_size, feat_fp_raw)) -----------
    from .fp import fp_atom, fp_norm
    from .data_io import chg_data, dos_mask

    dset, basis_mat, *_ = fp_atom(
        struct,
        args.grid_spacing,
        args.cut_off_rad,
        args.widest_gaussian,
        args.narrowest_gaussian,
        args.num_gamma,
    )

    # chg_data 返回四个 X_3D*，可能是 (1, P, feat_raw) 或 (P, feat_raw)
    X_3D1, X_3D2, X_3D3, X_3D4, _, _, _, _, C_m_1, H_m_1, N_m_1, O_m_1 = chg_data(
        dset, basis_mat, *at_elem, padding_size
    )

    # -------- squeeze 到 (P, feat_raw) ---------------------------------------
    def _squeeze_if_needed(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
        return arr

    X_3D1 = _squeeze_if_needed(X_3D1)  # 形状变为 (P, feat_raw)
    X_3D2 = _squeeze_if_needed(X_3D2)
    X_3D3 = _squeeze_if_needed(X_3D3)
    X_3D4 = _squeeze_if_needed(X_3D4)

    C_m_1 = _squeeze_if_needed(C_m_1)  # (P, 341)
    H_m_1 = _squeeze_if_needed(H_m_1)
    N_m_1 = _squeeze_if_needed(N_m_1)
    O_m_1 = _squeeze_if_needed(O_m_1)

    # -------- pad_or_trunc: 把 (P, feat_raw) → (P, 360)  ----------------------
    def _pad_or_trunc_feat(arr: np.ndarray, target_dim: int = 360) -> np.ndarray:
        P, D = arr.shape
        assert P == padding_size, f"Expected {padding_size} rows, got {P}"
        if D == target_dim:
            return arr.copy()
        if D > target_dim:
            return arr[:, :target_dim]
        # D < target_dim，需要右侧补零
        pad_width = target_dim - D
        pad_block = np.zeros((P, pad_width), dtype=arr.dtype)
        return np.concatenate([arr, pad_block], axis=-1)

    Xc_pt = _pad_or_trunc_feat(X_3D1, 360)  # (P, 360)
    Xh_pt = _pad_or_trunc_feat(X_3D2, 360)
    Xn_pt = _pad_or_trunc_feat(X_3D3, 360)
    Xo_pt = _pad_or_trunc_feat(X_3D4, 360)

    # -------- 归一化：先把 (P,360) → (1, P,360)，再送给 fp_norm  -------------------
    X_C_in = Xc_pt[np.newaxis, ...]  # 变为 (1, P, 360)
    X_H_in = Xh_pt[np.newaxis, ...]
    X_N_in = Xn_pt[np.newaxis, ...]
    X_O_in = Xo_pt[np.newaxis, ...]

    # fp_norm 返回 (1, P, 360) 类型的归一化结果
    X_C_norm, X_H_norm, X_N_norm, X_O_norm = fp_norm(
        X_C_in, X_H_in, X_N_in, X_O_in,
        padding_size,
        SCALER_PATHS
    )

    # squeeze 回 (P, 360)
    X_C_aug = X_C_norm[0]
    X_H_aug = X_H_norm[0]
    X_N_aug = X_N_norm[0]
    X_O_aug = X_O_norm[0]

    # -------- masks (DOS-specific) ---------------------------------------
    C_d, H_d, N_d, O_d = dos_mask(C_m_1, H_m_1, N_m_1, O_m_1, padding_size)
    C_d = _squeeze_if_needed(C_d)  # (P, 341)
    H_d = _squeeze_if_needed(H_d)
    N_d = _squeeze_if_needed(N_d)
    O_d = _squeeze_if_needed(O_d)

    assert C_d.shape == (padding_size, DOS_POINTS)
    assert H_d.shape == (padding_size, DOS_POINTS)
    assert N_d.shape == (padding_size, DOS_POINTS)
    assert O_d.shape == (padding_size, DOS_POINTS)

    # -------- ground-truth DOS & band edges ------------------------------
    elec_per_atom = {6: 4, 1: 1, 7: 5, 8: 6}
    total_elec = np.array(
        [elec_per_atom[z] for z in struct.atomic_numbers], dtype=np.float32
    ).sum()
    Prop_dos, VB, CB = _read_dos(folder, total_elec)
    assert Prop_dos.shape == (DOS_POINTS,)
    VB_CB = np.array([-VB, -CB], dtype=np.float32)

    # -------- 返回所有内容 -----------------------------------------------
    return (
        X_C_aug, X_H_aug, X_N_aug, X_O_aug,   # 四个 (padding_size, 360)
        np.array([total_elec], dtype=np.float32),  # (1,)
        C_d, H_d, N_d, O_d,                   # 四个 (padding_size, 341)
        Prop_dos,                             # (341,)
        VB_CB                                 # (2,)
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
