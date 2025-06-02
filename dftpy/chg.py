# dftpy/chg.py
# -----------------------------------------------------------------------------#
#  Charge-network  (C/H/N/O)  –  PyTorch 实现                                    #
# -----------------------------------------------------------------------------#
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pymatgen.io.vasp.outputs import Poscar, Chgcar
from torch.utils.data import DataLoader, Dataset

from .data_io import chg_data
from .fp import fp_atom

# -----------------------------------------------------------------------------#
#  DEVICE                                                                      #
# -----------------------------------------------------------------------------#
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------#
#  单原子子网                                                                   #
# -----------------------------------------------------------------------------#
class _SingleAtomNetCNO(nn.Module):
    """360 → 340  (93 exp + 247 coef)"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(360, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 340),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        exp_part = torch.abs(h[:, :93])   # 保证指数为正
        coef_part = h[:, 93:]
        return torch.cat([exp_part, coef_part], dim=1)


class _SingleAtomNetH(nn.Module):
    """360 → 208  (58 exp + 150 coef)"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(360, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 208),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        exp_part = torch.abs(h[:, :58])
        coef_part = h[:, 58:]
        return torch.cat([exp_part, coef_part], dim=1)


# -----------------------------------------------------------------------------#
#  四分支 ChargeModel                                                           #
# -----------------------------------------------------------------------------#
class ChargeModel(nn.Module):
    """
    输入 4 个张量 (batch, P, 360) 按 C/H/N/O 分别处理。
    输出 4 个系数张量：
        C / N / O : (batch, P, 340)    H : (batch, P, 208)
    """

    def __init__(self, padding_size: int) -> None:
        super().__init__()
        self.padding_size = padding_size
        self.netC = _SingleAtomNetCNO().to(DEVICE)
        self.netH = _SingleAtomNetH().to(DEVICE)
        self.netN = _SingleAtomNetCNO().to(DEVICE)
        self.netO = _SingleAtomNetCNO().to(DEVICE)

    # ---- forward ------------------------------------------------------------#
    def forward(
        self,
        X_C: torch.Tensor,
        X_H: torch.Tensor,
        X_N: torch.Tensor,
        X_O: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, p, _ = X_C.shape
        C_out = self.netC(X_C.reshape(b * p, -1)).view(b, p, -1)
        H_out = self.netH(X_H.reshape(b * p, -1)).view(b, p, -1)
        N_out = self.netN(X_N.reshape(b * p, -1)).view(b, p, -1)
        O_out = self.netO(X_O.reshape(b * p, -1)).view(b, p, -1)
        return C_out, H_out, N_out, O_out


# -----------------------------------------------------------------------------#
#  工具函数                                                                    #
# -----------------------------------------------------------------------------#
def init_chgmod(padding_size: int) -> ChargeModel:
    """建模但不加载权重"""
    return ChargeModel(padding_size).to(DEVICE)


def load_pretrained_chg_model(
    checkpoint_path: str, padding_size: int
) -> ChargeModel:
    """加载现成权重；若不存在直接抛错"""
    model = init_chgmod(padding_size)
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"ChargeModel 权重不存在: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


# -----------------------------------------------------------------------------#
#  Dataset                                                                    #
# -----------------------------------------------------------------------------#
class ChargeDataset(Dataset):
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
        self.C_C = torch.from_numpy(Coef_C).float()
        self.C_H = torch.from_numpy(Coef_H).float()
        self.C_N = torch.from_numpy(Coef_N).float()
        self.C_O = torch.from_numpy(Coef_O).float()

    def __len__(self) -> int:
        return self.X_C.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.X_C[idx],
            self.X_H[idx],
            self.X_N[idx],
            self.X_O[idx],
            self.C_C[idx],
            self.C_H[idx],
            self.C_N[idx],
            self.C_O[idx],
        )


# -----------------------------------------------------------------------------#
#  训练函数 (移除 CHGCAR 依赖)                                                  #
# -----------------------------------------------------------------------------#
_MSE = nn.MSELoss()

def _pad_coef(arr: np.ndarray, target_rows: int, dim_out: int) -> np.ndarray:
    pad = np.zeros((target_rows, dim_out), dtype=np.float32)
    if arr.size:
        pad[: min(len(arr), target_rows)] = arr[:target_rows]
    return pad


def _prepare_coef_data(
    folders: List[str], padding_size: int, args
) -> Tuple[np.ndarray, ...]:
    X_C, X_H, X_N, X_O = [], [], [], []
    Co_C, Co_H, Co_N, Co_O = [], [], [], []

    for fld in folders:
        # 指纹
        struct = Poscar.from_file(Path(fld) / "POSCAR").structure
        dset, basis, _, _, at_elem = fp_atom(
            struct,
            args.grid_spacing,
            args.cut_off_rad,
            args.widest_gaussian,
            args.narrowest_gaussian,
            args.num_gamma,
        )
        X1, X2, X3, X4, *_ = chg_data(
            dset, basis, *at_elem, padding_size
        )  # 每个 (1, P, 360)
        X_C.append(X1)
        X_H.append(X2)
        X_N.append(X3)
        X_O.append(X4)

        # 真实系数（假设用户自行准备 *.npy）
        load = lambda n: np.load(Path(fld) / n) if (Path(fld) / n).is_file() else np.empty((0, 1))
        Co_C.append(_pad_coef(load("Coef_C.npy"), padding_size, 340))
        Co_H.append(_pad_coef(load("Coef_H.npy"), padding_size, 208))
        Co_N.append(_pad_coef(load("Coef_N.npy"), padding_size, 340))
        Co_O.append(_pad_coef(load("Coef_O.npy"), padding_size, 340))

    stack = lambda lst: np.vstack(lst).astype(np.float32)
    return (
        stack(X_C),
        stack(X_H),
        stack(X_N),
        stack(X_O),
        np.stack(Co_C),
        np.stack(Co_H),
        np.stack(Co_N),
        np.stack(Co_O),
    )


def train_chg_model(
    train_folders: List[str],
    val_folders: List[str],
    padding_size: int,
    args,
) -> None:
    """完整训练流程（不再依赖 CHGCAR）"""
    # ---------- 数据 ---------- #
    data_tr = _prepare_coef_data(train_folders, padding_size, args)
    data_va = _prepare_coef_data(val_folders, padding_size, args)

    train_loader = DataLoader(
        ChargeDataset(*data_tr),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ChargeDataset(*data_va),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ---------- 模型 ---------- #
    model = init_chgmod(padding_size)
    optim_ = optim.Adam(model.parameters(), lr=args.learning_rate)

    best, best_epoch = 1e9, 0
    for epoch in range(1, args.epochs + 1):
        # ---- train ---- #
        model.train()
        tr_loss = 0.0
        for Xc, Xh, Xn, Xo, Cc, Ch, Cn, Co in train_loader:
            Xc, Xh, Xn, Xo = (t.to(DEVICE) for t in (Xc, Xh, Xn, Xo))
            Cc, Ch, Cn, Co = (t.to(DEVICE) for t in (Cc, Ch, Cn, Co))
            optim_.zero_grad()
            Pc, Ph, Pn, Po = model(Xc, Xh, Xn, Xo)
            loss = (
                _MSE(Pc, Cc)
                + _MSE(Ph, Ch)
                + _MSE(Pn, Cn)
                + _MSE(Po, Co)
            )
            loss.backward()
            optim_.step()
            tr_loss += loss.item()

        # ---- val ---- #
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for Xc, Xh, Xn, Xo, Cc, Ch, Cn, Co in val_loader:
                Xc, Xh, Xn, Xo = (t.to(DEVICE) for t in (Xc, Xh, Xn, Xo))
                Cc, Ch, Cn, Co = (t.to(DEVICE) for t in (Cc, Ch, Cn, Co))
                Pc, Ph, Pn, Po = model(Xc, Xh, Xn, Xo)
                va_loss += (
                    _MSE(Pc, Cc)
                    + _MSE(Ph, Ch)
                    + _MSE(Pn, Cn)
                    + _MSE(Po, Co)
                ).item()

        tr_loss /= len(train_loader)
        va_loss /= len(val_loader)
        print(f"Epoch {epoch:03d} | Train {tr_loss:.4e} | Val {va_loss:.4e}")

        if va_loss < best:
            best, best_epoch = va_loss, epoch
            torch.save(model.state_dict(), "best_chg.pth")
        if epoch - best_epoch >= args.patience:
            print("Early-stop triggered.")
            break

    print(f"[DONE] best Val Loss = {best:.4e} @ epoch {best_epoch}")


# -----------------------------------------------------------------------------#
#  推理：infer_charges                                                         #
# -----------------------------------------------------------------------------#
def _calc_charge(exp: np.ndarray, coef: np.ndarray) -> float:
    """简化的 Slater-type 积分 s-orbital 近似"""
    return float(np.sum(np.pi**1.5 * coef / np.power(exp, 1.5)))


def infer_charges(
    folder: str,
    chg_model: ChargeModel,
    padding_size: int,
    args,
) -> np.ndarray:
    """
    对单个结构预测原子净电荷。
    如果 args.write_chg 且 CHGCAR 存在，会将预测值写入文件。
    """
    poscar = Poscar.from_file(Path(folder) / "POSCAR")
    struct = poscar.structure

    dset, basis, *_ = fp_atom(
        struct,
        args.grid_spacing,
        args.cut_off_rad,
        args.widest_gaussian,
        args.narrowest_gaussian,
        args.num_gamma,
    )
    at_elem = [struct.species.count(el) for el in ("C", "H", "N", "O")]
    X1, X2, X3, X4, *_ = chg_data(dset, basis, *at_elem, padding_size)

    # → torch
    tens = lambda x: torch.from_numpy(x).float().to(DEVICE)
    C_pred, H_pred, N_pred, O_pred = chg_model(
        tens(X1), tens(X2), tens(X3), tens(X4)
    )
    C_np = C_pred.cpu().numpy().squeeze(0)
    H_np = H_pred.cpu().numpy().squeeze(0)
    N_np = N_pred.cpu().numpy().squeeze(0)
    O_np = O_pred.cpu().numpy().squeeze(0)

    charges: List[float] = []
    idx = 0
    for n, arr, dim in zip(at_elem, (C_np, H_np, N_np, O_np), (340, 208, 340, 340)):
        if n == 0:
            continue
        exp_dim = 93 if dim == 340 else 58
        for i in range(n):
            q = _calc_charge(arr[i, :exp_dim], arr[i, exp_dim:])
            charges.append(q)
            idx += 1

    charges_np = np.asarray(charges, dtype=np.float32)

    # --- 可选写文件 --- #
    if getattr(args, "write_chg", False):
        chg_path = Path(folder).name + "_Pred_CHG.dat"
        try:
            Chgcar.from_file(Path(folder) / "CHGCAR")  # 仅检测存在
            np.savetxt(chg_path, charges_np, fmt="%.6f")
            print(f"[write] {chg_path}")
        except FileNotFoundError:
            pass  # 没有 CHGCAR 就安静跳过

    return charges_np


# -----------------------------------------------------------------------------#
#  模块导出                                                                    #
# -----------------------------------------------------------------------------#
__all__ = [
    "init_chgmod",
    "load_pretrained_chg_model",
    "train_chg_model",
    "infer_charges",
]
