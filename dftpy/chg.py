# -----------------------------------------------------------------------------#
#  dftpy/chg.py  ——  统一指纹到 360 维，解决 vstack 维度不一致                   #
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
from torch.utils.data import Dataset, DataLoader

from .fp import fp_atom
from .data_io import chg_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- 单原子子网络 --------------------------------------#
class _NetCNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(360, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, 340),
        )

    def forward(self, x):
        h = self.net(x)
        exp, coef = torch.abs(h[:, :93]), h[:, 93:]
        return torch.cat([exp, coef], 1)


class _NetH(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(360, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, 208),
        )

    def forward(self, x):
        h = self.net(x)
        exp, coef = torch.abs(h[:, :58]), h[:, 58:]
        return torch.cat([exp, coef], 1)

# ------------------------- 四分支模型 ----------------------------------------#
class ChargeModel(nn.Module):
    def __init__(self, padding_size: int):
        super().__init__()
        self.padding_size = padding_size
        self.netC, self.netH = _NetCNO(), _NetH()
        self.netN, self.netO = _NetCNO(), _NetCNO()

    def forward(self, X_C, X_H, X_N, X_O):
        b, p, _ = X_C.shape
        f = lambda net, x: net(x.reshape(b * p, -1)).view(b, p, -1)
        return (
            f(self.netC, X_C),
            f(self.netH, X_H),
            f(self.netN, X_N),
            f(self.netO, X_O),
        )

def init_chgmod(padding_size: int):
    return ChargeModel(padding_size).to(DEVICE)

def load_pretrained_chg_model(path: str, padding_size: int):
    if not Path(path).is_file():
        raise FileNotFoundError(f"ChargeModel 权重不存在: {path}")
    m = init_chgmod(padding_size)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m

# ------------------------- Dataset ------------------------------------------#
class ChargeDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = [torch.from_numpy(a).float() for a in arrays]

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i):
        return tuple(a[i] for a in self.arrays)

# ------------------------- 工具：把特征调成 360 维 ---------------------------#
def _fix_feat(feat: np.ndarray, target: int = 360) -> np.ndarray:
    if feat.shape[-1] == target:
        return feat
    if feat.shape[-1] > target:                       # 截断
        return feat[..., :target]
    pad = np.zeros((*feat.shape[:-1], target - feat.shape[-1]), feat.dtype)
    return np.concatenate([feat, pad], axis=-1)       # 右侧 0-padding

# ------------------------- 数据准备 ------------------------------------------#
def _prepare(
    folders: List[str], padding: int, args
) -> Tuple[np.ndarray, ...]:
    XCs, XHs, XNs, XOs, CCs, CHs, CNs, COs = [], [], [], [], [], [], [], []

    def _pad_coef(a: np.ndarray, dim: int) -> np.ndarray:
        pad = np.zeros((padding, dim), np.float32)
        if a.size:
            pad[: min(len(a), padding)] = a[:padding]
        return pad

    for fld in folders:
        struct = Poscar.from_file(Path(fld) / "POSCAR").structure
        dset, basis, _, _, ats = fp_atom(
            struct,
            args.grid_spacing, args.cut_off_rad,
            args.widest_gaussian, args.narrowest_gaussian, args.num_gamma,
        )
        X1, X2, X3, X4, *_ = chg_data(dset, basis, *ats, padding)
        XCs.append(_fix_feat(X1))
        XHs.append(_fix_feat(X2))
        XNs.append(_fix_feat(X3))
        XOs.append(_fix_feat(X4))

        load = lambda n: np.load(Path(fld) / n) if (Path(fld) / n).is_file() else np.empty((0, 1))
        CCs.append(_pad_coef(load("Coef_C.npy"), 340))
        CHs.append(_pad_coef(load("Coef_H.npy"), 208))
        CNs.append(_pad_coef(load("Coef_N.npy"), 340))
        COs.append(_pad_coef(load("Coef_O.npy"), 340))

    stack = lambda lst: np.vstack(lst).astype(np.float32)
    return (
        stack(XCs), stack(XHs), stack(XNs), stack(XOs),
        np.stack(CCs), np.stack(CHs), np.stack(CNs), np.stack(COs),
    )

# ------------------------- 训练 ---------------------------------------------#
_MSE = nn.MSELoss()

def train_chg_model(train_folders, val_folders, padding_size, args):
    tr = DataLoader(
        ChargeDataset(*_prepare(train_folders, padding_size, args)),
        batch_size=args.batch_size, shuffle=True)
    va = DataLoader(
        ChargeDataset(*_prepare(val_folders, padding_size, args)),
        batch_size=args.batch_size, shuffle=False)

    model = init_chgmod(padding_size)
    opt = optim.Adam(model.parameters(), lr=args.learning_rate)

    best, best_ep = 1e9, 0
    for ep in range(1, args.epochs + 1):
        # ----- train -----
        model.train()
        tl = 0.0
        for Xc, Xh, Xn, Xo, Cc, Ch, Cn, Co in tr:
            Xc, Xh, Xn, Xo = (t.to(DEVICE) for t in (Xc, Xh, Xn, Xo))
            Cc, Ch, Cn, Co = (t.to(DEVICE) for t in (Cc, Ch, Cn, Co))
            opt.zero_grad()
            Pc, Ph, Pn, Po = model(Xc, Xh, Xn, Xo)
            loss = _MSE(Pc, Cc) + _MSE(Ph, Ch) + _MSE(Pn, Cn) + _MSE(Po, Co)
            loss.backward()
            opt.step()
            tl += loss.item()
        tl /= len(tr)

        # ----- val -----
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for Xc, Xh, Xn, Xo, Cc, Ch, Cn, Co in va:
                Xc, Xh, Xn, Xo = (t.to(DEVICE) for t in (Xc, Xh, Xn, Xo))
                Cc, Ch, Cn, Co = (t.to(DEVICE) for t in (Cc, Ch, Cn, Co))
                Pc, Ph, Pn, Po = model(Xc, Xh, Xn, Xo)
                vl += (_MSE(Pc, Cc) + _MSE(Ph, Ch) + _MSE(Pn, Cn) + _MSE(Po, Co)).item()
        vl /= len(va)
        print(f"Epoch {ep:03d} | Train {tl:.4e} | Val {vl:.4e}")

        if vl < best:
            best, best_ep = vl, ep
            torch.save(model.state_dict(), "best_chg.pth")
        if ep - best_ep >= args.patience:
            print("Early-stop.")
            break

    print(f"[DONE] best Val Loss = {best:.4e} @ epoch {best_ep}")

# ------------------------- 推理 ---------------------------------------------#
def _charge(exp, coef):
    return float(np.sum(np.pi**1.5 * coef / exp**1.5))

def infer_charges(folder: str, chg_model: ChargeModel, padding_size: int, args):
    """
    folder: 既可以是“包含 POSCAR 文件的目录”，也可以直接是 POSCAR 文件本身
    """
    # ----- 1) 先定位 POSCAR 文件的真实路径 -----
    p = Path(folder)
    if p.is_file() and p.name.upper() == "POSCAR":
        # 如果 folder 本身就是一个“POSCAR 文件路径”
        poscar_path = p
    else:
        # 否则 folder 应当是包含 POSCAR 的目录
        poscar_path = p / "POSCAR"
    if not poscar_path.is_file():
        raise FileNotFoundError(f"无法找到 POSCAR: {poscar_path}")

    # ----- 2) 从 POSCAR 里读结构 -----
    struct = Poscar.from_file(str(poscar_path)).structure

    # ----- 3) 计算指纹 & pad -----
    dset, basis, *_ = fp_atom(
        struct,
        args.grid_spacing, args.cut_off_rad,
        args.widest_gaussian, args.narrowest_gaussian, args.num_gamma,
    )
    ats = [struct.species.count(e) for e in ("C", "H", "N", "O")]
    X1, X2, X3, X4, *_ = chg_data(dset, basis, *ats, padding_size)

    # ----- 4) 转为 Tensor 并送进模型 -----
    tens = lambda x: torch.from_numpy(_fix_feat(x)).float().to(DEVICE)
    Xc_t, Xh_t, Xn_t, Xo_t = tens(X1), tens(X2), tens(X3), tens(X4)

    # ----- 5) 推理（no_grad 模式） -----
    chg_model.eval()
    with torch.no_grad():
        C_t, H_t, N_t, O_t = chg_model(Xc_t, Xh_t, Xn_t, Xo_t)

    # ----- 6) detach 并转回 NumPy -----
    C_arr, H_arr, N_arr, O_arr = (
        t.detach().cpu().numpy().squeeze(0) for t in (C_t, H_t, N_t, O_t)
    )

    # ----- 7) 逐个原子用 _charge 公式计算电荷值 -----
    charges = []
    for n, arr, dim in zip(ats, (C_arr, H_arr, N_arr, O_arr), (340, 208, 340, 340)):
        if n == 0:
            continue
        e = 93 if dim == 340 else 58
        for i in range(n):
            charges.append(_charge(arr[i, :e], arr[i, e:]))
    ch = np.array(charges, np.float32)

    # 如果用户指定了写文件，就把预测值写到 “<folder_name>_Pred_CHG.dat”
    if getattr(args, "write_chg", False):
        try:
            # 仅当原始结构文件夹里有 CHGCAR 时才写
            Chgcar.from_file(poscar_path.parent / "CHGCAR")
            fname = poscar_path.parent.name + "_Pred_CHG.dat"
            np.savetxt(fname, ch, fmt="%.6f")
        except FileNotFoundError:
            pass

    return ch

# ------------------------- 导出 ---------------------------------------------#
__all__ = [
    "init_chgmod", "load_pretrained_chg_model",
    "train_chg_model", "infer_charges",
]
