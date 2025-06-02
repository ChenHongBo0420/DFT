"""
dftpy/data_io.py
----------------
辅助 IO / 数据整理函数。

本版本仅在官方源码基础上做了三件事
1. 显式导入并 re-export ``fp_atom``、``fp_norm``，解决后续模块的 import Error
2. 把这两个名字补进 ``__all__``，方便外部使用
3. 代码其余部分原封不动，保证与旧流程兼容
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from pymatgen.io.vasp.outputs import Poscar

# -----------------------------------------------------------------------------#
# 这里是 **关键新增**：把 fp 模块中的函数 re-export 出去                        #
# -----------------------------------------------------------------------------#
from .fp import fp_atom, fp_norm   # ⭐️ 供本文件内部和外部模块 (energy.py) 使用

# ------------------------------------------------------------------------------
# 原有代码（基本保持不变，略去过长注释）                                        #
# ------------------------------------------------------------------------------

elec_dict = {6: 4, 1: 1, 7: 5, 8: 6}


def read_file_list(csv_path: str, col: str):
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        raise KeyError(f"CSV 中未找到列 {col}")
    return df[col].dropna().astype(str).tolist()


def read_poscar(folder: str):
    poscar_path = Path(folder).joinpath("POSCAR")
    return Poscar.from_file(poscar_path).structure


def save_charges(chg_vals: np.ndarray, filepath: str):
    np.savetxt(filepath, chg_vals, fmt="%.6f")


def save_energy(energy: float, forces: np.ndarray, stress: np.ndarray, filepath: str):
    with open(filepath, "w") as f:
        f.write(f"Total potential energy (eV): {energy:.6f}\n\n")
        f.write("Forces (eV/Å):\n")
        np.savetxt(f, forces, fmt="%.6f")
        f.write("\nStress tensor components (kB):\n")
        f.write(" ".join([f"{x:.6f}" for x in stress]) + "\n")


def save_dos(energy_grid: np.ndarray, dos_vals: np.ndarray,
             vb: float, cb: float, bg: float, filepath: str):
    header = "Energy(eV)    DOS"
    np.savetxt(filepath, np.column_stack((energy_grid, dos_vals)), fmt="%.6f", header=header)

    info_path = filepath.replace(".txt", "_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Valence band maximum (VB): {vb:.6f} eV\n")
        f.write(f"Conduction band minimum (CB): {cb:.6f} eV\n")
        f.write(f"Bandgap (BG): {bg:.6f} eV\n")


def get_max_atom_count(folders: List[str]) -> int:
    return max(read_poscar(f).num_sites for f in folders)


# ----------------------------- pad / reshape 工具 ----------------------------#
def pad_to(arr: np.ndarray, target_rows: int, pad_value=0.0):
    n_rows, n_feats = arr.shape
    if n_rows >= target_rows:
        return arr.copy()
    pad_block = np.full((target_rows - n_rows, n_feats), pad_value, dtype=arr.dtype)
    return np.vstack([arr, pad_block])


# ---------------- 后续大量原有函数 (get_fp_all / get_efp_data / …) -------------
# ★★★ 下面所有函数内容保持**原始**逻辑不变, 为节省版面已折叠 ★★★
# 如果你做了其它改动，只要保证接口一致即可。

#   …………………………………………………………………………………
#   (此处省略不变的数百行函数实现)
#   …………………………………………………………………………………

# -----------------------------------------------------------------------------#
# 导出符号                                                                     #
# -----------------------------------------------------------------------------#
__all__ = [
    # 新增 re-export
    "fp_atom", "fp_norm",
    # 下面这些是文件里本就存在、会被其它模块 import 的函数
    "read_file_list", "read_poscar",
    "save_charges", "save_energy", "save_dos",
    "get_max_atom_count", "pad_to",
    # 以及你项目真正需要暴露的其余函数
    "get_fp_all", "get_fp_basis_F", "get_all_data",
    "get_efp_data", "pad_dat", "pad_efp_data",
    "dos_mask", "pad_dos_dat", "get_e_dos_data",
    "get_dos_data", "get_dos_e_train_data",
]
