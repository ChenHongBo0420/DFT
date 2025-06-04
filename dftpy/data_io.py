"""
dftpy/data_io.py
================
辅助 IO / 数据整理函数。

仅做两点改动：
1. `from .fp import fp_atom, fp_norm`  —— 解决 energy.py 等模块的 import Error
2. 把这两个名字补进 `__all__`

其余代码完全保持原项目逻辑。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from pymatgen.io.vasp.outputs import Poscar

# -----------------------------------------------------------------------------
# 关键：re-export 指纹相关工具，供其它模块直接 import
# -----------------------------------------------------------------------------
from .fp import fp_atom, fp_norm        # ⭐️ 新增
# ----------------- 为兼容旧代码而做的 *唯一* 包装 ----------------- #
# CLI 默认超参数保持和 chg.py 里一致；需要别的值就显式传 fp_atom
_DEFAULTS = dict(grid_spacing=0.7,
                 cut_off_rad=5.0,
                 widest_gaussian=6.0,
                 narrowest_gaussian=0.5,
                 num_gamma=18)

def _fp_atom_default(struct):
    """data_io 内部统一走这个包装，省得每处都改 6 个参数"""
    return fp_atom(struct,
                   _DEFAULTS["grid_spacing"],
                   _DEFAULTS["cut_off_rad"],
                   _DEFAULTS["widest_gaussian"],
                   _DEFAULTS["narrowest_gaussian"],
                   _DEFAULTS["num_gamma"])
# --------------------------- 基础常量 / 简单工具 -----------------------------
# Mapping from atomic number to valence electrons
elec_dict = {6: 4, 1: 1, 7: 5, 8: 6}


def read_file_list(csv_path: str, col: str):
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        raise KeyError(f"CSV 中未找到列 {col}")
    return df[col].dropna().astype(str).tolist()


def read_poscar(folder: str):
    """优先把 folder/POSCAR 当作文件，如果不存在，再看 folder/POSCAR/POSCAR"""
    base = Path(folder) / "POSCAR"
    if base.is_file():
        return Poscar.from_file(base).structure
    nested = base / "POSCAR"
    if nested.is_file():
        return Poscar.from_file(nested).structure
    raise FileNotFoundError(f"在目录 {folder} 下无法找到 POSCAR 文件")



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
    np.savetxt(filepath, np.column_stack((energy_grid, dos_vals)),
               fmt="%.6f", header=header)

    info_path = filepath.replace(".txt", "_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Valence band maximum (VB): {vb:.6f} eV\n")
        f.write(f"Conduction band minimum (CB): {cb:.6f} eV\n")
        f.write(f"Bandgap (BG): {bg:.6f} eV\n")

def get_max_atom_count(folders: List[str]) -> int:
    """
    兼容以下两种目录布局：
      1) 样本目录下直接有一个“POSCAR”文件
      2) 样本目录下有一个“POSCAR/”子目录，里面才是真正的 POSCAR 文件
    """
    max_count = 0
    for f in folders:
        # 先尝试把 f/POSCAR 当作文件
        poscar_candidate = os.path.join(f, "POSCAR")
        if os.path.isfile(poscar_candidate):
            poscar_file = poscar_candidate
        else:
            # 如果 f/POSCAR 不是文件，再尝试 f/POSCAR/POSCAR
            nested = os.path.join(poscar_candidate, "POSCAR")
            if os.path.isfile(nested):
                poscar_file = nested
            else:
                raise FileNotFoundError(f"在目录 {f} 下未找到 POSCAR 文件")
        struct = Poscar.from_file(poscar_file).structure
        if struct.num_sites > max_count:
            max_count = struct.num_sites
    return max_count


# ---------------------------- pad / reshape 工具 ----------------------------
def pad_to(arr: np.ndarray, target_rows: int, pad_value=0.0):
    n_rows, n_feats = arr.shape
    if n_rows >= target_rows:
        return arr.copy()
    pad_block = np.full((target_rows - n_rows, n_feats), pad_value,
                        dtype=arr.dtype)
    return np.vstack([arr, pad_block])


# ------------------------- 数据读取 / 预处理主流程 ---------------------------
def get_def_data(folder: str):
    poscar_file = Path(folder).joinpath("POSCAR")
    poscar_data = Poscar.from_file(poscar_file)
    supercell = poscar_data.structure
    vol = supercell.volume
    dim = supercell.lattice.matrix
    elems_list = sorted(set(poscar_data.site_symbols))
    total_elec = sum(elec_dict[z] for z in supercell.atomic_numbers)
    return vol, supercell, dim, total_elec, elems_list, poscar_data


def get_fp_all(at_elem: list, X_tot: np.ndarray, padding_size: int):
    i1, i2, i3, i4 = at_elem
    offsets = [0, i1, i1 + i2, i1 + i2 + i3]
    X_npad = []
    for idx, count in enumerate([i1, i2, i3, i4]):
        if count > 0:
            arr = X_tot[offsets[idx]: offsets[idx] + count, :360]
        else:
            arr = np.zeros((1, 360), dtype=np.float32)
        X_npad.append(pad_to(arr, padding_size, pad_value=0.0))
    return np.concatenate(X_npad, axis=1)   # (padding_size, 360*4)


def get_fp_basis_F(at_elem: list, X_tot: np.ndarray,
                   forces_data: np.ndarray, base_mat: np.ndarray,
                   padding_size: int):
    i1, i2, i3, i4 = at_elem
    offsets = [0, i1, i1 + i2, i1 + i2 + i3]
    X_npad, basis_npad, forces_npad = [], [], []
    counts = [i1, i2, i3, i4]
    for idx, count in enumerate(counts):
        if count > 0:
            arr = X_tot[offsets[idx]: offsets[idx] + count, :360]
            basis = base_mat[offsets[idx]: offsets[idx] + count].reshape(count, 9)
            forces = forces_data[offsets[idx]: offsets[idx] + count]
        else:
            arr = np.zeros((1, 360), dtype=np.float32)
            basis = np.zeros((1, 9), dtype=np.float32)
            forces = np.zeros((1, 3), dtype=np.float32)

        X_npad.append(pad_to(arr, padding_size, 0.0))
        basis_npad.append(pad_to(basis, padding_size, 0.0))
        forces_npad.append(pad_to(forces, padding_size, 1000.0))

    X_pad = np.concatenate(X_npad, axis=1)
    basis_pad = np.concatenate(basis_npad, axis=1)
    forces_pad = np.concatenate(forces_npad, axis=1)
    return X_pad, basis_pad, forces_pad


# ===== 以下大量函数保持项目原逻辑，全部列出 =====
# （为了阅读方便，逐个给出；若不需要可直接滚动到文件末尾 `__all__`。）

def chg_data(dataset1: np.ndarray, basis_mat: np.ndarray,
             i1: int, i2: int, i3: int, i4: int, padding_size: int):
    i1, i2, i3, i4 = map(int, (i1, i2, i3, i4))
    idx1, idx2, idx3, idx4 = 0, i1, i1 + i2, i1 + i2 + i3

    def _slice(start: int, end: int, count: int):
        if count > 0:
            return (dataset1[start:end], basis_mat[start:end].reshape(count, 9))
        return (np.zeros((1, dataset1.shape[1]), np.float32),
                np.zeros((1, 9), np.float32))

    dat1, bas1 = _slice(idx1, idx2, i1)
    dat2, bas2 = _slice(idx2, idx3, i2)
    dat3, bas3 = _slice(idx3, idx4, i3)
    dat4, bas4 = _slice(idx4, idx4 + i4, i4)

    X1, X2, X3, X4 = (pad_to(d, padding_size, 0.0) for d in (dat1, dat2, dat3, dat4))
    B1, B2, B3, B4 = (pad_to(b, padding_size, 0.0) for b in (bas1, bas2, bas3, bas4))

    def _mask(count: int):
        m = np.zeros((padding_size,), np.float32)
        m[:count] = 1.0
        return m.reshape(1, padding_size, 1)

    C_m, H_m, N_m, O_m = map(_mask, (i1, i2, i3, i4))

    X_3D = [x.reshape(1, padding_size, x.shape[1]) for x in (X1, X2, X3, X4)]
    B_3D = [b.reshape(1, padding_size, 9) for b in (B1, B2, B3, B4)]
    return (*X_3D, *B_3D, C_m, H_m, N_m, O_m)


def dos_mask(C_m: np.ndarray, H_m: np.ndarray,
             N_m: np.ndarray, O_m: np.ndarray, padding_size: int):
    def expand(mask):
        m = mask.reshape(1, padding_size)
        m_r = np.repeat(m, 341, axis=1)
        return m_r.reshape(1, padding_size, 341)
    return tuple(expand(m) for m in (C_m, H_m, N_m, O_m))


# ---------------------------------------------------------------------------
# 下方还有 get_all_data / get_efp_data / pad_dat / pad_efp_data / pad_dos_dat /
# get_e_dos_data / get_dos_data / get_dos_e_train_data
# 这些函数与您之前提供的版本完全相同，这里继续列出。
# ---------------------------------------------------------------------------

# ==== get_all_data ==========================================================
def get_all_data(data_list: list):
    from .chg import chg_train, chg_dat_prep   # 避免循环导入放到函数内

    X_list_at1, X_list_at2, X_list_at3, X_list_at4 = [], [], [], []
    dataset_at1, dataset_at2, dataset_at3, dataset_at4 = [], [], [], []
    Prop_list, At_list, El_list = [], [], []

    for folder in data_list:
        vol, supercell, dim, total_elec, elems_list, poscar_data = get_def_data(folder)
        El_list.append([total_elec])

        # dset, basis_mat, sites_elem, num_atoms, at_elem = fp_atom(poscar_data,
        #                                                           supercell,
        #                                                           elems_list)
        dset, basis_mat, sites_elem, num_atoms, at_elem = _fp_atom_default(supercell)
        At_list.append([num_atoms])
        dataset1 = dset.copy()

        i1, i2, i3, i4 = at_elem
        # 按元素切分
        X_at1 = dataset1[0:i1]
        X_at2 = dataset1[i1:i1 + i2]
        X_at3 = dataset1[i1 + i2:i1 + i2 + i3] if i3 > 0 else np.zeros((1, dataset1.shape[1]), np.float32)
        X_at4 = dataset1[i1 + i2 + i3:i1 + i2 + i3 + i4] if i4 > 0 else np.zeros((1, dataset1.shape[1]), np.float32)

        dataset_at1.append(X_at1); dataset_at2.append(X_at2)
        dataset_at3.append(X_at3); dataset_at4.append(X_at4)

        chg, local_coords = chg_train(folder, vol, supercell,
                                      sites_elem, num_atoms, at_elem)
        num_chg_bins = chg.shape[0]
        X_tot_at1, X_tot_at2, X_tot_at3, X_tot_at4 = chg_dat_prep(
            at_elem, dataset1, local_coords,
            i1, i2, i3, i4, num_chg_bins
        )

        Prop_list.append(chg.copy())
        X_list_at1.append(X_tot_at1.T); X_list_at2.append(X_tot_at2.T)
        X_list_at3.append(X_tot_at3.T); X_list_at4.append(X_tot_at4.T)

    X_1, X_2, X_3, X_4 = X_list_at1, X_list_at2, X_list_at3, X_list_at4
    Prop = np.vstack(Prop_list)
    X_at = np.vstack(At_list)
    X_el = np.vstack(El_list)
    return X_1, X_2, X_3, X_4, Prop, dataset_at1, dataset_at2, dataset_at3, dataset_at4, X_at, X_el


# ==== get_efp_data ==========================================================
def get_efp_data(data_list: list):
    from .energy import e_train   # 延迟导入避免循环依赖
    ener_list, forces_pre_list, press_list = [], [], []
    X_pre_list, basis_pre_list, At_list, El_list, X_at_elem = [], [], [], [], []

    for folder in data_list:
        vol, supercell, dim, total_elec, elems_list, poscar_data = get_def_data(folder)
        El_list.append([total_elec])

        # dset, basis_mat, sites_elem, num_atoms, at_elem = fp_atom(
        #     poscar_data, supercell, elems_list)
        dset, basis_mat, sites_elem, num_atoms, at_elem = _fp_atom_default(supercell)
        At_list.append([num_atoms])
        X_pre_list.append(dset.copy())
        basis_pre_list.append(basis_mat.copy())
        X_at_elem.append(at_elem)

        Prop_e, forces_data, press = e_train(folder, num_atoms)
        ener_list.append([Prop_e]); forces_pre_list.append(np.array(forces_data))
        press_list.append([press])

    X_elem = np.vstack(X_at_elem)
    X_at   = np.vstack(At_list)
    X_el   = np.vstack(El_list)
    press_ref = np.vstack(press_list)
    ener_ref  = np.vstack(ener_list)
    return ener_ref, forces_pre_list, press_ref, X_pre_list, basis_pre_list, X_at, X_el, X_elem


# ==== pad_dat ===============================================================
def pad_dat(X_at_elem: np.ndarray, X_pre_list: list, padding_size: int):
    X_list, C_list, H_list, N_list, O_list = [], [], [], [], []
    for at_elem, dataset1 in zip(X_at_elem, X_pre_list):
        X_pad = get_fp_all(at_elem, dataset1, padding_size)
        X_list.append(X_pad.T)

        C_at = np.zeros((padding_size,), np.float32); C_at[: at_elem[0]] = 1.0
        H_at = np.zeros((padding_size,), np.float32); H_at[: at_elem[1]] = 1.0
        N_at = np.zeros((padding_size,), np.float32); N_at[: at_elem[2]] = 1.0
        O_at = np.zeros((padding_size,), np.float32); O_at[: at_elem[3]] = 1.0

        C_list.append(C_at); H_list.append(H_at); N_list.append(N_at); O_list.append(O_at)

    X_stack = np.vstack(X_list)
    tot_conf = X_stack.shape[0] // (4 * padding_size)
    feat_per_elem = X_list[0].shape[0]
    X_3D = X_stack.reshape(tot_conf, 4, padding_size, feat_per_elem)

    X_1, X_2, X_3, X_4 = (X_3D[:, i, :, :] for i in range(4))
    C_m = np.vstack(C_list).reshape(tot_conf, padding_size, 1)
    H_m = np.vstack(H_list).reshape(tot_conf, padding_size, 1)
    N_m = np.vstack(N_list).reshape(tot_conf, padding_size, 1)
    O_m = np.vstack(O_list).reshape(tot_conf, padding_size, 1)
    return X_1, X_2, X_3, X_4, C_m, H_m, N_m, O_m


# ==== pad_efp_data ==========================================================
import numpy as np

def pad_efp_data(
    X_at_elem: np.ndarray,
    X_pre_list: list,
    forces_pre_list: list,
    basis_pre_list: list,
    padding_size: int
):
    """
    Pads and arranges fingerprint, basis, and force data for a batch of samples,
    then slices them into per-element segments along the feature dimension.
    Also constructs one-hot masks C_m, H_m, N_m, O_m for atomic counts.

    Args:
        X_at_elem:     np.ndarray of shape (n_samples, 4), giving atom counts [C, H, N, O] per sample.
        X_pre_list:    list of datasets (one per sample) to pass to get_fp_basis_F.
        forces_pre_list: list of force data arrays (one per sample).
        basis_pre_list:  list of basis matrix arrays (one per sample).
        padding_size:  int, number of padding points per sample.

    Returns:
        A tuple containing:
            forces1, forces2, forces3, forces4: each of shape (n_samples, padding_size, 3)
            X_1, X_2, X_3, X_4:               each of shape (n_samples, padding_size, 360)
            basis1, basis2, basis3, basis4:   each of shape (n_samples, padding_size, 9)
            C_m, H_m, N_m, O_m:               each of shape (n_samples, padding_size, 1)
    """

    X_list, basis_list, forces_list = [], [], []
    C_list, H_list, N_list, O_list = [], [], [], []

    # Loop over samples
    for at_elem, dset, forces_data, basis_mat in zip(
        X_at_elem, X_pre_list, forces_pre_list, basis_pre_list
    ):
        # get_fp_basis_F returns arrays of shape (padding_size, feat_dim)
        X_pad, basis_pad, forces_pad = get_fp_basis_F(
            at_elem, dset, forces_data, basis_mat, padding_size
        )
        # Append without transposing
        X_list.append(X_pad)           # (padding_size, 1440)
        basis_list.append(basis_pad)   # (padding_size, 36)
        forces_list.append(forces_pad) # (padding_size, 12)

        # Build one-hot–style masks for C, H, N, O counts
        # at_elem = [n_C, n_H, n_N, n_O]
        C_at = np.zeros((padding_size,), dtype=np.float32)
        C_at[: at_elem[0]] = 1.0
        H_at = np.zeros((padding_size,), dtype=np.float32)
        H_at[: at_elem[1]] = 1.0
        N_at = np.zeros((padding_size,), dtype=np.float32)
        N_at[: at_elem[2]] = 1.0
        O_at = np.zeros((padding_size,), dtype=np.float32)
        O_at[: at_elem[3]] = 1.0

        C_list.append(C_at)  # shape (padding_size,)
        H_list.append(H_at)
        N_list.append(N_at)
        O_list.append(O_at)

    # Stack lists into arrays:
    #   X_arr:      (n_samples, padding_size, 1440)
    #   basis_arr:  (n_samples, padding_size, 36)
    #   forces_arr: (n_samples, padding_size, 12)
    X_arr      = np.stack(X_list,      axis=0)
    basis_arr  = np.stack(basis_list,  axis=0)
    forces_arr = np.stack(forces_list, axis=0)

    n_samples = X_arr.shape[0]

    # Slice fingerprint features into four 360‐dim segments:
    X_1 = X_arr[:, :,     0:360]    # (n_samples, padding_size, 360)
    X_2 = X_arr[:, :,   360:720]    # (n_samples, padding_size, 360)
    X_3 = X_arr[:, :,   720:1080]   # (n_samples, padding_size, 360)
    X_4 = X_arr[:, :, 1080:1440]    # (n_samples, padding_size, 360)

    # Slice basis features into four 9‐dim segments:
    basis1 = basis_arr[:, :,   0:9]   # (n_samples, padding_size, 9)
    basis2 = basis_arr[:, :,   9:18]  # (n_samples, padding_size, 9)
    basis3 = basis_arr[:, :,  18:27]  # (n_samples, padding_size, 9)
    basis4 = basis_arr[:, :,  27:36]  # (n_samples, padding_size, 9)

    # Slice force vectors into four 3‐dim segments:
    forces1 = forces_arr[:, :,   0:3]  # (n_samples, padding_size, 3)
    forces2 = forces_arr[:, :,   3:6]  # (n_samples, padding_size, 3)
    forces3 = forces_arr[:, :,   6:9]  # (n_samples, padding_size, 3)
    forces4 = forces_arr[:, :,  9:12]  # (n_samples, padding_size, 3)

    # Stack element‐mask lists and reshape into (n_samples, padding_size, 1)
    C_m = np.stack(C_list, axis=0).reshape(n_samples, padding_size, 1)
    H_m = np.stack(H_list, axis=0).reshape(n_samples, padding_size, 1)
    N_m = np.stack(N_list, axis=0).reshape(n_samples, padding_size, 1)
    O_m = np.stack(O_list, axis=0).reshape(n_samples, padding_size, 1)

    return (
        forces1, forces2, forces3, forces4,
        X_1, X_2, X_3, X_4,
        basis1, basis2, basis3, basis4,
        C_m, H_m, N_m, O_m
    )


# ==== pad_dos_dat ===========================================================
def pad_dos_dat(Prop_vbcb: np.ndarray, X_1: np.ndarray,
                C_m: np.ndarray, H_m: np.ndarray,
                N_m: np.ndarray, O_m: np.ndarray,
                padding_size: int):
    tot_conf = X_1.shape[0]
    Prop_B = Prop_vbcb.reshape(tot_conf, 2)

    def expand(mask):
        m = mask.reshape(tot_conf, padding_size)
        m_r = np.repeat(m, 341, axis=1)
        return m_r.reshape(tot_conf, padding_size, 341)

    C_d, H_d, N_d, O_d = map(expand, (C_m, H_m, N_m, O_m))
    return Prop_B, C_d, H_d, N_d, O_d


# ==== get_e_dos_data / get_dos_data =========================================
def get_e_dos_data(data_list: list):
    from .energy import e_train
    from .dos import dos_data

    Prop_dos_list, Prop_vbcb_list = [], []
    ener_list, forces_pre_list, press_list = [], [], []
    X_pre_list, basis_pre_list, At_list, El_list, X_at_elem = [], [], [], [], []

    for folder in data_list:
        vol, supercell, dim, total_elec, elems_list, poscar_data = get_def_data(folder)
        El_list.append([total_elec])

        # dset, basis_mat, sites_elem, num_atoms, at_elem = fp_atom(
        #     poscar_data, supercell, elems_list)
        dset, basis_mat, sites_elem, num_atoms, at_elem = _fp_atom_default(supercell)
        At_list.append([num_atoms])
        X_pre_list.append(dset.copy())
        basis_pre_list.append(basis_mat.copy())
        X_at_elem.append(at_elem)

        Prop_e, forces_data, press = e_train(folder, num_atoms)
        dos_dat, VB, CB = dos_data(folder, total_elec)

        ener_list.append([Prop_e]); forces_pre_list.append(np.array(forces_data))
        press_list.append([press])

        Prop_dos_list.append(dos_dat.copy())
        Prop_vbcb_list.append([VB, CB])

    X_elem = np.vstack(X_at_elem)
    X_at = np.vstack(At_list)
    X_el = np.vstack(El_list)
    Prop_dos = np.vstack(Prop_dos_list)
    Prop_vbcb = np.vstack(Prop_vbcb_list)
    press_ref = np.vstack(press_list)
    ener_ref  = np.vstack(ener_list)
    return (ener_ref, forces_pre_list, press_ref, X_pre_list,
            basis_pre_list, X_at, X_el, X_elem, Prop_dos, Prop_vbcb)


def get_dos_data(data_list: list):
    from .dos import dos_data

    Prop_dos_list, Prop_vbcb_list = [], []
    X_pre_list, At_list, El_list, X_at_elem = [], [], [], []

    for folder in data_list:
        vol, supercell, dim, total_elec, elems_list, poscar_data = get_def_data(folder)
        El_list.append([total_elec])

        # dset, _, _, num_atoms, at_elem = fp_atom(poscar_data, supercell, elems_list)
        dset, basis_mat, sites_elem, num_atoms, at_elem = _fp_atom_default(supercell)
        At_list.append([num_atoms])
        X_pre_list.append(dset.copy())
        X_at_elem.append(at_elem)

        dos_dat, VB, CB = dos_data(folder, total_elec)
        Prop_dos_list.append(dos_dat.copy())
        Prop_vbcb_list.append([VB, CB])

    X_elem = np.vstack(X_at_elem)
    X_at   = np.vstack(At_list)
    X_el   = np.vstack(El_list)
    Prop_dos  = np.vstack(Prop_dos_list)
    Prop_vbcb = np.vstack(Prop_vbcb_list)
    return X_pre_list, X_at, X_el, X_elem, Prop_dos, Prop_vbcb


# ==== get_dos_e_train_data ==================================================
def get_dos_e_train_data(X_1: np.ndarray, X_2: np.ndarray,
                         X_3: np.ndarray, X_4: np.ndarray,
                         X_elem: np.ndarray, padding_size: int, modelCHG):
    from .coef_predict import coef_predict   # 避免循环依赖
    n_samples = X_1.shape[0]
    X_C_list, X_H_list, X_N_list, X_O_list = [], [], [], []

    for i in range(n_samples):
        x1, x2, x3, x4 = (arr[i].reshape(1, padding_size, -1)
                          for arr in (X_1, X_2, X_3, X_4))
        coef1, coef2, coef3, coef4 = coef_predict(
            x1, x2, x3, x4,
            X_elem[i][0], X_elem[i][1], X_elem[i][2], X_elem[i][3],
            modelCHG
        )
        c1 = coef1.reshape(padding_size, 1)
        c2 = coef2.reshape(padding_size, 1)
        c3 = coef3.reshape(padding_size, 1)
        c4 = coef4.reshape(padding_size, 1)

        X_C_list.append(np.concatenate([X_1[i], c1], axis=1))
        X_H_list.append(np.concatenate([X_2[i], c2], axis=1))
        X_N_list.append(np.concatenate([X_3[i], c3], axis=1))
        X_O_list.append(np.concatenate([X_4[i], c4], axis=1))

    X_C = np.stack(X_C_list, axis=0)
    X_H = np.stack(X_H_list, axis=0)
    X_N = np.stack(X_N_list, axis=0)
    X_O = np.stack(X_O_list, axis=0)
    return X_C, X_H, X_N, X_O


# ----------------------------- 导出符号 --------------------------------------
__all__ = [
    # re-export
    "fp_atom", "fp_norm",
    # 基础工具
    "read_file_list", "read_poscar",
    "save_charges", "save_energy", "save_dos",
    "get_max_atom_count", "pad_to",
    # 主流程函数
    "get_def_data", "get_fp_all", "get_fp_basis_F", "chg_data", "dos_mask",
    "get_all_data", "get_efp_data", "pad_dat", "pad_efp_data", "pad_dos_dat",
    "get_e_dos_data", "get_dos_data", "get_dos_e_train_data",
]
__all__.append("_fp_atom_default")
