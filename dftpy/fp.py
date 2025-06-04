# dftpy/fp.py  ‑‑ fully revised
"""Fingerprint generation & normalization utilities for DFTpy.

This module now exposes **all** helpers that上层代码依赖：

* ``pad_to`` – pad 2‑D NumPy arrays on the first axis.
* ``fp_atom`` – build radial / dipole / quadrupole fingerprints & local basis for every atom in a structure.
* ``fp_chg_norm`` – concatenate predicted charge coefficients to the fingerprints **and** run ``MaxAbsScaler`` normalisation.
* ``fp_norm`` – apply the same ``MaxAbsScaler`` normalisation when *only* fingerprints are needed.

The constant ``DOS_POINTS`` is kept at **341** – the length used everywhere else (mask creation, network heads, plotting).

The implementation is self‑contained (only ``numpy`` + ``pymatgen`` at import time) so that running
``from dftpy.fp import fp_chg_norm`` will always succeed.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import itertools
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Poscar
import os
import numpy as np
import sys
from joblib import load
# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
DOS_POINTS: int = 341  # <- must stay in sync with dftpy.dos

__all__ = [
    "pad_to",
    "fp_atom",
    "fp_chg_norm",
    "fp_norm",
    "DOS_POINTS",
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def pad_to(arr: np.ndarray, target_rows: int, pad_value: float = 0.0) -> np.ndarray:
    """Pad *arr* along axis‑0 up to *target_rows* using *pad_value*.

    Parameters
    ----------
    arr
        ``(n_rows, n_feat)`` matrix.
    target_rows
        Desired row count after padding.
    pad_value
        Value written into the padded rows.
    """
    n_rows, n_feat = arr.shape
    if n_rows >= target_rows:
        return arr.copy()

    pad = np.full((target_rows - n_rows, n_feat), pad_value, dtype=arr.dtype)
    return np.vstack([arr, pad])


# ---------------------------------------------------------------------------
# Fingerprint builder – largely identical to the earlier version but with
# variable names tidied and a couple of bug‑fixes (radial_fp_elem→radial_vals).
# ---------------------------------------------------------------------------

def fp_atom(
    structure: Structure | Poscar | str,
    grid_spacing: float,
    cut_off_rad: float,
    widest_gaussian: float,
    narrowest_gaussian: float,
    num_gamma: int,
) -> Tuple[np.ndarray, np.ndarray, List[List[Structure]], int, List[int]]:
    """Compute per‑atom fingerprints & local basis.

    Returns
    -------
    dset
        ``(n_atoms, feat_dim)`` fingerprint matrix.
    basis_mat
        ``(n_atoms, 9)`` – flattened 3×3 orthonormal basis per atom.
    sites_elem
        Nested list of four lists (C/H/N/O sites) – empty if that element is absent.
    num_atoms
        Total atom count.
    at_elem
        ``[nC, nH, nN, nO]`` counts for each element type.
    """
    # Accept filename / Poscar / Structure transparently
    if isinstance(structure, (str, Path)):
        structure = Poscar.from_file(structure).structure
    elif isinstance(structure, Poscar):
        structure = structure.structure

    # strict ordering
    elem_order = ["C", "H", "N", "O"]
    elem_to_idx = {e: i for i, e in enumerate(elem_order)}

    at_elem = [0, 0, 0, 0]
    sites_elem: List[List[Structure]] = [[], [], [], []]

    for site in structure.sites:
        sym = site.specie.symbol
        if sym in elem_to_idx:
            idx = elem_to_idx[sym]
            at_elem[idx] += 1
            sites_elem[idx].append(site)
        # silently ignore other elements – can be extended later

    num_atoms = structure.num_sites
    if sum(at_elem) == 0:
        # no C/H/N/O in the structure – return empty arrays so caller can decide what to do
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 9), dtype=np.float32),
            sites_elem,
            0,
            at_elem,
        )

    # ------------------------------------------------------------------
    # Build cartesian grids for fingerprints
    # ------------------------------------------------------------------
    # concatenate coords once – referenced below
    cart_coords_all = structure.cart_coords.astype(np.float32)

    # Gaussian width → gamma list (match original code)
    sigma = np.logspace(np.log10(narrowest_gaussian), np.log10(widest_gaussian), num=num_gamma)
    gamma_list = 0.5 / (sigma ** 2)  # shape (num_gamma,)

    radial_blocks: List[np.ndarray] = []  # list of (n_atoms, num_gamma*10) blocks per element

    for elem_idx, elem_sites in enumerate(sites_elem):
        if not elem_sites:
            continue  # skip absent element

        # unique neighbour positions (including self)
        frac_neigh = []
        for site in elem_sites:
            neighs = structure.get_neighbors(site, cut_off_rad)
            frac_neigh.extend([n[0].frac_coords for n in neighs])
            frac_neigh.append(site.frac_coords)

        frac_neigh = np.unique(np.asarray(frac_neigh), axis=0)
        cart_neigh = (frac_neigh @ structure.lattice.matrix).astype(np.float32)  # (M,3)

        # pairwise differences → (M, n_atoms, 3)
        diff = cart_coords_all[None, :, :] - cart_neigh[:, None, :]
        dist = np.linalg.norm(diff, axis=2)  # (M, n_atoms)
        with np.errstate(divide="ignore"):
            dist_inv = np.where(dist != 0.0, 1.0 / dist, 0.0)

        # cut‑off function f_c(r)
        r_cut = np.minimum(dist, cut_off_rad)
        fcut = (np.cos(np.pi * r_cut / cut_off_rad) + 1.0) * 0.5  # (M, n_atoms)

        # Holder lists for this element
        radial_vals: List[np.ndarray] = []
        dipole_vals = [[] for _ in range(3)]  # x,y,z
        quad_vals = [[] for _ in range(6)]    # xx,yy,zz,xy,yz,zx

        rx, ry, rz = diff[..., 0], diff[..., 1], diff[..., 2]

        for gamma in gamma_list:
            norm = (gamma / np.pi) ** 1.5
            g = norm * np.exp(-gamma * (dist ** 2)) * fcut  # (M, n_atoms)

            radial_vals.append(g.sum(axis=0))

            # dipole = Σ (r_i * g / r)
            for dim, comp in enumerate((rx, ry, rz)):
                dip = np.where(dist != 0.0, comp * g * dist_inv, 0.0).sum(axis=0)
                dipole_vals[dim].append(dip)

            # quadrupole terms Σ (r_a r_b g / r^2)
            with np.errstate(divide="ignore"):
                r2_inv = np.where(dist != 0.0, dist_inv ** 2, 0.0)
                quad_vals[0].append((rx * rx * g * r2_inv).sum(axis=0))  # xx
                quad_vals[1].append((ry * ry * g * r2_inv).sum(axis=0))  # yy
                quad_vals[2].append((rz * rz * g * r2_inv).sum(axis=0))  # zz
                quad_vals[3].append((rx * ry * g * r2_inv).sum(axis=0))  # xy
                quad_vals[4].append((ry * rz * g * r2_inv).sum(axis=0))  # yz
                quad_vals[5].append((rz * rx * g * r2_inv).sum(axis=0))  # zx

        # stack & reshape to (n_atoms, num_gamma*10)
        radial_block = np.stack(radial_vals, axis=0)                      # (G, n_atoms)
        dipole_block = np.stack([np.stack(v, axis=0) for v in dipole_vals], axis=0)  # (3,G,n_atoms)
        quad_block = np.stack([np.stack(v, axis=0) for v in quad_vals], axis=0)      # (6,G,n_atoms)

        block_concat = np.concatenate([
            radial_block,
            dipole_block.reshape(3 * num_gamma, num_atoms),
            quad_block.reshape(6 * num_gamma, num_atoms),
        ], axis=0).T  # -> (n_atoms, G*10)

        radial_blocks.append(block_concat.astype(np.float32))

    # concatenate blocks from all present elements along feature axis
    dset = np.concatenate(radial_blocks, axis=1).astype(np.float32)  # (n_atoms, feat_dim_total)

    # ------------------------------------------------------------------
    # Local orthonormal basis – identical to original logic
    # ------------------------------------------------------------------
    basis_rows = []
    cutoff_nn = 5.0
    for site in structure.sites:
        pos = site.coords
        neighs = sorted(structure.get_neighbors(site, cutoff_nn), key=lambda x: x[1])
        if len(neighs) < 2:
            basis_rows.append(np.eye(3, dtype=np.float32).flatten())
            continue
        v1 = neighs[0][0].coords - pos
        v2 = neighs[1][0].coords - pos
        u3 = np.cross(v1, v2)
        u2 = np.cross(u3, v1)
        u1 = v1 / np.linalg.norm(v1)
        u2 = u2 / (np.linalg.norm(u2) + 1e-8)
        u3 = u3 / (np.linalg.norm(u3) + 1e-8)
        basis_rows.append(np.vstack([u1, u2, u3]).T.astype(np.float32).flatten())

    basis_mat = np.vstack(basis_rows)  # (n_atoms, 9)
    return dset, basis_mat, sites_elem, num_atoms, at_elem


# ---------------------------------------------------------------------------
# Normalisation helpers – rely on joblib.MaxAbsScaler stored on disk
# ---------------------------------------------------------------------------

# def _load_scalers(paths: Tuple[str, str, str, str]):
#     from joblib import load

#     try:
#         return tuple(load(p) for p in paths)
#     except Exception as exc:  # pragma: no cover – keep error readable
#         raise FileNotFoundError(
#             "MaxAbsScaler .joblib files not found or corrupted. "
#             "Ensure the path tuple points to four valid pickles."
#         ) from exc

def _load_scalers(paths: tuple[str, str, str, str]):
    """
    paths: 四个 .joblib 文件的绝对路径。顺序是 (Scale_model_C, Scale_model_H, Scale_model_N, Scale_model_O)。

    本函数先把 “sklearn.preprocessing.data” 这个旧模块名，指向新版的
    “sklearn.preprocessing._data”。然后再逐个调用 joblib.load(...)。
    如果任意一个文件不存在，就抛 FileNotFoundError。
    """

    # 如果 sklearn 版本 >=1.5，MaxAbsScaler 放到 sklearn.preprocessing._data 里
    try:
        import sklearn.preprocessing._data as _data_module
        # 把旧路径 “sklearn.preprocessing.data” 临时指到新版模块
        sys.modules['sklearn.preprocessing.data'] = _data_module
    except ImportError:
        # 如果环境里不是新版 sklearn，就忽略这一步
        pass

    scalers = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"MaxAbsScaler .joblib 文件未找到: {p}\n"
                "请检查 dftpy/scalers/ 目录下是否有 Scale_model_C.joblib, Scale_model_H.joblib, "
                "Scale_model_N.joblib, Scale_model_O.joblib 并且文件名拼写完全一致。"
            )
        scalers.append(load(p))
    return tuple(scalers)

def _norm_concat(
    X_in: np.ndarray,
    coef_in: np.ndarray,
    scaler
) -> np.ndarray:
    
    if X_in.shape[1] == 0:
        arr_norm = X_in.astype(np.float32)           # (P, 0)
    else:
        arr_norm = scaler.transform(X_in)            # (P, feat_dim)

    return np.concatenate([arr_norm, coef_in], axis=1)


def fp_chg_norm(
    X_C: np.ndarray, X_H: np.ndarray, X_N: np.ndarray, X_O: np.ndarray,
    Coef_C: np.ndarray, Coef_H: np.ndarray, Coef_N: np.ndarray, Coef_O: np.ndarray,
    padding_size: int,
    scaler_paths: tuple[str, str, str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    scaler_C, scaler_H, scaler_N, scaler_O = _load_scalers(scaler_paths)

    def _proc(X: np.ndarray, Coef: np.ndarray, scaler):
        X    = X.reshape(padding_size, -1)
        Coef = Coef.reshape(padding_size, -1)
        return _norm_concat(X, Coef, scaler).reshape(1, padding_size, -1)

    return (
        _proc(X_C, Coef_C, scaler_C),
        _proc(X_H, Coef_H, scaler_H),
        _proc(X_N, Coef_N, scaler_N),
        _proc(X_O, Coef_O, scaler_O),
    )
    
def fp_norm(
    X_C: np.ndarray, X_H: np.ndarray, X_N: np.ndarray, X_O: np.ndarray,
    padding_size: int,
    scaler_paths: tuple[str, str, str, str]
):
    # 1) 先 load scalers (并做模块别名 hack)
    scaler_C, scaler_H, scaler_N, scaler_O = _load_scalers(scaler_paths)

    # 2) 对每个类别分别 reshape → transform → reshape 回原形
    def _normalize(X_in: np.ndarray, scaler) -> np.ndarray:
        # X_in: (n_samples, padding_size, feat_dim) → 展平成 (n_samples*padding_size, feat_dim)
        nS, P, feat = X_in.shape
        flat = X_in.reshape(-1, feat)
        # 可能 scaler.scale_.shape=(feat,) 或其他，如果不匹配就底层会报错
        flat_n = scaler.transform(flat)
        return flat_n.reshape(nS, P, feat).astype(np.float32)

    X_Cn = _normalize(X_C, scaler_C)
    X_Hn = _normalize(X_H, scaler_H)
    X_Nn = _normalize(X_N, scaler_N)
    X_On = _normalize(X_O, scaler_O)
    return X_Cn, X_Hn, X_Nn, X_On
