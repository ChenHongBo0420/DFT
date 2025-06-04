"""
dftpy.fp  ·  fingerprint generation & normalisation utilities
=============================================================

Public API
----------
pad_to          – pad a 2-D array on axis-0  
fp_atom         – build per-atom radial / dipole / quadrupole fingerprints  
fp_chg_norm     – fingerprints ⊕ charge-coefficients + MaxAbsScaler  
fp_norm         – fingerprints + MaxAbsScaler (无系数)

Constant
--------
DOS_POINTS = 341  (must stay in sync with dftpy.dos)
"""

from __future__ import annotations

import itertools, os, sys, math
from pathlib import Path
from typing  import List, Tuple

import numpy as np
from joblib import load
from pymatgen.core             import Structure
from pymatgen.io.vasp.outputs  import Poscar

# --------------------------------------------------------------------- public
DOS_POINTS: int = 341
__all__ = ["pad_to", "fp_atom", "fp_chg_norm", "fp_norm", "DOS_POINTS"]

# ---------------------------------------------------------------- helpers
def pad_to(arr: np.ndarray,
           target_rows: int,
           pad_value: float = 0.0) -> np.ndarray:
    """Pad *arr* (n_rows, n_feat) to *target_rows* on axis-0."""
    n, feat = arr.shape
    if n >= target_rows:
        return arr.copy()
    pad = np.full((target_rows - n, feat), pad_value, dtype=arr.dtype)
    return np.vstack([arr, pad])

# ---------------------------------------------------------------- fp builder
def fp_atom(structure: Structure | Poscar | str,
            grid_spacing: float,
            cut_off_rad:  float,
            widest_gaussian: float,
            narrowest_gaussian: float,
            num_gamma: int
) -> Tuple[np.ndarray, np.ndarray, List[List[Structure]], int, List[int]]:
    """
    Return: (fingerprints, basis_mat, sites_by_elem, n_atoms, [nC,nH,nN,nO])
    Fingerprints shape = (n_atoms, 10 * num_gamma * n_present_elements)
    Basis shape        = (n_atoms, 9)
    """
    if isinstance(structure, (str, Path)):
        structure = Poscar.from_file(structure).structure
    elif isinstance(structure, Poscar):
        structure = structure.structure

    elem_order = ["C", "H", "N", "O"]
    idx_map    = {e: i for i, e in enumerate(elem_order)}
    at_elem    = [0, 0, 0, 0]
    sites_by   = [[], [], [], []]

    for s in structure.sites:
        sym = s.specie.symbol
        if sym in idx_map:
            i = idx_map[sym]
            at_elem[i] += 1
            sites_by[i].append(s)

    n_atoms = structure.num_sites
    if sum(at_elem) == 0:
        z = np.zeros
        return z((0, 0), np.float32), z((0, 9), np.float32), sites_by, 0, at_elem

    cart_all = structure.cart_coords.astype(np.float32)
    sigma    = np.logspace(math.log10(narrowest_gaussian),
                           math.log10(widest_gaussian),
                           num=num_gamma)
    gamma_l  = 0.5 / (sigma ** 2)

    blocks: list[np.ndarray] = []
    for elem_sites in sites_by:
        if not elem_sites:
            continue
        # --- neighbour set (self + within cutoff)
        frac_neigh = []
        for site in elem_sites:
            neigh = structure.get_neighbors(site, cut_off_rad)
            frac_neigh.extend([n[0].frac_coords for n in neigh])
            frac_neigh.append(site.frac_coords)
        frac_neigh = np.unique(np.asarray(frac_neigh), axis=0)
        cart_neigh = (frac_neigh @ structure.lattice.matrix).astype(np.float32)

        diff = cart_all[None, :, :] - cart_neigh[:, None, :]
        dist = np.linalg.norm(diff, axis=2)
        with np.errstate(divide="ignore"):
            dist_inv = np.where(dist != 0.0, 1.0 / dist, 0.0)
        fcut = 0.5 * (np.cos(np.pi * np.minimum(dist, cut_off_rad) / cut_off_rad) + 1.0)

        rx, ry, rz = diff[..., 0], diff[..., 1], diff[..., 2]
        radial, dip_x, dip_y, dip_z, quad = [], [], [], [], [[], [], [], [], [], []]

        for g in gamma_l:
            norm = (g / np.pi) ** 1.5
            G = norm * np.exp(-g * dist ** 2) * fcut
            radial.append(G.sum(0))

            for comp, dip_lst in zip((rx, ry, rz), (dip_x, dip_y, dip_z)):
                dip_lst.append((comp * G * dist_inv).sum(0))

            with np.errstate(divide="ignore"):
                r2_inv = np.where(dist != 0.0, dist_inv ** 2, 0.0)
                quad[0].append((rx * rx * G * r2_inv).sum(0))
                quad[1].append((ry * ry * G * r2_inv).sum(0))
                quad[2].append((rz * rz * G * r2_inv).sum(0))
                quad[3].append((rx * ry * G * r2_inv).sum(0))
                quad[4].append((ry * rz * G * r2_inv).sum(0))
                quad[5].append((rz * rx * G * r2_inv).sum(0))

        blk = np.concatenate([
            np.stack(radial),                       # (G, N)
            np.stack(dip_x + dip_y + dip_z),        # (3G, N)
            np.stack(list(itertools.chain(*quad)))  # (6G, N)
        ], axis=0).T.astype(np.float32)             # → (N, 10G)
        blocks.append(blk)

    dset = np.concatenate(blocks, axis=1)

    # --- local orthonormal basis
    basis = []
    for s in structure.sites:
        pos = s.coords
        neigh = sorted(structure.get_neighbors(s, 5.0), key=lambda x: x[1])
        if len(neigh) < 2:
            basis.append(np.eye(3, dtype=np.float32).flatten()); continue
        v1, v2 = neigh[0][0].coords - pos, neigh[1][0].coords - pos
        u3 = np.cross(v1, v2)
        u2 = np.cross(u3, v1)
        u1 = v1 / np.linalg.norm(v1)
        u2 = u2 / (np.linalg.norm(u2) + 1e-8)
        u3 = u3 / (np.linalg.norm(u3) + 1e-8)
        basis.append(np.vstack([u1, u2, u3]).T.astype(np.float32).flatten())
    basis_mat = np.vstack(basis)

    return dset, basis_mat, sites_by, n_atoms, at_elem

# ---------------------------------------------------------------- scaler IO
def _load_scalers(paths: Tuple[str, str, str, str]):
    """Load four MaxAbsScaler pickles; patch sklearn ≤1.4 import path."""
    try:
        import sklearn.preprocessing._data as _m
        sys.modules['sklearn.preprocessing.data'] = _m
    except ImportError:
        pass
    scalers = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Scaler file missing: {p}")
        scalers.append(load(p))
    return tuple(scalers)

# ---------------------------------------------------------------- concat / norm
def _norm_concat(X: np.ndarray,
                 coef: np.ndarray,
                 scaler) -> np.ndarray:
    """Skip scaler if X has zero columns (element absent)."""
    if X.shape[1] == 0:
        Xn = X.astype(np.float32)
    else:
        Xn = scaler.transform(X)
    return np.concatenate([Xn, coef], axis=1)

def fp_chg_norm(
    X_C, X_H, X_N, X_O,
    Coef_C, Coef_H, Coef_N, Coef_O,
    padding_size: int,
    scaler_paths: Tuple[str, str, str, str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sc_C, sc_H, sc_N, sc_O = _load_scalers(scaler_paths)

    def _proc(X, Coef, sc):
        X    = X.reshape(padding_size, -1)
        Coef = Coef.reshape(padding_size, -1)
        return _norm_concat(X, Coef, sc).reshape(1, padding_size, -1)

    return (_proc(X_C, Coef_C, sc_C),
            _proc(X_H, Coef_H, sc_H),
            _proc(X_N, Coef_N, sc_N),
            _proc(X_O, Coef_O, sc_O))

def fp_norm(
    X_C, X_H, X_N, X_O,
    padding_size: int,
    scaler_paths: Tuple[str, str, str, str]
):
    sc_C, sc_H, sc_N, sc_O = _load_scalers(scaler_paths)

    def _norm_block(X, sc):
        nS, P, feat = X.shape
        flat = X.reshape(-1, feat)
        flat_n = sc.transform(flat)
        return flat_n.reshape(nS, P, feat).astype(np.float32)

    return (_norm_block(X_C, sc_C),
            _norm_block(X_H, sc_H),
            _norm_block(X_N, sc_N),
            _norm_block(X_O, sc_O))
