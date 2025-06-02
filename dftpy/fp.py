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

def _load_scalers(paths: Tuple[str, str, str, str]):
    from joblib import load

    try:
        return tuple(load(p) for p in paths)
    except Exception as exc:  # pragma: no cover – keep error readable
        raise FileNotFoundError(
            "MaxAbsScaler .joblib files not found or corrupted. "
            "Ensure the path tuple points to four valid pickles."
        ) from exc


def fp_chg_norm(
    Coef_at1: np.ndarray,
    Coef_at2: np.ndarray,
    Coef_at3: np.ndarray,
    Coef_at4: np.ndarray,
    X_3D1: np.ndarray,
    X_3D2: np.ndarray,
    X_3D3: np.ndarray,
    X_3D4: np.ndarray,
    padding_size: int,
    scaler_paths: Tuple[str, str, str, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate predicted charge coefficient (single scalar per atom) to fingerprints
    *and* run ``MaxAbsScaler`` normalisation.

    Each ``Coef_at#`` is expected to be ``(1, padding_size, 1)``; each ``X_3D#``
    is ``(1, padding_size, feat_dim)``.
    """
    # reshape helpers – keeps code readable
    def _flat(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(padding_size, -1)

    feats = [
        np.concatenate([_flat(X_3D1), _flat(Coef_at1)], axis=1),
        np.concatenate([_flat(X_3D2), _flat(Coef_at2)], axis=1),
        np.concatenate([_flat(X_3D3), _flat(Coef_at3)], axis=1),
        np.concatenate([_flat(X_3D4), _flat(Coef_at4)], axis=1),
    ]  # list of (P, feat+1)

    scalerC, scalerH, scalerN, scalerO = _load_scalers(scaler_paths)
    scalers = [scalerC, scalerH, scalerN, scalerO]

    feats_scaled = [
        s.transform(f) if f.size else f  # allow empty if element absent
        for f, s in zip(feats, scalers)
    ]

    # reshape back to 3‑D tensors
    out = [f.reshape(1, padding_size, -1).astype(np.float32) for f in feats_scaled]
    return tuple(out)


def fp_norm(
    X_C: np.ndarray,
    X_H: np.ndarray,
    X_N: np.ndarray,
    X_O: np.ndarray,
    padding_size: int,
    scaler_paths: Tuple[str, str, str, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply MaxAbsScaler normalisation *without* appending charge coefficient."""
    scalers = _load_scalers(scaler_paths)
    tensors = [X_C, X_H, X_N, X_O]

    outs = []
    for tens, sc in zip(tensors, scalers):
        n_samples = tens.shape[0]
        feat = tens.shape[-1]
        flat = tens.reshape(n_samples * padding_size, feat)
        norm = sc.transform(flat).reshape(n_samples, padding_size, feat).astype(np.float32)
        outs.append(norm)

    return tuple(outs)
