# dftpy/fp.py  – revised version
# -----------------------------------------------------------------------------
#  Generating atomic fingerprints (radial + dipole + quadrupole) in pure NumPy.
#  This implementation removes the NameError produced by the mixed use of
#  `rad_fp_elem`/`radial_fp_elem`, keeps a single pad_to helper, and does **not**
#  silently override the global padding_size used by the training / inference
#  pipeline – padding should be handled by the caller.
# -----------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
from pymatgen.core import Structure
from typing import List, Tuple

# ----------------------------------------------------------------------------
# Helper: pad a 2-D array (n_rows × n_feats) with `pad_value` until it reaches
# `target_rows`.
# ----------------------------------------------------------------------------

def pad_to(arr: np.ndarray, target_rows: int, pad_value: float = 0.0) -> np.ndarray:
    """Pad *rows* of a 2-D array. If the array already has ≥target_rows, it is
    returned unchanged (but copied)."""
    n_rows, n_feats = arr.shape
    if n_rows >= target_rows:
        return arr.copy()

    pad = np.full((target_rows - n_rows, n_feats), pad_value, dtype=arr.dtype)
    return np.vstack((arr, pad))


# ----------------------------------------------------------------------------
# Core routine: build atomic fingerprints for C/H/N/O atoms.
# ----------------------------------------------------------------------------

def fp_atom(
    structure: Structure,
    grid_spacing: float,
    cut_off_rad: float,
    widest_gaussian: float,
    narrowest_gaussian: float,
    num_gamma: int,
) -> Tuple[np.ndarray, np.ndarray, List[List], int, List[int]]:
    """Return *(dset, basis_mat, sites_elem, num_atoms, at_elem)* where

    * **dset**      (n_atoms, feat_dim_total) – concatenated fingerprint per atom
    * **basis_mat** (n_atoms, 9)            – flattened local orthogonal axes
    * **sites_elem** list of 4 lists of Site – [C,H,N,O] order
    * **num_atoms** total number of atoms in *structure*
    * **at_elem**   [ nC, nH, nN, nO ]

    The function itself **does not pad** to any unified *padding_size*; the
    caller (usually *data_io.chg_data* or similar) decides how much to pad.
    """

    # 1. Count atoms per element ------------------------------------------------
    elem_to_idx = {"C": 0, "H": 1, "N": 2, "O": 3}
    at_elem = [0, 0, 0, 0]
    sites_elem: List[List] = [[], [], [], []]

    for site in structure.sites:
        sym = site.specie.symbol
        idx = elem_to_idx.get(sym, None)
        if idx is not None:
            at_elem[idx] += 1
            sites_elem[idx].append(site)
        # (atoms of other species are ignored silently)

    num_atoms = structure.num_sites

    # Early-exit: if structure contains no C/H/N/O atoms -----------------------
    if sum(at_elem) == 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 9), dtype=np.float32),
            sites_elem,
            0,
            at_elem,
        )

    # 2. Cartesian coordinates of *all* atoms (order preserved from structure)
    cart_grid_K = structure.cart_coords  # shape = (num_atoms, 3)

    # 3. Build σ (gamma) list for Gaussian basis -------------------------------
    sigma = np.logspace(
        np.log10(narrowest_gaussian), np.log10(widest_gaussian), num_gamma
    )
    gamma_list = 0.5 / (sigma**2)  # 1/(2σ²)

    # 4. For each element build radial / dipole / quadrupole contributions ------
    # Feature layout per *gamma*:
    #   1    radial
    #   3    dipole (x,y,z)
    #   6    quadrupole (xx,yy,zz,xy,yz,zx)
    # → 10 × num_gamma dimensions per element

    elem_feature_blocks: List[np.ndarray] = []

    for elem_idx, elem_sites in enumerate(sites_elem):
        if not elem_sites:
            continue  # skip missing element type

        # Unique neighbour positions for the *current* element -----------------
        # Collect fractional coords of the atom itself + neighbours within cutoff
        neighbours_frac: List[np.ndarray] = []
        for site in elem_sites:
            # self
            neighbours_frac.append(site.frac_coords)
            for neigh_site, dist in structure.get_neighbors(site, cut_off_rad):
                neighbours_frac.append(neigh_site.frac_coords)

        frac_unique = (
            np.unique(np.vstack(neighbours_frac), axis=0) if neighbours_frac else np.empty((0, 3))
        )
        cart_unique = frac_unique @ structure.lattice.matrix  # (n_unique, 3)

        # Pair-wise vectors & distances ----------------------------------------
        # cart_unique[:,None,:] – (n_unique,1,3)
        # cart_grid_K[None,:,:] – (1,n_atoms,3)
        diff = cart_grid_K[None, :, :] - cart_unique[:, None, :]
        rad = np.linalg.norm(diff, axis=2)  # (n_unique, n_atoms)

        with np.errstate(divide="ignore"):
            rad_inv = np.where(rad != 0.0, 1.0 / rad, 0.0)

        # Cosine cutoff ---------------------------------------------------------
        r_cut = np.minimum(rad, cut_off_rad)
        cutoff = 0.5 * (np.cos(np.pi * r_cut / cut_off_rad) + 1.0)  # (n_unique, n_atoms)

        # Allocate feature holders --------------------------------------------
        radial_fp_elem: List[np.ndarray] = []  # length = num_gamma, each (n_atoms,)
        dipole_elem = [[]
                        for _ in range(3)]     # 3 × num_gamma arrays
        quad_elem = [[]
                      for _ in range(6)]       # 6 × num_gamma arrays

        # Loop over all γ values ----------------------------------------------
        for gamma in gamma_list:
            norm = (gamma / np.pi) ** 1.5
            g = norm * np.exp(-gamma * rad**2) * cutoff   # (n_unique, n_atoms)

            # 1) radial
            radial_fp_elem.append(g.sum(axis=0))

            # 2) dipole   Σ (Δr_i * g / r)
            for dim in range(3):
                comp = diff[:, :, dim]
                dip = np.where(rad != 0.0, comp * g * rad_inv, 0.0).sum(axis=0)
                dipole_elem[dim].append(dip)

            # 3) quadrupole Σ ((Δr_a Δr_b) g / r²)
            rx, ry, rz = diff[:, :, 0], diff[:, :, 1], diff[:, :, 2]
            with np.errstate(divide="ignore"):
                r_inv2 = np.where(rad != 0.0, rad_inv**2, 0.0)
                quad_arrays = [
                    (rx * rx) * g * r_inv2,  # xx
                    (ry * ry) * g * r_inv2,  # yy
                    (rz * rz) * g * r_inv2,  # zz
                    (rx * ry) * g * r_inv2,  # xy
                    (ry * rz) * g * r_inv2,  # yz
                    (rz * rx) * g * r_inv2,  # zx
                ]
                for q_idx, q_val in enumerate(quad_arrays):
                    quad_elem[q_idx].append(q_val.sum(axis=0))

        # Stack feature lists --------------------------------------------------
        radial_block = np.stack(radial_fp_elem, axis=0)            # (num_gamma, n_atoms)
        dipole_block = np.stack([np.stack(v, axis=0) for v in dipole_elem], axis=0)
        dipole_block = dipole_block.reshape(3 * num_gamma, num_atoms)  # (3*num_gamma, n_atoms)
        quad_block = np.stack([np.stack(v, axis=0) for v in quad_elem], axis=0)
        quad_block = quad_block.reshape(6 * num_gamma, num_atoms)      # (6*num_gamma, n_atoms)

        elem_feats = np.concatenate((radial_block, dipole_block, quad_block), axis=0)
        elem_feature_blocks.append(elem_feats.T)  # transpose → (n_atoms, feat_dim_elem)

    # Concatenate element blocks along *feature* axis --------------------------
    dset = np.concatenate(elem_feature_blocks, axis=1).astype(np.float32)

    # 5. Build local orthogonal basis (flattened 3×3 → 9) ----------------------
    basis_list: List[np.ndarray] = []
    cutoff_basis = 5.0

    for site in structure.sites:
        pos = site.coords
        neighs = sorted(structure.get_neighbors(site, cutoff_basis), key=lambda x: x[1])

        if len(neighs) < 2:
            mat = np.eye(3)
        else:
            v1 = neighs[0][0].coords - pos
            v2 = neighs[1][0].coords - pos
            v1 /= np.linalg.norm(v1)
            u3 = np.cross(v1, v2)
            if np.linalg.norm(u3) == 0:
                # Degenerate neighbours – fallback
                u3 = np.array([0.0, 0.0, 1.0])
            else:
                u3 /= np.linalg.norm(u3)
            u2 = np.cross(u3, v1)
            u2 /= np.linalg.norm(u2)
            mat = np.vstack((v1, u2, u3)).T  # shape (3,3)
        basis_list.append(mat.flatten())

    basis_mat = np.array(basis_list, dtype=np.float32)

    return dset, basis_mat, sites_elem, num_atoms, at_elem
