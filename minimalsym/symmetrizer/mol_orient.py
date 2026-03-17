"""
mol_orient.py — Molecular orientation utilities.

Provides rotate_mol_to_symels(), which aligns paxis with z and saxis with x
so that the molecule sits in the canonical frame expected by pg_to_symels().
"""

import numpy as np
from numba import njit

from .sym_ops import vec_norm, normalize, vec_isclose
from .mol_ops import transform
from .constants import NUMERICAL_TOL as global_tol


@njit
def _jit_rotate_mol_to_symels(positions, paxis, saxis):
    if vec_isclose(vec_norm(paxis), 0.0, atol=global_tol):
        rmat = rmat_inv = np.eye(3)
        return positions, rmat, rmat_inv
    z = paxis
    if vec_isclose(vec_norm(saxis), 0.0, atol=global_tol):
        x = np.zeros(3)
        for trial_vec in np.eye(3):
            x = np.cross(trial_vec, z)
            if not vec_isclose(vec_norm(x), 0.0, atol=global_tol):
                x = normalize(x)
                break
        y = normalize(np.cross(z, x))
    else:
        x = saxis
        y = np.cross(z, x)
    rmat = np.column_stack((x, y, z))
    rmat_inv = rmat.T
    new_positions = transform(positions, rmat_inv)
    return new_positions, rmat, rmat_inv


def rotate_mol_to_symels(mol, paxis, saxis):
    """
    Rotate molecule so that paxis aligns with z and saxis aligns with x.

    Returns the rotated molecule and the forward/inverse rotation matrices so
    that computed properties can be rotated back to the original orientation.

    Parameters
    ----------
    mol : ase.Atoms
    paxis : np.array, shape (3,)
    saxis : np.array, shape (3,)

    Returns
    -------
    tuple(ase.Atoms, np.ndarray, np.ndarray)
        Rotated molecule, rotation matrix, inverse rotation matrix (shape 3×3).
    """
    new_mol = mol.copy()
    paxis = np.asarray(paxis, dtype=np.float64)
    saxis = np.asarray(saxis, dtype=np.float64)
    positions, rmat, rmat_inv = _jit_rotate_mol_to_symels(mol.positions, paxis, saxis)
    new_mol.positions = positions
    return new_mol, rmat, rmat_inv
