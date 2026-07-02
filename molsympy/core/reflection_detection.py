"""
reflection_detection.py — Search for reflection planes (sigma).

Public names consumed by pg_detect.py:
  _is_there_sigmah, _is_there_sigmav, mol_is_planar, _planar_mol_axis
"""

import numpy as np
from numba import njit
from numba.typed import List
from numba import types

from .sym_ops import reflection_matrix, normalize, issame_axis, get_unique_axes, mean_axis_0_numba
from .mol_ops import transform_isequivalent, transform
from .mol_orient import _jit_rotate_mol_to_symels


# ── Plane detection ───────────────────────────────────────────────────────────

@njit(cache=True)
def _is_there_sigmah(positions, masses, geom_tol, paxis):
    """
    Check for a horizontal reflection plane (normal = paxis).

    Parameters
    ----------
    positions : np.ndarray
    masses : np.ndarray
    geom_tol : float
    paxis : np.ndarray, shape (3,)

    Returns
    -------
    bool
    """
    sigmah = reflection_matrix(paxis)
    return transform_isequivalent(positions, masses, geom_tol, sigmah)

@njit(cache=True)
def _is_there_sigmav(positions, masses, geom_tol, list_sea_subset, paxis):
    """
    Check for vertical reflection planes (normal perpendicular to paxis).

    Parameters
    ----------
    positions : np.ndarray
    masses : np.ndarray
    geom_tol : float
    list_sea_subset: list[np.ndarray]
    paxis : np.ndarray, shape (3,)

    Returns
    -------
    tuple(bool, np.ndarray)
    """
    axes = List.empty_list(types.float64[:])
    for sea in list_sea_subset:
        length = len(sea)
        if length < 2:
            continue
        A = sea[0]
        for i in range(1, length):
            B = sea[i]
            n = normalize(positions[A, :] - positions[B, :])
            if not (n == np.zeros(3)).all():
                sigma = reflection_matrix(n)
                if transform_isequivalent(positions, masses, geom_tol, sigma):
                    axes.append(n)
    if len(axes) < 1:
        if mol_is_planar(positions, geom_tol):
            return True, _planar_mol_axis(positions)
        else:
            return False, np.zeros(3)
    unique_axes = get_unique_axes(axes)
    for i in unique_axes:
        if not issame_axis(i, paxis):
            return True, i
    return False, np.zeros(3)


# ── Planarity helpers ─────────────────────────────────────────────────────────

@njit(cache=True)
def mol_is_planar(positions, geom_tol):
    """
    Check if all atoms lie in a common plane.

    Parameters
    ----------
    positions : ase.Atoms

    Returns
    -------
    bool
    """
    rank = np.linalg.matrix_rank(positions, tol=geom_tol)
    if rank < 3:
        axis = _planar_mol_axis(positions)
        new_positions, _, _ = _jit_rotate_mol_to_symels(positions, axis, np.array([0.0, 0.0, 0.0]))
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        positions_B = transform(new_positions, matrix)
        for i in range(len(positions)):
            if np.linalg.norm(new_positions[i,:] - positions_B[i,:]) >= geom_tol:
                return False
        return True
    return False

@njit(cache=True)
def _planar_mol_axis(positions):
    """
    Return the normal to the plane of a planar molecule.

    Parameters
    ----------
    positions : np.ndarray

    Returns
    -------
    : np.ndarray
        shape (3,)
    """
    coords = positions - mean_axis_0_numba(positions)
    _, _, vh = np.linalg.svd(coords, full_matrices=False)
    return normalize(vh[-1])
