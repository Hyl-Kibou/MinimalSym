"""
reflection_detection.py — Search for reflection planes (sigma).

Public names consumed by pg_detect.py:
  _is_there_sigmah, _is_there_sigmav, mol_is_planar, _planar_mol_axis
"""

import numpy as np

from .sym_ops import reflection_matrix, normalize, issame_axis
from .mol_ops import transform_isequivalent, transform
from .mol_orient import rotate_mol_to_symels


# ── Plane detection ───────────────────────────────────────────────────────────

def _is_there_sigmah(mol, paxis):
    """
    Check for a horizontal reflection plane (normal = paxis).

    Parameters
    ----------
    mol : ase.Atoms
    paxis : np.ndarray, shape (3,)

    Returns
    -------
    bool
    """
    sigmah = reflection_matrix(paxis)
    return transform_isequivalent(mol.positions, mol.get_masses(), mol.info["geom_tol"], sigmah)


def _is_there_sigmav(mol, SEAs, paxis):
    """
    Check for vertical reflection planes (normal perpendicular to paxis).

    Parameters
    ----------
    mol : ase.Atoms
    SEAs : List[SEA]
    paxis : np.ndarray, shape (3,)

    Returns
    -------
    tuple(bool, np.ndarray or None)
    """
    axes = []
    positions = mol.positions
    masses = mol.get_masses()
    geom_tol = mol.info["geom_tol"]
    for sea in SEAs:
        length = len(sea.subset)
        if length < 2:
            continue
        A = sea.subset[0]
        for i in range(1, length):
            B = sea.subset[i]
            n = normalize(positions[A, :] - positions[B, :])
            if n is not None and not (n == np.zeros(3)).all():
                sigma = reflection_matrix(n)
                if transform_isequivalent(positions, masses, geom_tol, sigma):
                    axes.append(n)
    if len(axes) < 1:
        if mol_is_planar(mol):
            return True, _planar_mol_axis(mol)
        else:
            return False, None
    unique_axes = [axes[0]]
    for i in axes:
        check = True
        for j in unique_axes:
            if issame_axis(i, j):
                check = False
                break
        if check:
            unique_axes.append(i)
    for i in unique_axes:
        if not issame_axis(i, paxis):
            return True, i
    return False, None


# ── Planarity helpers ─────────────────────────────────────────────────────────

def mol_is_planar(mol):
    """
    Check if all atoms lie in a common plane.

    Parameters
    ----------
    mol : ase.Atoms

    Returns
    -------
    bool
    """
    geom_tol = mol.info["geom_tol"]
    rank = np.linalg.matrix_rank(mol.positions, tol=geom_tol)
    if rank < 3:
        axis = _planar_mol_axis(mol)
        new_mol, _, _ = rotate_mol_to_symels(mol, axis, np.array([0, 0, 0]))
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        new_positions = new_mol.positions
        positions_B = transform(new_positions, matrix)
        for i in range(len(mol)):
            if np.linalg.norm(new_positions[i,:] - positions_B[i,:]) >= geom_tol:
                return False
        return True
    return False


def _planar_mol_axis(mol):
    """
    Return the normal to the plane of a planar molecule.

    Parameters
    ----------
    mol : ase.Atoms

    Returns
    -------
    : np.ndarray
        shape (3,) or None
    """
    coords = mol.positions - mol.positions.mean(axis=0)
    _, _, vh = np.linalg.svd(coords, full_matrices=False)
    return normalize(vh[-1])
