"""
atom_mapping.py — Atom permutation map builders.

Provides _get_atom_mapping() and _get_linear_atom_mapping(), which build the
(n_atoms × n_symels) integer array that records where each atom goes under
each symmetry operation.
"""

import numpy as np
from numba import njit

from .symel import Symel


# ── JIT kernels ───────────────────────────────────────────────────────────────

@njit
def _jit_where_you_go(positions, mol_tol, atom, rrep):
    """Return the index of the atom that *atom* maps to under *rrep*."""
    ratom = np.dot(rrep, positions[atom, :].T)
    tol2 = mol_tol*mol_tol
    for i in range(len(positions)):
        dist = positions[i,:] - ratom
        if (dist[0]*dist[0] + dist[1]*dist[1] + dist[2]*dist[2]) < tol2:
            return i
    return np.int64(-1)


@njit
def _jit_get_atom_mapping(positions, mol_tol, rreps):
    """Build the full (n_atoms × n_symels) permutation map."""
    natoms = len(positions)
    nsymels = rreps.shape[0]
    amap = np.empty((natoms, nsymels), dtype=np.int64)
    for i in range(natoms):
        for j in range(nsymels):
            amap[i, j] = np.int64(-1)
    for atom in range(natoms):
        for s in range(nsymels):
            amap[atom, s] = _jit_where_you_go(positions, mol_tol, atom, rreps[s])
    return amap


# ── Public builders ───────────────────────────────────────────────────────────

def _where_you_go(mol, atom, symel):
    """
    Find the atom index that *atom* maps to under *symel*.

    Parameters
    ----------
    mol : ase.Atoms
    atom : int
    symel : Symel

    Returns
    -------
    int or None
    """
    w = _jit_where_you_go(mol.positions, mol.info["geom_tol"], atom, symel.rrep)
    return w if w != -1 else None


def _get_atom_mapping(mol, symels):
    """
    Build the (n_atoms × n_symels) atom permutation map for non-linear groups.

    Parameters
    ----------
    mol : ase.Atoms
    symels : List[Symel]

    Returns
    -------
    np.ndarray, shape (n_atoms, n_symels)

    Raises
    ------
    Exception
        If any atom fails to map under a symmetry operation.
    """
    rreps = np.array([s.rrep for s in symels])
    amap = _jit_get_atom_mapping(mol.positions, mol.info["geom_tol"], rreps)
    failed = np.argwhere(amap == -1)
    if len(failed) > 0:
        atom, s = int(failed[0, 0]), int(failed[0, 1])
        raise Exception(
            f"Atom {atom} not mapped to another atom "
            f"under symel {symels[s]}\nPositions: {mol.positions}"
        )
    return amap


def _get_linear_atom_mapping(mol, pg):
    """
    Build the permutation map for linear point groups (C0v / D0h).

    For C0v: identity only (each atom maps to itself).
    For D0h: identity + inversion.

    Parameters
    ----------
    mol : ase.Atoms
    pg : PointGroup

    Returns
    -------
    np.ndarray
    """
    natoms = len(mol)
    amap = np.arange(natoms, dtype=int).reshape((natoms, 1))
    if pg.family == "D":
        ungerade_map = np.zeros(natoms, dtype=int)
        inversion_symel = Symel("i", None, -np.eye(3), None, None, None)
        for atom in range(natoms):
            w = _where_you_go(mol, atom, inversion_symel)
            if w is not None:
                ungerade_map[atom] = w
            else:
                raise Exception(f"Atom {atom} not mapped to another atom under symel i")
        return np.column_stack((amap, ungerade_map))
    return amap
