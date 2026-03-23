"""
rotation_detection.py — Search for proper rotation axes (Cn).

Public names consumed by pg_detect.py:
  RotationElement, _find_rotation_sets, _find_rotations,
  _linear_mol_axis, _find_a_c2, _is_there_ortho_c2, _num_C2,
  _highest_order_axis
"""

import numpy as np
from numba import njit

from .sym_ops import Cn, vec_norm_axis, normalize, inertia_isclose, issame_axis, isfactor
from .mol_ops import calcmoit, transform_isequivalent


# ── Data structure ────────────────────────────────────────────────────────────

class RotationElement():
    """Data structure holding a candidate rotation axis and its order."""

    def __init__(self, axis, order):
        self.axis = axis
        self.order = order

    def __eq__(self, other):
        if isinstance(other, RotationElement):
            return issame_axis(self.axis, other.axis) and self.order == other.order


# ── Rotation-set intersection ─────────────────────────────────────────────────

@njit
def _jit_intersect(a_axis, a_rots, b_axis, b_rots):
    a_size = len(a_axis)
    indexes = np.zeros(a_size, dtype=np.bool_)
    b_size = len(b_axis)
    for ii in range(a_size):
        for jj in range(b_size):
            if issame_axis(a_axis[ii], b_axis[jj]) and a_rots[ii] == b_rots[jj]:
                indexes[ii] = True
                break
    return indexes


def _intersect(a, b):
    """Return elements present in both lists (equality by axis and order)."""

    if len(a) == 0 or len(b) == 0:
        return []

    a_axis = np.asarray([elem.axis for elem in a], dtype=np.float64)
    b_axis = np.asarray([elem.axis for elem in b], dtype=np.float64)
    a_rots = np.asarray([elem.order for elem in a], dtype=np.int64)
    b_rots = np.asarray([elem.order for elem in b], dtype=np.int64)
    indexes = _jit_intersect(a_axis, a_rots, b_axis, b_rots)
    return [elem for ii, elem in enumerate(a) if indexes[ii]]


def _rotation_set_intersection(rotation_set):
    """Return the intersection of all per-SEA rotation sets."""
    out = rotation_set[0]
    if len(rotation_set) > 1:
        for i in range(len(rotation_set)):
            out = _intersect(out, rotation_set[i])
            if len(out) == 0:
                break
    return out


# ── Rotation-set discovery ────────────────────────────────────────────────────

def _find_rotation_sets(mol, SEAs):
    """
    For each SEA, find the set of possible RotationElements.

    Parameters
    ----------
    mol : ase.Atoms
    SEAs : List[SEA]

    Returns
    -------
    List[List[RotationElement]]
    """
    mol_tol = mol.info["geom_tol"]
    eigen_tol = mol.info['eigen_tol']
    out_all_SEAs = []
    for sea in SEAs:
        length = len(sea.subset)
        out_per_SEA = []
        if length < 2:
            sea.label = "Single Atom"
        elif length == 2:
            sea.label = "Linear"
            sea.axis = normalize(mol[sea.subset[0]].position)
        else:
            sea_mol = mol[sea.subset]
            sea_mol.translate(-sea_mol.get_center_of_mass())
            evals, evecs = np.linalg.eigh(calcmoit(mol[sea.subset]))
            idx = evals.argsort()
            Ia, Ib, Ic = evals[idx]
            Iav, Ibv, Icv = [evecs[:, i] for i in idx]
            if inertia_isclose(Ia, Ib, atol=mol_tol, rtol=eigen_tol) and inertia_isclose(Ia, Ic, atol=mol_tol, rtol=eigen_tol):
                sea.label = "Spherical"
            elif inertia_isclose(Ia + Ib, Ic, atol=mol_tol, rtol=eigen_tol):
                axis = Icv
                sea.axis = axis
                if inertia_isclose(Ia, Ib, atol=mol_tol, rtol=eigen_tol):
                    sea.label = "Regular Polygon"
                    for i in range(2, length + 1):
                        if isfactor(length, i):
                            out_per_SEA.append(RotationElement(axis, i))
                else:
                    sea.label = "Irregular Polygon"
                    for i in range(2, length):
                        if isfactor(length, i):
                            out_per_SEA.append(RotationElement(axis, i))
            else:
                if not (inertia_isclose(Ia, Ib, atol=mol_tol, rtol=eigen_tol) or inertia_isclose(Ib, Ic, atol=mol_tol, rtol=eigen_tol)):
                    sea.label = "Asymmetric Rotor"
                    for ax in [Iav, Ibv, Icv]:
                        out_per_SEA.append(RotationElement(ax, 2))
                else:
                    if inertia_isclose(Ia, Ib, atol=mol_tol, rtol=eigen_tol):
                        sea.label = "Oblate Symmetric Top"
                        axis = Icv
                        sea.axis = Icv
                    else:
                        sea.label = "Prolate Symmetric Top"
                        axis = Iav
                        sea.axis = Iav
                    k = length // 2
                    for i in range(2, k + 1):
                        if isfactor(k, i):
                            out_per_SEA.append(RotationElement(axis, i))
            if len(out_per_SEA) > 0:
                out_all_SEAs.append(out_per_SEA)
    return out_all_SEAs


def _find_rotations(mol, rotation_set):
    """
    Find RotationElements in rotation_set that leave the molecule invariant.

    Parameters
    ----------
    mol : ase.Atoms
    rotation_set : List[List[RotationElement]]

    Returns
    -------
    List[RotationElement]
    """
    positions = mol.positions
    masses = mol.get_masses()
    mol_tol = mol.info["geom_tol"]
    eigen_tol = mol.info['eigen_tol']
    if len(rotation_set) < 1:
        return []
    molmoit = calcmoit(mol)
    evals = np.sort(np.linalg.eigh(molmoit)[0])
    if evals[0] == 0.0 and inertia_isclose(evals[1], evals[2], atol=mol_tol, rtol=eigen_tol):
        for i in range(np.shape(positions)[0]):
            if normalize(positions[i, :]) is not None:
                axis = normalize(positions[0, :])
        re = RotationElement(axis, 0)
        return [re]
    rsi = _rotation_set_intersection(rotation_set)
    out = []
    for i in rsi:
        rmat = Cn(i.axis, i.order)
        if transform_isequivalent(positions, masses, mol_tol, rmat):
            out.append(i)
    return out


# ── Axis helpers ──────────────────────────────────────────────────────────────

def _linear_mol_axis(mol):
    """
    Return the axis that best aligns with a linear molecule.

    Parameters
    ----------
    mol : ase.Atoms

    Returns
    -------
    np.array, shape (3,)
    """
    coords = mol.positions - mol.positions.mean(axis=0)
    _, _, vh = np.linalg.svd(coords, full_matrices=False)
    return normalize(vh[0])


def _highest_order_axis(rotations):
    """Return the order of the highest-order rotation in the list."""
    m = rotations[0].order
    for elem_rot in rotations:
        m = max(m, elem_rot.order)
    return m


# ── C2 searches ───────────────────────────────────────────────────────────────

@njit
def _compute_R_max(positions, axis):
    """Return the maximum perpendicular distance of any atom from the given axis."""
    axis = normalize(axis)
    proj = np.dot(positions, axis)[:, np.newaxis] * axis[np.newaxis, :]
    dists = vec_norm_axis(positions - proj)
    return np.max(dists)


@njit
def _validate_c2_candidate(raw_axis, positions, masses, mol_tol, exclude_axis):
    """
    Shared C2 validation kernel: normalize → exclude-filter → invariance test.

    Normalises *raw_axis*, rejects it if it coincides with *exclude_axis*
    (pass ``np.zeros(3)`` to disable the filter), and tests whether the
    corresponding C2 rotation leaves the molecule invariant.

    Parameters
    ----------
    raw_axis : np.ndarray, shape (3,)   — candidate vector (need not be unit)
    exclude_axis : np.ndarray, shape (3,) — axis to reject (zeros(3) = no filter)

    Returns
    -------
    np.ndarray, shape (3,)
        Normalised unit axis if the candidate passes all checks; zeros(3) otherwise.
    """
    c2_axis = normalize(raw_axis)
    if c2_axis is None or (c2_axis == np.zeros(3)).all():
        return np.zeros(3)
    if not (exclude_axis == np.zeros(3)).all() and issame_axis(c2_axis, exclude_axis):
        return np.zeros(3)
    if transform_isequivalent(positions, masses, mol_tol, Cn(c2_axis, 2)):
        return c2_axis
    return np.zeros(3)


@njit
def _c2a(positions, masses, mol_tol, sea_subset,
         exclude_axis=np.zeros(3), return_all=False):
    """
    Find C_2 axes from origin-to-midpoint vectors of atom pairs within a SEA.

    Candidate = normalised (pos[i] + pos[j]).
    """
    length = len(sea_subset)
    out = []
    for i in range(length):
        for j in range(i + 1, length):
            raw = positions[sea_subset[i], :] + positions[sea_subset[j], :]
            result = _validate_c2_candidate(raw, positions, masses, mol_tol, exclude_axis)
            if not (result == np.zeros(3)).all():
                if return_all:
                    out.append(result)
                else:
                    return [result]
    return out


@njit
def _c2b(positions, masses, mol_tol, sea_subset,
         exclude_axis=np.zeros(3), return_all=False):
    """
    Find C_2 axes from individual atom position vectors within a SEA.

    Candidate = normalised pos[i].
    """
    length = len(sea_subset)
    out = []
    for i in range(length):
        result = _validate_c2_candidate(
            positions[sea_subset[i], :], positions, masses, mol_tol, exclude_axis
        )
        if not (result == np.zeros(3)).all():
            if return_all:
                out.append(result)
            else:
                return [result]
    return out


@njit
def _c2c(positions, masses, mol_tol, sea1_subset, sea2_subset,
         exclude_axis=np.zeros(3)):
    """
    Find a C_2 axis from the cross-product of two linear-SEA bond vectors.

    Candidate = normalised cross(r_SEA1, r_SEA2).
    Returns the unit axis or zeros(3) if the candidate is rejected.
    """
    rij = positions[sea1_subset[0], :] - positions[sea1_subset[1], :]
    rkl = positions[sea2_subset[0], :] - positions[sea2_subset[1], :]
    return _validate_c2_candidate(
        np.cross(rij, rkl), positions, masses, mol_tol, exclude_axis
    )


def _find_a_c2(mol, SEAs):
    """
    Search for any C_2 axis; return the first one found.

    Parameters
    ----------
    mol : ase.Atoms
    SEAs : List[SEA]

    Returns
    -------
    np.array, shape (3,) or None
    """
    positions = mol.positions
    masses = mol.get_masses()
    mol_tol = mol.info["geom_tol"]
    for sea in SEAs:
        a = _c2a(positions, masses, mol_tol, sea.subset)
        if len(a) != 0:
            return a[0]
        b = _c2b(positions, masses, mol_tol, sea.subset)
        if len(b) != 0:
            return b[0]
        if sea.label == "Linear":
            for sea2 in SEAs:
                if sea == sea2:
                    continue
                elif sea2.label == "Linear":
                    c = _c2c(positions, masses, mol_tol, sea.subset, sea2.subset)
                    if not (c == np.zeros(3)).all():
                        return c
    return None



def _is_there_ortho_c2(mol, SEAs, paxis):
    """
    Search for a C_2 axis orthogonal to paxis; return the first one found.

    Parameters
    ----------
    mol : ase.Atoms
    SEAs : List[SEA]
    paxis : np.array, shape (3,)

    Returns
    -------
    tuple(bool, np.array or None)
    """
    ortho_tol = mol.info["geom_tol"] / _compute_R_max(mol.positions, paxis) * 1.10
    positions = mol.positions
    masses = mol.get_masses()
    mol_tol = mol.info["geom_tol"]
    for sea in SEAs:
        b = _c2b(positions, masses, mol_tol, sea.subset, exclude_axis=paxis)
        if len(b) != 0:
            b = b[0]
            if abs(np.dot(b, paxis)) <= ortho_tol:
                return True, b
        else:
            a = _c2a(positions, masses, mol_tol, sea.subset, exclude_axis=paxis)
            if len(a) != 0:
                a = a[0]
                if abs(np.dot(a, paxis)) <= ortho_tol:
                    return True, a
            else:
                if sea.label == "Linear":
                    for sea2 in SEAs:
                        if sea == sea2:
                            continue
                        elif sea2.label == "Linear":
                            c = _c2c(positions, masses, mol_tol, sea.subset, sea2.subset,
                                     exclude_axis=paxis)
                            if not (c == np.zeros(3)).all() and abs(np.dot(c, paxis)) <= ortho_tol:
                                return True, c
    return False, None


def _num_C2(mol, SEAs):
    """
    Count distinct C_2 axes and return them.

    Parameters
    ----------
    mol : ase.Atoms
    SEAs : List[SEA]

    Returns
    -------
    tuple(int, List[np.array]) or None
    """
    axes = []
    positions = mol.positions
    masses = mol.get_masses()
    mol_tol = mol.info["geom_tol"]
    for sea in SEAs:
        a = _c2a(positions, masses, mol_tol, sea.subset, return_all=True)
        if len(a) != 0:
            axes.extend(a)
        b = _c2b(positions, masses, mol_tol, sea.subset, return_all=True)
        if len(b) != 0:
            axes.extend(b)
    if len(axes) < 1:
        return None
    unique_axes = [axes[0]]
    for i in axes:
        check = True
        for j in unique_axes:
            if issame_axis(i, j):
                check = False
                break
        if check:
            unique_axes.append(i)
    return len(unique_axes), unique_axes
