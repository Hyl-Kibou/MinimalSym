"""
rotation_detection.py — Search for proper rotation axes (Cn).

Public names consumed by pg_detect.py:
  RotationElement, _find_rotation_sets, _find_rotations,
  _linear_mol_axis, _find_a_c2, _is_there_ortho_c2, _num_C2,
  _highest_order_axis
"""

import numpy as np
from numba.typed import List
from numba import njit, types

from .sym_ops import Cn, Sn, vec_norm_axis, normalize, inertia_isclose, issame_axis, isfactor, get_unique_axes, mean_axis_0_numba
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
        
    def __repr__(self) -> str:
        return f"Axis: {self.axis} Order: {self.order}"


# ── Rotation-set union ─────────────────────────────────────────────────

def _rotation_set_union(rotation_set:RotationElement) -> list[RotationElement]:
    """Return the union of all per-SEA rotation sets."""
    out = rotation_set[0]
    if len(rotation_set) > 1:
        for i in range(1, len(rotation_set)):
            out.extend(rotation_set[i][:])
    out = sort_descending_RotationElements(unique_RotationElements(out))
    return out

# ── Rotation-set unique and sorted elements ────────────────────────────────────────────────────

def unique_RotationElements(elements):
    """Return a list of RotationElement objects with duplicates removed, using __eq__."""
    unique = []
    for el in elements:
        if not any(el == u for u in unique):  # uses __eq__
            unique.append(el)
    return unique

def sort_descending_RotationElements(elements):
    """Return a list of RotationElement objects sorted by
    RotationElement.order in descending order."""
    return sorted(elements, key=lambda x: x.order, reverse=True)

# ── Rotation-set discovery ────────────────────────────────────────────────────

def _find_rotation_sets(mol, SEAs) -> list[list[RotationElement]]:
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
    geom_tol = mol.info["geom_tol"]
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
            if inertia_isclose(Ia, Ib, atol=geom_tol, rtol=eigen_tol) and inertia_isclose(Ia, Ic, atol=geom_tol, rtol=eigen_tol):
                sea.label = "Spherical"
            elif inertia_isclose(Ia + Ib, Ic, atol=geom_tol, rtol=eigen_tol):
                axis = Icv
                sea.axis = axis
                if inertia_isclose(Ia, Ib, atol=geom_tol, rtol=eigen_tol):
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
                if not (inertia_isclose(Ia, Ib, atol=geom_tol, rtol=eigen_tol) or inertia_isclose(Ib, Ic, atol=geom_tol, rtol=eigen_tol)):
                    sea.label = "Asymmetric Rotor"
                    for ax in [Iav, Ibv, Icv]:
                        out_per_SEA.append(RotationElement(ax, 2))
                else:
                    if inertia_isclose(Ia, Ib, atol=geom_tol, rtol=eigen_tol):
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
    geom_tol = mol.info["geom_tol"]
    eigen_tol = mol.info['eigen_tol']
    if len(rotation_set) < 1:
        return []
    molmoit = calcmoit(mol)
    evals = np.sort(np.linalg.eigh(molmoit)[0])
    if evals[0] == 0.0 and inertia_isclose(evals[1], evals[2], atol=geom_tol, rtol=eigen_tol):
        for i in range(np.shape(positions)[0]):
            norm = normalize(positions[i, :])
            if not (norm == np.zeros(3)).all():
                axis = norm
                break
        re = RotationElement(axis, 0)
        return [re]
    rsu = _rotation_set_union(rotation_set)
    for i in rsu:
        rotation_is_valid = validate_cn_subrotations(i.order, i.axis, positions, masses, geom_tol)
        if rotation_is_valid:
            return [i]
    return []

# ── Validate symmetry elements ────────────────────────────────────────────────

@njit(cache=True)
def validate_cn_subrotations(order:int, axis:np.ndarray, positions:np.ndarray, masses:np.ndarray, geom_tol:float) -> bool:
    """
    Validate subrotations

    Parameters
    ----------
    order: int
        Order of rotation
    axis: np.ndarray
        Axis of rotation
    positions: np.ndarray
        Atom positions, shape(n, 3)
    masses: np.ndarray
        Atom masses, shape(n, )
    geom_tol: float
        Geometric tolerance

    Returns
    -------
    : bool
        True if all subrotations and rotation are valid
    """
    for subrotation in range(1, order):
        rmat = Cn(axis, (order/subrotation))
        if not transform_isequivalent(positions, masses, geom_tol, rmat):
            return False

    return True

@njit(cache=True)
def validate_sn_subrotations(order:int, axis:np.ndarray, positions:np.ndarray, masses:np.ndarray, geom_tol:float, has_sigmah:bool=False) -> bool:
    """
    Validate improper subrotations, Sn

    Parameters
    ----------
    order: int
        Order of rotation
    axis: np.ndarray
        Axis of rotation
    positions: np.ndarray
        Atom positions, shape(n, 3)
    masses: np.ndarray
        Atom masses, shape(n, )
    geom_tol: float
        Geometric tolerance
    has_sigmah: bool
        If True improper rotation is done for each rotation in order.
        Else it is only done for reachable states of pure improper rotations.

    Returns
    -------
    : bool
        True if all subrotations and rotation are valid
    """
    if has_sigmah:
        for subrotation in range(1, order):
            rmat = Sn(axis, (order/subrotation))
            if not transform_isequivalent(positions, masses, geom_tol, rmat):
                return False
    else:
        for subrotation in range(1, order * 2, 2):
            subrotation = subrotation % order
            if subrotation == 0:
                continue
            rmat = Sn(axis, (order/subrotation))
            if not transform_isequivalent(positions, masses, geom_tol, rmat):
                return False

    return True

# ── Axis helpers ──────────────────────────────────────────────────────────────

@njit(cache=True)
def _linear_mol_axis(positions):
    """
    Return the axis that best aligns with a linear molecule.

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
    return normalize(vh[0])


def _highest_order_axis(rotations):
    """Return the order of the highest-order rotation in the list."""
    m = rotations[0].order
    for elem_rot in rotations:
        m = max(m, elem_rot.order)
    return m


# ── C2 searches ───────────────────────────────────────────────────────────────

@njit(cache=True)
def _compute_R_max(positions, axis):
    """Return the maximum perpendicular distance of any atom from the given axis."""
    axis = normalize(axis)
    proj = np.dot(positions, axis)[:, np.newaxis] * axis[np.newaxis, :]
    dists = vec_norm_axis(positions - proj)
    return np.max(dists)


@njit(cache=True)
def _validate_c2_candidate(raw_axis, positions, masses, geom_tol, exclude_axis):
    """
    Shared C2 validation kernel: normalize → exclude-filter → invariance test.

    Normalises *raw_axis*, rejects it if it coincides with *exclude_axis*
    (pass ``np.zeros(3)`` to disable the filter), and tests whether the
    corresponding C2 rotation leaves the molecule invariant.

    Parameters
    ----------
    raw_axis : np.ndarray, shape (3,)
        candidate vector (need not be unit)
    exclude_axis : np.ndarray, shape (3,)
        axis to reject (zeros(3) = no filter)

    Returns
    -------
    : np.ndarray, shape (3,)
        Normalised unit axis if the candidate passes all checks; zeros(3) otherwise.
    """
    c2_axis = normalize(raw_axis)
    if (c2_axis == np.zeros(3)).all():
        return np.zeros(3)
    if not (exclude_axis == np.zeros(3)).all() and issame_axis(c2_axis, exclude_axis):
        return np.zeros(3)
    if transform_isequivalent(positions, masses, geom_tol, Cn(c2_axis, 2)):
        return c2_axis
    return np.zeros(3)


@njit(cache=True)
def _c2a(positions, masses, geom_tol, sea_subset,
         exclude_axis=np.zeros(3), max_num_c2=1, ortho_tol = -1):
    """
    Find C_2 axes from origin-to-midpoint vectors of atom pairs within a SEA.

    Candidate = normalised (pos[i] + pos[j]).
    """
    length = len(sea_subset)
    out = List.empty_list(types.float64[:])
    for ii in range(length):
        for jj in range(ii + 1, length):
            origin_midpoint = normalize(positions[sea_subset[ii], :] + positions[sea_subset[jj], :])
            if ortho_tol >= 0 and abs(np.dot(origin_midpoint, exclude_axis)) > ortho_tol:
                continue
            result = _validate_c2_candidate(origin_midpoint, positions, masses, geom_tol, exclude_axis)
            if not (result == np.zeros(3)).all():
                out.append(result)
                if len(out) >= max_num_c2:
                    out = get_unique_axes(out)
                    if len(out) >= max_num_c2:
                        return out
    return out


@njit(cache=True)
def _c2b(positions, masses, geom_tol, sea_subset,
         exclude_axis=np.zeros(3), max_num_c2=1, ortho_tol = -1):
    """
    Find C_2 axes from individual atom position vectors within a SEA.

    Candidate = normalised pos[i].
    """
    length = len(sea_subset)
    out = List.empty_list(types.float64[:])
    for ii in range(length):
        from_origin = normalize(positions[sea_subset[ii]])
        if ortho_tol >= 0 and abs(np.dot(from_origin, exclude_axis)) > ortho_tol:
            continue
        result = _validate_c2_candidate(
            from_origin, positions, masses, geom_tol, exclude_axis
        )
        if not (result == np.zeros(3)).all():
            out.append(result)
            if len(out) >= max_num_c2:
                out = get_unique_axes(out)
                if len(out) >= max_num_c2:
                    return out
    return out


@njit(cache=True)
def _c2c(positions, masses, geom_tol, sea1_subset, sea2_subset,
         exclude_axis=np.zeros(3), ortho_tol = -1):
    """
    Find a C_2 axis from the cross-product of two linear-SEA bond vectors.

    Candidate = normalised cross(r_SEA1, r_SEA2).
    Returns the unit axis or zeros(3) if the candidate is rejected.
    """
    rij = positions[sea1_subset[0], :] - positions[sea1_subset[1], :]
    rkl = positions[sea2_subset[0], :] - positions[sea2_subset[1], :]
    cross_prod = normalize(np.cross(rij, rkl))
    if ortho_tol >= 0 and abs(np.dot(cross_prod, exclude_axis)) > ortho_tol:
        return np.zeros(3)
    return _validate_c2_candidate(
        cross_prod, positions, masses, geom_tol, exclude_axis
    )

@njit(cache=True)
def _find_a_c2(positions: np.ndarray, masses: np.ndarray, geom_tol: float, list_sea_subset: list[np.ndarray]) -> np.ndarray:
    """
    Search for any C_2 axis; return the first one found.

    Parameters
    ----------
    positions: np.ndarray
    masses: np.ndarray
    geom_tol: float
    list_sea_subset: list[np.ndarray]

    Returns
    -------
    np.ndarray, shape (3,)
    """
    num_seas = len(list_sea_subset)
    for id_sea1 in range(num_seas):
        sea1 = list_sea_subset[id_sea1]
        a = _c2a(positions, masses, geom_tol, sea1)
        if len(a) != 0:
            return a[0]
        b = _c2b(positions, masses, geom_tol, sea1)
        if len(b) != 0:
            return b[0]
        if len(sea1) == 2:
            for id_sea2 in range(id_sea1, num_seas):
                sea2 = list_sea_subset[id_sea2]
                if len(sea2) == 2:
                    c = _c2c(positions, masses, geom_tol, sea1, sea2)
                    if not (c == np.zeros(3)).all():
                        return c
    return np.zeros(3)


@njit(cache=True)
def _is_there_ortho_c2(positions, masses, geom_tol, list_sea_subset, paxis):
    """
    Search for a C_2 axis orthogonal to paxis; return the first one found.

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
    ortho_tol = geom_tol / _compute_R_max(positions, paxis) * 1.10
    paxis = np.ascontiguousarray(paxis)
    positions = np.ascontiguousarray(positions)
    num_seas = len(list_sea_subset)
    for id_sea1 in range(num_seas):
        sea1 = list_sea_subset[id_sea1]
        b = _c2b(positions, masses, geom_tol, sea1, exclude_axis=paxis, ortho_tol=ortho_tol)
        if len(b) != 0:
            b = b[0]
            return True, b
        else:
            a = _c2a(positions, masses, geom_tol, sea1, exclude_axis=paxis, ortho_tol=ortho_tol)
            if len(a) != 0:
                a = a[0]
                return True, a
            else:
                if len(sea1) == 2:
                    for id_sea2 in range(id_sea1, num_seas):
                        sea2 = list_sea_subset[id_sea2]
                        if len(sea2) == 2:
                            c = _c2c(positions, masses, geom_tol, sea1, sea2,
                                     exclude_axis=paxis, ortho_tol=ortho_tol)
                            if not (c == np.zeros(3)).all():
                                return True, c
    return False, np.zeros(3)

@njit(cache=True)
def _num_C2(positions: np.ndarray, masses: np.ndarray, geom_tol: float, list_sea_subset: list[np.ndarray], max_num_c2: int) -> list[np.ndarray]:
    """
    Count distinct C_2 axes and return them.

    Parameters
    ----------
    positions: np.ndarray
    masses: np.ndarray
    geom_tol: float
    list_sea_subset: list[np.ndarray]
    max_num_c2: int

    Returns
    -------
    list[np.ndarray]
    """
    axes = List.empty_list(types.float64[:])
    for subset in list_sea_subset:
        a = _c2a(positions, masses, geom_tol, subset, max_num_c2=max_num_c2)
        if len(a) != 0:
            axes.extend(a)
        b = _c2b(positions, masses, geom_tol, subset, max_num_c2=max_num_c2)
        if len(b) != 0:
            axes.extend(b)
        if len(axes) > max_num_c2:
            axes = get_unique_axes(axes)
            if len(axes) >= max_num_c2:
                break
    if len(axes) < 1:
        return axes
    unique_axes = get_unique_axes(axes)
    return unique_axes