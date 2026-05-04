"""
special_geometry.py — Special-geometry axis detection for icosahedral (Ih/I)
and octahedral (Oh/O) point groups.

Public names consumed by pg_detect.py:
  _find_C3s_for_Ih, _find_C4s_for_Oh
"""

import numpy as np
from numba.typed import List
from numba import njit, types

from .sym_ops import Cn, normalize, float_isclose, get_unique_axes
from .mol_ops import transform_isequivalent
from .constants import PRINT_WARNINGS


# ── Icosahedral geometry ──────────────────────────────────────────────────────

@njit(cache=True)
def _jit_find_C3s_for_Ih(size, positions, masses, geom_tol):
    """
    Find the 10 unique C3 axes of an icosahedral (Ih/I) molecule.

    Geometric basis
    ---------------
    The icosahedron has 20 triangular faces; each face defines one C3 axis
    (through its centroid), but opposite faces share an axis, giving 10
    distinct axes. Each C3 axis passes through a pair of antipodal triangular
    face-centers.

    Strategy
    --------
    Enumerate all triples (i, j, k) of atoms. A triple forms an equilateral
    triangle when all three pairwise squared distances are equal within geom_tol.
    The C3 axis candidate is the normal to the plane of the triangle.
    The candidate is accepted only if the full C3 rotation leaves the molecule
    invariant.

    Complexity: O(n^3) atom triples.

    Parameters
    ----------
    size : int
    positions : np.ndarray, shape (n, 3)
    masses : np.ndarray, shape (n,)
    geom_tol : float

    Returns
    -------
    list[np.ndarray], shape (3,)
        exactly 10 unique unit C3-axis vectors.

    Raises
    ------
    Exception
        If the number of unique axes found is not 10.
    """
    c3_axes = List.empty_list(types.float64[:])
    for i in range(size):
        for j in range(i + 1, size):
            for k in range(j + 1, size):
                rij = positions[i, :] - positions[j, :]
                rjk = positions[j, :] - positions[k, :]
                rik = positions[i, :] - positions[k, :]
                nij2 = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]
                njk2 = rjk[0]*rjk[0] + rjk[1]*rjk[1] + rjk[2]*rjk[2]
                nik2 = rik[0]*rik[0] + rik[1]*rik[1] + rik[2]*rik[2]
                if float_isclose(nij2, njk2, atol=geom_tol) and float_isclose(nij2, nik2, atol=geom_tol):
                    c3_axis = normalize(np.cross(rij, rjk))
                    if not (c3_axis == np.zeros(3)).all():
                        c3 = Cn(c3_axis, 3)
                        if transform_isequivalent(positions, masses, geom_tol, c3):
                            c3_axes.append(c3_axis)
    if len(c3_axes) >= 1:
        return get_unique_axes(c3_axes)
    else:
        return List.empty_list(types.float64[:])


def _find_C3s_for_Ih(mol, seas):
    """
    Find the 10 unique C3 axes for an Ih/I molecule so paxis and saxis can be defined.

    Parameters
    ----------
    mol : ase.Atoms
    seas: list[SEA]

    Returns
    -------
    List[np.ndarray], shape (3,)
    """

    c3_axes = List.empty_list(types.float64[:])
    for sea in seas:
        c3_axes.extend(_jit_find_C3s_for_Ih(len(sea.subset), mol.positions[sea.subset], mol.get_masses()[sea.subset], mol.info["geom_tol"]))
        c3_axes = get_unique_axes(c3_axes)
        chk = len(c3_axes)
        if chk == 10:
            return c3_axes
    if PRINT_WARNINGS:
        print("DEBUG: C3 axes count:", chk)
    raise RuntimeError(
        "Unexpected number of C3 axes for Ih point group, expected 10."
    )


# ── Octahedral geometry ───────────────────────────────────────────────────────

@njit(cache=True)
def _check_square(va, vb, a, b, c, d, positions, masses, geom_tol):
    """
    Test whether four edge lengths form a square and, if so, return the C4 axis.

    A square has four equal sides (a==b==c==d within geom_tol). The C4 axis
    is the normal to the plane of the square: cross(edge1, edge2).

    Parameters
    ----------
    va, vb : np.ndarray, shape (3,)
        Two adjacent edge vectors of the candidate quadrilateral.
    a, b, c, d : float
        Squared lengths of the four sides to compare.
    positions, masses : np.ndarray
    geom_tol : float

    Returns
    -------
    np.ndarray, shape (3,)
        Unit C4-axis if valid square and rotation leaves molecule invariant;
        otherwise a zero vector.
    """
    if (float_isclose(a, b, atol=geom_tol) and
            float_isclose(c, d, atol=geom_tol) and
            float_isclose(a, c, atol=geom_tol)):
        c4_axis = normalize(np.cross(va, vb))
        if not (c4_axis == np.zeros(3)).all():
            c4 = Cn(c4_axis, 4)
            if transform_isequivalent(positions, masses, geom_tol, c4):
                return c4_axis
    return np.zeros(3)


@njit(cache=True)
def _jit_find_C4s_for_Oh(size, positions, masses, geom_tol):
    """
    Find the 3 unique C4 axes of an octahedral (Oh/O) molecule.

    Geometric basis
    ---------------
    The octahedron has 6 vertices in 3 orthogonal pairs; each pair defines one
    C4 axis, giving 3 perpendicular axes.

    Strategy
    --------
    Enumerate all quadruples (i, j, k, l). Each is tested in three cyclic
    orderings to check whether any forms a square. Complexity: O(n^4).

    Parameters
    ----------
    size : int
    positions : np.ndarray, shape (n, 3)
    masses : np.ndarray, shape (n,)
    geom_tol : float

    Returns
    -------
    list of np.ndarray, shape (3,)
        exactly 3 unique unit C4-axis vectors.

    Raises
    ------
    Exception
        If the number of unique axes found is not 3.
    """
    c4_axes = List.empty_list(types.float64[:])
    for i in range(size):
        for j in range(i + 1, size):
            for k in range(j + 1, size):
                for l in range(k + 1, size):
                    if i != j and k != l and i != k:
                        rij = positions[i, :] - positions[j, :]
                        rjk = positions[j, :] - positions[k, :]
                        rkl = positions[k, :] - positions[l, :]
                        ril = positions[i, :] - positions[l, :]
                        rik = positions[i, :] - positions[k, :]
                        rjl = positions[j, :] - positions[l, :]

                        nij2 = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]
                        njk2 = rjk[0]*rjk[0] + rjk[1]*rjk[1] + rjk[2]*rjk[2]
                        nkl2 = rkl[0]*rkl[0] + rkl[1]*rkl[1] + rkl[2]*rkl[2]
                        nil2 = ril[0]*ril[0] + ril[1]*ril[1] + ril[2]*ril[2]
                        nik2 = rik[0]*rik[0] + rik[1]*rik[1] + rik[2]*rik[2]
                        njl2 = rjl[0]*rjl[0] + rjl[1]*rjl[1] + rjl[2]*rjl[2]

                        c4_axis = _check_square(rij, rjk, nij2, njk2, nkl2, nil2,
                                                positions, masses, geom_tol)
                        if not (c4_axis == np.zeros(3)).all():
                            c4_axes.append(c4_axis)
                        c4_axis = _check_square(rij, rjl, nij2, njl2, nkl2, nik2,
                                                positions, masses, geom_tol)
                        if not (c4_axis == np.zeros(3)).all():
                            c4_axes.append(c4_axis)
                        c4_axis = _check_square(rik, rjk, nik2, njk2, njl2, nil2,
                                                positions, masses, geom_tol)
                        if not (c4_axis == np.zeros(3)).all():
                            c4_axes.append(c4_axis)
    if len(c4_axes) >= 1:
        return get_unique_axes(c4_axes)
    else:
        return List.empty_list(types.float64[:])


def _find_C4s_for_Oh(mol, seas):
    """
    Find the 3 C4 axes for an Oh/O molecule so paxis and saxis can be defined.

    Parameters
    ----------
    mol : ase.Atoms
    seas : list[SEA]

    Returns
    -------
    List[np.ndarray], shape (3,)
    """
    c4_axes = List.empty_list(types.float64[:])
    for sea in seas:
        c4_axes.extend(_jit_find_C4s_for_Oh(len(sea.subset), mol.positions[sea.subset], mol.get_masses()[sea.subset], mol.info["geom_tol"]))
        c4_axes = get_unique_axes(c4_axes)
        chk = len(c4_axes)
        if chk == 3:
            return c4_axes
    if PRINT_WARNINGS:
        print("DEBUG: c4 axes count", chk)
    raise RuntimeError(
            "Unexpected number of C4 axes for Oh point group, expected 3."
        )