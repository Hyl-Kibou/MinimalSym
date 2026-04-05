"""
pg_detect.py — Point-group detection.

This module exposes a single public function, find_point_group(), which
implements the Beruski & Vidal (2013) decision-tree algorithm.

The megafunction is decomposed into three private classifiers:
  _classify_linear()          — Ia ~= 0  (linear molecules)
  _classify_spherical_top()   — Ia ~= Ib ~= Ic  (cubic/icosahedral)
  _classify_symmetric_top()   — two equal MOIT eigenvalues

Internal implementation is split across three focused sub-modules:
  rotation_detection.py   — Cn axis searches
  reflection_detection.py — sigma plane searches
  special_geometry.py     — icosahedral and octahedral geometry
"""

import numpy as np
from dataclasses import dataclass
import warnings

from .sym_ops import rotation_matrix, inversion_matrix, Sn, normalize, inertia_isclose, generate_cyclic_axes, Cn, reflection_matrix
from .mol_ops import calcmoit, transform_isequivalent, find_SEAs
from .constants import IH_C2_C3_ANGLE, IH_ANGLE_TOL

from .rotation_detection import (
    _find_rotation_sets, _find_rotations, _linear_mol_axis,
    _find_a_c2, _is_there_ortho_c2, _num_C2, _highest_order_axis,
)
from .reflection_detection import (
    _is_there_sigmah, _is_there_sigmav, mol_is_planar, _planar_mol_axis,
)
from .special_geometry import _find_C3s_for_Ih, _find_C4s_for_Oh


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class PointGroupResult:
    """
    Return value of find_point_group().

    Attributes
    ----------
    pg : str
        Schoenflies point-group symbol (e.g. "C2v", "D3h", "Oh").
    paxis : np.ndarray, shape (3,)
        Principal symmetry axis in the original molecule frame.
        Zero vector when not applicable (e.g. C1).
    saxis : np.ndarray, shape (3,)
        Secondary axis defining the canonical orientation.
        Zero vector when not applicable.
    """
    pg: str
    paxis: np.ndarray
    saxis: np.ndarray


# ── Private classifier helpers ────────────────────────────────────────────────

def _classify_linear(mol, positions, masses, geom_tol):
    """Classify a linear molecule (Ia ~= 0). Returns PointGroupResult."""
    paxis = _linear_mol_axis(mol)
    pg = "D0h" if transform_isequivalent(positions, masses, geom_tol, inversion_matrix()) else "C0v"
    return PointGroupResult(pg=pg, paxis=paxis, saxis=np.zeros(3))


def _classify_spherical_top(mol, positions, masses, geom_tol):
    """
    Classify a spherical top (Ia ~= Ib ~= Ic).
    Discriminates T/O/I families by the number of distinct C2 axes.
    Returns PointGroupResult.
    """
    seas = find_SEAs(mol)
    num_C2 = _num_C2(mol, seas)
    if num_C2 is None:
        warnings.warn("Molecule was wrongly classified as a spherical top, (num_C2 is None), probably due to high eigen_tol. " \
            "Process will continue as general symmetry.")
        return _classify_general(mol, positions, masses, geom_tol)
    n, axes = num_C2
    invertable = transform_isequivalent(positions, masses, geom_tol, inversion_matrix())

    if n >= 16:
        # Icosahedral: paxis = C5 axis, saxis = C2 axis from golden-ratio geometry.
        try:
            c2_axis = axes[0]
            c3s = _find_C3s_for_Ih(mol)
            saxis = np.zeros(3)
            for c3 in c3s:
                if np.isclose(np.arccos(abs(np.dot(c3, c2_axis))), IH_C2_C3_ANGLE, atol=IH_ANGLE_TOL):
                    taxis = normalize(np.cross(c3, c2_axis))
                    saxis = normalize(np.cross(taxis, c2_axis))
                    break
            phi = (1 + np.sqrt(5.0)) / 2
            theta = np.arccos(phi / np.sqrt(1 + phi**2))
            paxis = np.dot(rotation_matrix(saxis, theta), c2_axis)
            pg = "Ih" if invertable else "I"
        except:
            warnings.warn(f"Molecule was wrongly classified as a spherical top, (n is {n}), probably due to high eigen_tol or geom_tol. " \
                "Process will continue as general symmetry.")
            return _classify_general(mol, positions, masses, geom_tol)
    elif n == 15:
        # Icosahedral: paxis = C5 axis, saxis = C2 axis from golden-ratio geometry.
        c2_axis = axes[0]
        c3s = _find_C3s_for_Ih(mol)
        saxis = np.zeros(3)
        for c3 in c3s:
            if np.isclose(np.arccos(abs(np.dot(c3, c2_axis))), IH_C2_C3_ANGLE, atol=IH_ANGLE_TOL):
                taxis = normalize(np.cross(c3, c2_axis))
                saxis = normalize(np.cross(taxis, c2_axis))
                break
        phi = (1 + np.sqrt(5.0)) / 2
        theta = np.arccos(phi / np.sqrt(1 + phi**2))
        paxis = np.dot(rotation_matrix(saxis, theta), c2_axis)
        pg = "Ih" if invertable else "I"

    elif n == 9:
        # Octahedral: paxis and saxis are two orthogonal C4 axes.
        c4s = _find_C4s_for_Oh(mol)
        paxis, saxis = c4s[0], c4s[1]
        pg = "Oh" if invertable else "O"

    elif n == 3:
        # Tetrahedral (n == 3): use two of the three C2 axes.
        paxis, saxis = axes[0], axes[1]

        # Detect reflection symmetry (any sigma plane)
        sigmav_chk, _ = _is_there_sigmav(mol, seas, paxis)
        sigmah_chk = _is_there_sigmah(mol, paxis)

        # Detect improper rotation S4 (characteristic of Td/Th)
        S4 = Sn(paxis, 4)
        has_S4 = transform_isequivalent(positions, masses, geom_tol, S4)

        if invertable:
            pg = "Th"
        # elif sigmav_chk or sigmah_chk or has_S4:
        elif has_S4: # Must have S4
            pg = "Td"
        else:
            pg = "T"
    else:
        warnings.warn(f"Molecule was wrongly classified as a spherical top, (n is {n}), probably due to high eigen_tol or geom_tol. " \
            "Process will continue as general symmetry.")
        return _classify_general(mol, positions, masses, geom_tol)

    return PointGroupResult(pg=pg, paxis=paxis, saxis=saxis)

def _validate_all_c2_ortho(positions, masses, geom_tol, paxis, c2_ortho, Cn_order):
    """
    Validate the c2 orthogonal rotations for a molecule.
    Generates all the c2 orthogonal axis from c2_ortho,
    generates their rotation matrix and validates it.
    """
    c2_ortho_axes = generate_cyclic_axes(paxis, c2_ortho, Cn_order)
    for c2_ortho_axis in c2_ortho_axes:
        if not transform_isequivalent(positions, masses, geom_tol, Cn(c2_ortho_axis, 2)):
            return False
    return True

def _validate_all_sigmav(positions, masses, geom_tol, paxis, sigmav, Cn_order):
    """
    Validate the vertical mirror planes for a molecule.
    Generates all the norm axis of the mirror planes from sigmav,
    generates their reflection matrix and validates it.
    """
    sigmav_axes = generate_cyclic_axes(paxis, sigmav, Cn_order)
    for sigmav_axis in sigmav_axes:
        if not transform_isequivalent(positions, masses, geom_tol, reflection_matrix(sigmav_axis)):
            return False
    return True

def _classify_subfamily(mol, seas, positions, masses, geom_tol, paxis, Cn_order):
    """
    Determine the point-group subfamily (h/v/d/S2n/pure) once paxis and
    Cn_order are known. Returns the full Schoenflies symbol and updated saxis.
    """
    saxis = np.zeros(3)
    ortho_c2_chk, c2_ortho = _is_there_ortho_c2(mol, seas, paxis)
    sigmav_chk, sigmav = _is_there_sigmav(mol, seas, paxis)
    sigmah_chk = _is_there_sigmah(mol, paxis)

    if ortho_c2_chk:
        ortho_c2_chk = _validate_all_c2_ortho(positions, masses, geom_tol, paxis, c2_ortho, Cn_order)

    if sigmav_chk:
        sigmav_chk = _validate_all_sigmav(positions, masses, geom_tol, paxis, sigmav, Cn_order)

    if ortho_c2_chk:
        saxis = c2_ortho
        if sigmah_chk:
            pg = "D" + str(Cn_order) + "h"
        elif sigmav_chk:
            S2n = Sn(paxis, Cn_order * 2)
            if transform_isequivalent(positions, masses, geom_tol, S2n):
                pg = "D" + str(Cn_order) + "d"
            else:
                pg = "D" + str(Cn_order)
        else:
            pg = "D" + str(Cn_order)
    elif sigmah_chk:
        pg = "C" + str(Cn_order) + "h"
    elif sigmav_chk:
        pg = "C" + str(Cn_order) + "v"
        if mol_is_planar(mol):
            saxis = _planar_mol_axis(mol)
        elif sigmav is not None and hasattr(sigmav, '__len__') and any(sigmav):
            saxis = normalize(np.cross(paxis, sigmav))
    else:
        S2n = Sn(paxis, Cn_order * 2)
        if transform_isequivalent(positions, masses, geom_tol, S2n):
            pg = "S" + str(2 * Cn_order)
        else:
            pg = "C" + str(Cn_order)

    return pg, saxis


def _classify_general(mol, positions, masses, geom_tol):
    """
    Classify a general symmetry.
    Returns PointGroupResult.
    """
    paxis = np.zeros(3)
    seas = find_SEAs(mol)
    rot_set = _find_rotation_sets(mol, seas)
    rots = _find_rotations(mol, rot_set)

    if len(rots) >= 1:
        Cn_order = _highest_order_axis(rots)
        paxis = rots[0].axis
    else:
        c2 = _find_a_c2(mol, seas)
        if c2 is None:
            # No proper rotation -> Ci, Cs, or C1.
            if transform_isequivalent(positions, masses, geom_tol, inversion_matrix()):
                return PointGroupResult(pg="Ci", paxis=paxis, saxis=np.zeros(3))
            sigmav_chk, sigmav = _is_there_sigmav(mol, seas, np.zeros(3))
            if sigmav_chk:
                if sigmav is not None:
                    paxis = sigmav
                return PointGroupResult(pg="Cs", paxis=paxis, saxis=np.zeros(3))
            return PointGroupResult(pg="C1", paxis=paxis, saxis=np.zeros(3))
        paxis = c2
        Cn_order = 2

    pg, saxis = _classify_subfamily(mol, seas, positions, masses, geom_tol, paxis, Cn_order)
    return PointGroupResult(pg=pg, paxis=paxis, saxis=saxis)

# ── Public API ────────────────────────────────────────────────────────────────

def find_point_group(mol):
    """
    Find the point group of a molecule.

    Returns the Schoenflies symbol and the primary/secondary axes that define
    the canonical orientation with respect to the generated symmetry elements.

    Algorithm summary (Beruski & Vidal 2013)
    ----------------------------------------
    1. Compute the moment-of-inertia tensor (MOIT) and its eigenvalues
       Ia <= Ib <= Ic.
    2. Classify the rotor type from the eigenvalues:
       - Linear (Ia ~= 0): C0v or D0h.
       - Spherical top (Ia ~= Ib ~= Ic): T, Td, Th, O, Oh, I, Ih.
       - Symmetric top and Asymmetric rotor: Cn, Cnv, Cnh, Dn, Dnh, Dnd, Sn, C1, Cs, Ci.
    3. Subfamily (h/v/d) is determined from sigma_h, sigma_v, and ortho-C2.

    Based on:
        Beruski, Otávio; Vidal, Luciano N. J. Comp. Chem. 2013.
        doi:10.1002/jcc.23493

    Parameters
    ----------
    mol : ase.Atoms
        Molecule with mol.info["geom_tol"] set to the geometric tolerance.

    Returns
    -------
    : PointGroupResult

    Raises
    ------
    Exception
        If mol.info["geom_tol"] is not set.
    """
    try:
        geom_tol = mol.info["geom_tol"]
    except KeyError:
        raise Exception("Atoms object geometric tolerance hasn't been set. Set it with Atoms.info[\"geom_tol\"]=.")

    try:
        eigen_tol = mol.info['eigen_tol']
    except KeyError:
        raise Exception("Atoms object eigen tolerance hasn't been set. Set it with Atoms.info[\"eigen_tol\"]=.")

    mol = mol.copy()
    positions = mol.positions
    masses = mol.get_masses()

    # Diagonalize MOIT; eigenvectors become candidate symmetry axes.
    moit = calcmoit(mol)
    evals_mol, evecs_mol = np.linalg.eigh(moit)
    _idx = evals_mol.argsort()
    Ia_mol, Ib_mol, Ic_mol = evals_mol[_idx]

    if inertia_isclose(Ia_mol, 0.0, atol=geom_tol, rtol=eigen_tol):
        return _classify_linear(mol, positions, masses, geom_tol)

    elif inertia_isclose(Ia_mol, Ib_mol, atol=geom_tol, rtol=eigen_tol) and inertia_isclose(Ia_mol, Ic_mol, atol=geom_tol, rtol=eigen_tol):
        return _classify_spherical_top(mol, positions, masses, geom_tol)

    return _classify_general(mol, positions, masses, geom_tol)
