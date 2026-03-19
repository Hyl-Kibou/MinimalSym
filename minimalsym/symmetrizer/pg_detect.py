"""
pg_detect.py — Point-group detection: public API.

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

from .sym_ops import rotation_matrix, inversion_matrix, Sn, Cn, normalize, reflection_matrix
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

def _classify_linear(mol, positions, masses, mol_tol):
    """Classify a linear molecule (Ia ~= 0). Returns PointGroupResult."""
    paxis = _linear_mol_axis(mol)
    pg = "D0h" if transform_isequivalent(positions, masses, mol_tol, inversion_matrix()) else "C0v"
    return PointGroupResult(pg=pg, paxis=paxis, saxis=np.zeros(3))


def _classify_spherical_top(mol, positions, masses, mol_tol):
    """
    Classify a spherical top (Ia ~= Ib ~= Ic).
    Discriminates T/O/I families by the number of distinct C2 axes.
    Returns PointGroupResult.
    """
    seas = find_SEAs(mol)
    n, axes = _num_C2(mol, seas)
    invertable = transform_isequivalent(positions, masses, mol_tol, inversion_matrix())

    if n == 15:
        # Icosahedral: paxis = C5 axis, saxis from golden-ratio geometry.
        tempaxis = axes[0]
        c3s = _find_C3s_for_Ih(mol)
        saxis = np.zeros(3)
        for c3 in c3s:
            if np.isclose(np.arccos(abs(np.dot(c3, tempaxis))), IH_C2_C3_ANGLE, atol=IH_ANGLE_TOL):
                taxis = normalize(np.cross(c3, tempaxis))
                saxis = normalize(np.cross(taxis, tempaxis))
                break
        phi = (1 + np.sqrt(5.0)) / 2
        theta = np.arccos(phi / np.sqrt(1 + phi**2))
        paxis = np.dot(rotation_matrix(saxis, theta), tempaxis)
        pg = "Ih" if invertable else "I"

    elif n == 9:
        # Octahedral: paxis and saxis are two orthogonal C4 axes.
        c4s = _find_C4s_for_Oh(mol)
        paxis, saxis = c4s[0], c4s[1]
        pg = "Oh" if invertable else "O"

    else:
        # Tetrahedral (n == 3): use two of the three C2 axes.
        paxis, saxis = axes[0], axes[1]
        pg = "Th" if invertable else "Td"

    return PointGroupResult(pg=pg, paxis=paxis, saxis=saxis)


def _classify_subfamily(mol, seas, positions, masses, mol_tol, paxis, Cn_order):
    """
    Determine the point-group subfamily (h/v/d/S2n/pure) once paxis and
    Cn_order are known. Returns the full Schoenflies symbol and updated saxis.
    """
    saxis = [0, 0, 0]
    ortho_c2_chk, c2_ortho = _is_there_ortho_c2(mol, seas, paxis)
    sigmav_chk, sigmav = _is_there_sigmav(mol, seas, paxis)
    sigmah_chk = _is_there_sigmah(mol, paxis)

    if ortho_c2_chk:
        saxis = c2_ortho
        if sigmah_chk:
            pg = "D" + str(Cn_order) + "h"
        elif sigmav_chk:
            pg = "D" + str(Cn_order) + "d"
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
        if transform_isequivalent(positions, masses, mol_tol, S2n):
            pg = "S" + str(2 * Cn_order)
        else:
            pg = "C" + str(Cn_order)

    return pg, saxis


def _classify_general(mol, positions, masses, mol_tol):
    """
    Classify a symmetric top (two equal MOIT eigenvalues).
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
            if transform_isequivalent(positions, masses, mol_tol, inversion_matrix()):
                return PointGroupResult(pg="Ci", paxis=paxis, saxis=np.zeros(3))
            sigmav_chk, sigmav = _is_there_sigmav(mol, seas, np.zeros(3))
            if sigmav_chk:
                if sigmav is not None:
                    paxis = sigmav
                return PointGroupResult(pg="Cs", paxis=paxis, saxis=np.zeros(3))
            return PointGroupResult(pg="C1", paxis=paxis, saxis=np.zeros(3))
        paxis = c2
        Cn_order = 2

    pg, saxis = _classify_subfamily(mol, seas, positions, masses, mol_tol, paxis, Cn_order)
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
       - Symmetric top (Ia ~= Ib or Ib ~= Ic): Cn, Cnv, Cnh, Dn, Dnh, Dnd, Sn.
       - Asymmetric rotor (all distinct): C1, Cs, Ci, C2, C2v, C2h, D2, D2h.
    3. Subfamily (h/v/d) is determined from sigma_h, sigma_v, and ortho-C2.

    Based on:
        Beruski, Otávio; Vidal, Luciano N. J. Comp. Chem. 2013.
        doi:10.1002/jcc.23493

    Parameters
    ----------
    mol : ase.Atoms
        Molecule with mol.info['tol'] set to the geometric tolerance.

    Returns
    -------
    PointGroupResult
        .pg    — Schoenflies symbol (e.g. "C2v", "D3h", "Oh").
        .paxis — principal axis, shape (3,); zeros when not applicable.
        .saxis — secondary axis, shape (3,); zeros when not applicable.

    Raises
    ------
    Exception
        If mol.info['tol'] is not set.
    """
    try:
        mol_tol = mol.info['tol']
    except KeyError:
        raise Exception("Atoms object tolerance hasn't been set. Set it with Atoms.info['tol']=.")

    mol = mol.copy()
    positions = mol.positions
    masses = mol.get_masses()

    # Diagonalize MOIT; eigenvectors become candidate symmetry axes.
    moit = calcmoit(mol)
    evals_mol, evecs_mol = np.linalg.eigh(moit)
    _idx = evals_mol.argsort()
    Ia_mol, Ib_mol, Ic_mol = evals_mol[_idx]

    if np.isclose(Ia_mol, 0.0, atol=mol_tol):
        return _classify_linear(mol, positions, masses, mol_tol)

    elif np.isclose(Ia_mol, Ib_mol, atol=mol_tol) and np.isclose(Ia_mol, Ic_mol, atol=mol_tol):
        return _classify_spherical_top(mol, positions, masses, mol_tol)
        
    return _classify_general(mol, positions, masses, mol_tol)
