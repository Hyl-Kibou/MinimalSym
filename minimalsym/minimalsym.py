"""
minimalsym.py — Public API for molecular symmetry analysis.

Public functions
----------------
  get_point_group(mol, geom_tol)  -> str

  is_planar(mol, tol)             -> bool

  get_inequivalent(mol, geom_tol) -> (unique, parent)

  symmetrize(mol_in, geom_tol)    -> ase.Atoms
"""

import numpy as np
from typing import TYPE_CHECKING

from .symmetrizer.pg_detect import find_point_group, mol_is_planar
from .symmetrizer.symtext import Symtext
from .symmetrizer.mol_ops import find_SEAs
from .symmetrizer.constants import SYMMETRIZED_TOL

if TYPE_CHECKING:
    from ase import Atoms


# ── Input validation ───────────────────────────────────────────────────────────

def _validate_mol(mol, tol, min_atoms=1):
    """
    Validate common inputs for public API functions.

    Parameters
    ----------
    mol : ase.Atoms
    tol : float
        Must be positive.
    min_atoms : int
        Minimum number of atoms required (default 1).

    Raises
    ------
    ImportError  — if ase is not installed.
    TypeError    — if mol is not an ase.Atoms object.
    ValueError   — if tol <= 0 or mol has fewer than min_atoms atoms.
    """
    try:
        from ase import Atoms as _Atoms
    except ImportError:
        raise ImportError("ase is required but not installed.")

    if not isinstance(mol, _Atoms):
        raise TypeError(f"Expected an ase.Atoms object, got {type(mol).__name__}.")
    if tol <= 0:
        raise ValueError(f"Tolerance must be positive, got {tol}.")
    if len(mol) < min_atoms:
        raise ValueError(
            f"Molecule must have at least {min_atoms} atom(s), got {len(mol)}."
        )

def _set_tolerances(mol, geom_tol, eigen_tol):
    mol.info["geom_tol"] = geom_tol
    if eigen_tol is None:
        mol.info["eigen_tol"] = _estimate_eigen_tol(
            mol.positions, mol.get_masses(), geom_tol
        )
    else:
        mol.info["eigen_tol"] = eigen_tol

# ── Estimate a value for eigen tolerance ───────────────────────────────────────

def _estimate_eigen_tol(positions, masses, mol_tol, factor=2.0):
    R2 = np.average(np.sum(positions**2, axis=1), weights=masses)
    R = np.sqrt(R2)
    if R == 0:
        return 1.0  # fallback for degenerate case
    return factor * mol_tol / R

# ── Point group & planarity ────────────────────────────────────────────────────

def get_point_group(mol: "Atoms", geom_tol: float = 0.05, eigen_tol: float|None = None) -> str:
    """
    Determine the point group of a molecule.

    Parameters
    ----------
    mol : ase.Atoms
    geom_tol : float, optional
        Geometric tolerance (default 0.05 Å).
    eigen_tol : float, optional
        Relative tolerance for eigenvalues (default None,
        internal worker will determine an appropriate float).

    Returns
    -------
    str — Schoenflies symbol (e.g. "C2v", "D3h", "Oh").

    Raises
    ------
    TypeError   — mol is not ase.Atoms.
    ValueError  — geom_tol <= 0 or mol is empty.
    RuntimeError — point-group detection failed internally.
    """
    _validate_mol(mol, geom_tol, min_atoms=1)
    mol = mol.copy()
    _set_tolerances(mol, geom_tol, eigen_tol)
    try:
        result = find_point_group(mol)
    except Exception as exc:
        raise RuntimeError(
            f"Point group detection failed for molecule with {len(mol)} atoms "
            f"and tolerance {geom_tol}. Original error: {exc}"
        ) from exc
    return result.pg


def is_planar(mol: "Atoms", geom_tol: float = 0.05) -> bool:
    """
    Check whether all atoms of a molecule lie in a common plane.

    Parameters
    ----------
    mol : ase.Atoms
    geom_tol : float, optional
        Geometric tolerance (default 0.05 Å).

    Returns
    -------
    bool

    Raises
    ------
    TypeError  — mol is not ase.Atoms.
    ValueError — tol <= 0 or mol has fewer than 3 atoms.
    """
    _validate_mol(mol, geom_tol, min_atoms=3)
    mol = mol.copy()
    mol.info["geom_tol"] = geom_tol
    return mol_is_planar(mol)


# ── Atom equivalence (union-find) ─────────────────────────────────────────────

def _get_ancestor(parent, atom_num):
    """
    Path-compression union-find: return (root, path) for *atom_num*.

    Parameters
    ----------
    parent : np.ndarray of int
    atom_num : int

    Returns
    -------
    tuple(int, list[int])  — (root_index, nodes_on_path_to_root)
    """
    ancestor_line = [atom_num]
    while parent[atom_num] != atom_num:
        atom_num = parent[atom_num]
        ancestor_line.append(atom_num)
    return atom_num, ancestor_line


def _union_find_pass(atom_map, parent, symel_indices):
    """
    Merge equivalence classes for all atoms over the given symel indices.

    Parameters
    ----------
    atom_map : np.ndarray, shape (n_atoms, n_symels)
    parent : np.ndarray of int, shape (n_atoms,)  — modified in-place
    symel_indices : iterable of int
    """
    for atom in range(atom_map.shape[0]):
        for s in symel_indices:
            w = atom_map[atom, s]
            owner_w, ancestor_w = _get_ancestor(parent, w)
            owner_atom, ancestor_atom = _get_ancestor(parent, atom)
            for idx in ancestor_w:
                parent[idx] = owner_atom
            for idx in ancestor_atom:
                parent[idx] = owner_atom
                
    # Final path-compression pass to flatten all chains
    for ii in range(len(parent)):
        owner, ancestor = _get_ancestor(parent, ii)
        for idx in ancestor:
            parent[idx] = owner


def get_inequivalent(mol_in: "Atoms", geom_tol: float = 0.3, eigen_tol: float|None = None) -> tuple:
    """
    Find symmetry-inequivalent atoms using all symmetry operations.

    Two atoms are in the same equivalence class if any symmetry operation
    (proper or improper) maps one onto the other.

    Parameters
    ----------
    mol_in : ase.Atoms
    geom_tol : float, optional
        Geometric tolerance (default 0.3 Å)
    eigen_tol : float, optional
        Relative tolerance for eigenvalues (default None,
        internal worker will determine an appropriate float).

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        unique : sorted representative atom indices (one per class).
        parent : parent[i] is the representative of atom i.

    Raises
    ------
    TypeError, ValueError, RuntimeError
    """
    _validate_mol(mol_in, geom_tol, min_atoms=1)
    mol_in = mol_in.copy()
    _set_tolerances(mol_in, geom_tol, eigen_tol)

    try:
        asym_symtext = Symtext.from_molecule(mol_in)
    except Exception as exc:
        raise RuntimeError(f"Symtext construction failed: {exc}") from exc

    atom_map = asym_symtext.atom_map
    parent = np.arange(len(mol_in))
    _union_find_pass(atom_map, parent, range(atom_map.shape[1]))

    map_change_to = {parent[0]: 0}

    for ii, elem in enumerate(parent):
        if elem not in map_change_to.keys():
            map_change_to[elem] = ii

        parent[ii]=map_change_to[parent[ii]]

    return np.unique(parent), parent


# ── Symmetrization (geometric projection) ─────────────────────────────────────

def _project_linear(mol, sea, asym_symtext):
    """
    Project atoms onto the molecular axis for linear groups (C0v / D0h).

    For C0v all atoms are collapsed onto the z-axis.
    For D0h atoms come in inversion-related pairs; the central atom (if any)
    is placed at the origin.

    Parameters
    ----------
    mol : ase.Atoms
        Molecule being symmetrized (modified in-place).
    sea : SEA
        Symmetry-equivalent-atoms group to process.
    asym_symtext : Symtext

    Returns
    -------
    bool
        True if this is C0v (caller should break after the first SEA).
    """
    z = np.array([0.0, 0.0, 1.0])
    atom_i = sea.subset[0]

    if asym_symtext.pg.family == "C":
        # C0v: all atoms lie on z; zero out x and y.
        for atom_j in sea.subset:
            mol.positions[atom_j, :] = np.array([0.0, 0.0, np.dot(mol.positions[atom_j, :], z)])
        return True  # signal to break — single SEA covers the whole molecule

    # Maybe should ask elif asym_symtext.pg.family == "D":
    # D0h: atoms come in inversion-related pairs.
    if atom_i == asym_symtext.atom_map[atom_i, 1]:
        mol.positions[atom_i, :] = np.array([0.0, 0.0, 0.0])
    else:
        mol.positions[atom_i, :] = np.array([0.0, 0.0, np.dot(mol.positions[atom_i, :], z)])

    for atom_j in sea.subset[1:]:
        mol.positions[atom_j, :] = np.array([0.0, 0.0, np.dot(mol.positions[atom_j, :], z)])
        if atom_j == asym_symtext.atom_map[atom_i, 1]:
            # Inversion partner of atom_i: negate its axial position.
            mol.positions[atom_j, :] = np.dot(-np.eye(3), mol.positions[atom_i, :])

    return False


def _project_atom(mol, atom_i, asym_symtext):
    """
    Project the SEA representative atom_i onto the symmetry element that fixes it.

    Searches for a non-identity symel g such that atom_map[atom_i, g] == atom_i,
    then applies the appropriate geometric projection:

    - C_n axis  : keep only the projection along the axis vector.
    - sigma plane : subtract the normal component (project into the plane).
    - i / S_n   : place the atom at the origin.

    Parameters
    ----------
    mol : ase.Atoms
        Molecule being symmetrized (modified in-place).
    atom_i : int
        Index of the SEA representative atom.
    asym_symtext : Symtext

    Raises
    ------
    Exception
        If an unrecognised symmetry element type is encountered.
    """
    for g in range(1, asym_symtext.order):
        if atom_i != asym_symtext.atom_map[atom_i, g]:
            continue
        sym = asym_symtext.symels[g]
        if sym.symbol == "i" or sym.symbol[0] == "S":
            mol.positions[atom_i, :] = np.array([0.0, 0.0, 0.0])
            break
        elif sym.symbol[0] == "C":
            l = np.dot(mol.positions[atom_i, :], sym.vector)
            mol.positions[atom_i, :] = l * sym.vector
        elif sym.symbol[:5] == "sigma":
            l = np.dot(mol.positions[atom_i, :], sym.vector)
            mol.positions[atom_i, :] -= l * sym.vector
        else:
            raise Exception(f"Unexpected symmetry element type: '{sym.symbol}'")


def _map_sea_from_representative(mol, sea, atom_i, asym_symtext):
    """
    Map the remaining SEA atoms from the (already projected) representative.

    For each atom_j in sea.subset[1:], finds the symel g that sends atom_i
    to atom_j and applies its rotation matrix.

    Parameters
    ----------
    mol : ase.Atoms
        Molecule being symmetrized (modified in-place).
    sea : SEA
    atom_i : int
        Index of the (already projected) SEA representative.
    asym_symtext : Symtext
    """
    for atom_j in sea.subset[1:]:
        for g in range(1, asym_symtext.order):
            if atom_j == asym_symtext.atom_map[atom_i, g]:
                mol.positions[atom_j, :] = np.dot(
                    asym_symtext.symels[g].rrep, mol.positions[atom_i, :]
                )
                break

def symmetrize(mol_in: "Atoms", geom_tol: float = 0.05, eigen_tol: float|None = None) -> "Atoms":
    """
    Symmetrize the geometry of a molecule to exact point-group symmetry.

    Algorithm overview
    ------------------
    1. Build a Symtext at tolerance *geom_tol* to detect the point group
       and the atom permutation map.
    2. For each set of symmetry-equivalent atoms (SEA):

       a. **Linear molecules** (C0v / D0h): project every atom onto the
          molecular axis (z). For D0h, inversion partners are handled
          explicitly.

       b. **Non-linear molecules**: find the first non-trivial symmetry element
          that fixes the SEA representative (atom_i):

          - C_n axis: project position onto the axis.
          - sigma plane: project into the plane.
          - i or S_n: place at the origin.

       c. Map the remaining SEA atoms from the representative using the stored
          matrix representation of the connecting symel.

    3. Set mol.info["geom_tol"] = 1e-12 on the result so downstream detection
       sees exact symmetry.

    Parameters
    ----------
    mol_in : ase.Atoms
        Molecule to be symmetrized.
    geom_tol : float, optional
        Tolerance for detecting the initial (possibly distorted) point group.
        Default is 0.05 Å.
    eigen_tol : float, optional
        Relative tolerance for eigenvalues (default None,
        internal worker will determine an appropriate float).

    Returns
    -------
    ase.Atoms
        New Atoms object with symmetrized coordinates.

    Raises
    ------
    RuntimeError
        If Symtext construction fails.
    Exception
        Re-raises unexpected symmetry element types.
    """
    mol_in = mol_in.copy()

    _set_tolerances(mol_in, geom_tol, eigen_tol)

    seas = find_SEAs(mol_in)

    try:
        asym_symtext = Symtext.from_molecule(mol_in)
    except Exception as exc:
        raise RuntimeError(
            f"Symtext construction failed during symmetrize: {exc}"
        ) from exc

    mol = asym_symtext.mol

    for sea in seas:
        atom_i = sea.subset[0]

        if asym_symtext.pg.is_linear:
            done = _project_linear(mol, sea, asym_symtext)
            if done:
                break
            continue

        _project_atom(mol, atom_i, asym_symtext)
        _map_sea_from_representative(mol, sea, atom_i, asym_symtext)

    mol.info["geom_tol"] = SYMMETRIZED_TOL
    return mol