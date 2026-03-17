import numpy as np
from ..molecule import transform, global_tol
from ..symtools import normalize, vec_isclose, vec_norm
from .symel import Symel
from numba import njit

@njit
def _jit_rotate_mol_to_symels(positions, paxis, saxis):
    if vec_isclose(vec_norm(paxis), 0.0, atol=global_tol):
        # Symmetry is C1 and paxis not defined, just return mol
        rmat = rmat_inv = np.eye(3)
        return positions, rmat, rmat_inv
    z = paxis
    if vec_isclose(vec_norm(saxis), 0.0, atol=global_tol):
        # Find a trial vector that works
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
    rmat = np.column_stack((x, y, z)) # This matrix rotates z to paxis, etc., ...
    rmat_inv = rmat.T # ... so invert it to take paxis to z, etc.
    new_positions = transform(positions, rmat_inv)
    return new_positions, rmat, rmat_inv
    
def rotate_mol_to_symels(mol, paxis, saxis):
    """
    Rotate molecule so that paxis aligns with z and saxis aligns with x.

    Returns the rotated molecule and the forward/inverse rotation matrices so
    that computed properties can be rotated back to the original orientation.

    Parameters
    ----------
    mol: ase.Atoms
    paxis: np.array of shape (3,)
    saxis: np.array of shape (3,)

    Returns
    -------
    tuple(ase.Atoms, np.array, np.array)
        Rotated molecule, rotation matrix, inverse rotation matrix; matrices shape (3,3).
    """

    new_mol = mol.copy()
    paxis = np.asarray(paxis, dtype=np.float64)
    saxis = np.asarray(saxis, dtype=np.float64)
    positions, rmat, rmat_inv = _jit_rotate_mol_to_symels(mol.positions, paxis, saxis)
    new_mol.positions = positions
    return new_mol, rmat, rmat_inv

def get_atom_mapping(mol, symels):
    """
    Build the (natom × nsymel) atom permutation map.

    Parameters
    ----------
    mol: ase.Atoms
    symels: List[Symel]

    Returns
    -------
    np.array of shape (natom, nsymel)
    """
    # symels after transformation
    amap = np.zeros((len(mol), len(symels)), dtype=int)
    for atom in range(len(mol)):
        for s, symel in enumerate(symels):
            w = _where_you_go(mol, atom, symel)
            if w is not None:
                amap[atom, s] = w
            else:
                raise Exception(f"Atom {atom} {mol.info["num"]} not mapped to another atom under symel {symel}\nPositions: {mol.positions}")
    return amap

def get_linear_atom_mapping(mol, pg):
    """
    Atom map for linear point groups. Still under development.
    """
    amap = np.array([atom for atom in range(len(mol))], dtype=int).reshape((len(mol), 1))
    if pg.family == "D":
        ungerade_map = np.zeros((len(mol)), dtype=int)
        for atom in range(len(mol)):
            w = _where_you_go(mol, atom, Symel("i", None, -1*np.eye(3), None, None, None))
            if w is not None:
                ungerade_map[atom] = w
            else:
                raise Exception(f"Atom {atom} not mapped to another atom under symel i")
        return np.column_stack((amap, ungerade_map))
    return amap

@njit
def _jit_where_you_go(positions, mol_tol, atom, rrep):
    ratom = np.dot(rrep, positions[atom,:].T)
    tol2 = mol_tol*mol_tol
    for i in range(len(positions)):
        dist = positions[i,:] - ratom
        if (dist[0]*dist[0] + dist[1]*dist[1] + dist[2]*dist[2]) < tol2:
        #if np.isclose(positions[i,:], ratom, atol=mol_tol).all():
            return i
    return None

def _where_you_go(mol, atom, symel):
    """
    Find the atom index that atom maps to under symel.

    Parameters
    ----------
    mol: ase.Atoms
    atom: int
    symel: Symel

    Returns
    -------
    int or None
    """
    return _jit_where_you_go(mol.positions, mol.info['tol'], atom, symel.rrep)