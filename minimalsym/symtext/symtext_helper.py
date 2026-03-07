import numpy as np
from ..molecule import transform, global_tol
from ..symtools import normalize
from .symel import Symel
    
def rotate_mol_to_symels(mol, paxis, saxis):
    """
    Rotate molecule with symmetry defined by paxis and saxis to symmetry elements.
    paxis -> z axis and saxis -> x axis.

    :type mol: Atoms object from ASE
    :type paxis: NumPy array of shape (3,)
    :type saxis: NumPy array of shape (3,)
    :return: New rotated molecule, rotation matrix, inverse rotation matrix
    :rtype: (Atoms object from ASE, NumPy array of shape (3,3), NumPy array of shape (3,3))
    """
    if np.isclose(np.linalg.norm(paxis), 0.0, atol=global_tol): 
        # Symmetry is C1 and paxis not defined, just return mol
        rmat = rmat_inv = np.eye(3)
        return mol, rmat, rmat_inv
    z = paxis
    if np.isclose(np.linalg.norm(saxis), 0.0, atol=global_tol): 
        # Find a trial vector that works
        x = None
        for trial_vec in np.eye(3):
            x = np.cross(trial_vec, z)
            if not np.isclose(np.linalg.norm(x), 0, atol=global_tol):
                x = normalize(x)
                break
        y = normalize(np.cross(z, x))
    else:
        x = saxis
        y = np.cross(z,x)
    rmat = np.column_stack((x,y,z)) # This matrix rotates z to paxis, etc., ...
    rmat_inv = rmat.T # ... so invert it to take paxis to z, etc.
    new_mol = transform(mol, rmat_inv)
    return new_mol, rmat, rmat_inv

def get_atom_mapping(mol, symels):
    """
    Map of each atom under each symmetry element.

    :type mol: Atoms object from ASE
    :type symels: List[molsym.Symel]
    :return: Atom by Symel array
    :rtype: NumPy array of shape (natom, nsymel)
    """
    # symels after transformation
    amap = np.zeros((len(mol), len(symels)), dtype=int)
    for atom in range(len(mol)):
        for (s, symel) in enumerate(symels):
            w = where_you_go(mol, atom, symel)
            if w is not None:
                amap[atom,s] = w
            else:
                raise Exception(f"Atom {atom} {mol.info["num"]} not mapped to another atom under symel {symel}\nPositions: {mol.positions}")
    return amap

def get_linear_atom_mapping(mol, pg):
    """
    Atom map for linear point groups. Still under development.
    """
    amap = np.array([atom for atom in range(len(mol))], dtype=int).reshape((len(mol),1))
    if pg.family == "D":
        ungerade_map = np.zeros((len(mol)), dtype=int)
        for atom in range(len(mol)):
            w = where_you_go(mol, atom, Symel("i", None, -1*np.eye(3), None, None, None))
            if w is not None:
                ungerade_map[atom] = w
            else:
                raise Exception(f"Atom {atom} not mapped to another atom under symel i")
        return np.column_stack((amap, ungerade_map))
    return amap

def where_you_go(mol, atom, symel):
    """
    Find the resulting atom after applying a symmetry operation

    :type mol: Atoms object from ASE
    :type atom: int
    :type symel: molsym.Symel
    :rtype: int
    """
    ratom = np.dot(symel.rrep, mol.positions[atom,:].T)
    for i in range(len(mol)):
        if np.isclose(mol.positions[i,:], ratom, atol=mol.info["tol"]).all():
            return i
    return None