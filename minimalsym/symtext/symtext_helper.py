from .point_group import PointGroup
from ..symtools import normalize
from .multiplication_table import *
import re
from .general_irrep_mats import Symel
    
def rotate_mol_to_symels(mol, paxis, saxis):
    """
    Rotate molecule with symmetry defined by paxis and saxis to symmetry elements.
    paxis -> z axis and saxis -> x axis.

    :type mol: molsym.Molecule
    :type paxis: NumPy array of shape (3,)
    :type saxis: NumPy array of shape (3,)
    :return: New rotated molecule, rotation matrix, inverse rotation matrix
    :rtype: (molsym.Molecule, NumPy array of shape (3,3), NumPy array of shape (3,3))
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
    new_mol = mol.transform(rmat_inv)
    return new_mol, rmat, rmat_inv

def get_atom_mapping(mol, symels):
    """
    Map of each atom under each symmetry element.

    :type mol: molsym.Molecule
    :type symels: List[molsym.Symel]
    :return: Atom by Symel array
    :rtype: NumPy array of shape (natom, nsymel)
    """
    # symels after transformation
    amap = np.zeros((mol.natoms, len(symels)), dtype=int)
    for atom in range(mol.natoms):
        for (s, symel) in enumerate(symels):
            w = where_you_go(mol, atom, symel)
            if w is not None:
                amap[atom,s] = w
            else:
                raise Exception(f"Atom {atom} not mapped to another atom under symel {symel}")
    return amap

def get_linear_atom_mapping(mol, pg):
    """
    Atom map for linear point groups. Still under development.
    """
    amap = np.array([atom for atom in range(mol.natoms)], dtype=int).reshape((mol.natoms,1))
    if pg.family == "D":
        ungerade_map = np.zeros((mol.natoms), dtype=int)
        for atom in range(mol.natoms):
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

    :type mol: molsym.Molecule
    :type atom: int
    :type symel: molsym.Symel
    :rtype: int
    """
    ratom = np.dot(symel.rrep, mol.coords[atom,:].T)
    for i in range(mol.natoms):
        if np.isclose(mol.coords[i,:], ratom, atol=mol.tol).all():
            return i
    return None

def get_class_name(symels_in_class):
    """
    Get the name of the class that a set of Symels belong to.

    :type symels_in_class: List[molsym.Symel]
    :rtype: str
    """
    if "^" in symels_in_class[0].symbol:
        rot_order = []
        for symel in symels_in_class:
            s = re.search(r"\^(\d+)", symel.symbol)
            if s:
                rot_order.append(int(s.groups()[0]))
            else:
                rot_order.append(1)
        pickem = symels_in_class[np.argmin(rot_order)].symbol
    else:
        pickem = symels_in_class[0].symbol
    return re.sub(r"\(\w+\)", "", pickem)
