import numpy as np
from dataclasses import dataclass

global_tol = 1e-8 # TODO It would be nice to get rid of this...

@dataclass
class SEA():
    """
    Symmetry equivalent atoms (SEA).

    SEAs are atoms that can be swapped with no distinguishable change in the molecule.

    Parameters
    ----------
    label : str or None, optional
        Optionally defines rotor type of SEA set (e.g. Single Atom, Linear, Spherical, Regular Polygon, Oblate Symmetric Top, etc.)
    subset: np.array
        Sublist of atom indices in molecule that define the SEA set, shape (N,)
    axis: np.array or None, optional
        Optional possible rotational symmetry vector, shape (3,)
    """
    label:str
    subset:np.array
    axis:np.array

def transform(mol, M):
    """
    Transform coordinates of molecule by matrix M and return new molecule.

    Parameters
    ----------
    mol: ase.Atoms
        Molecule object.
    M: np.array
        Transformation matrix (e.g. rotation, reflection, etc.), shape (3,3)

    Returns
    -------
    ase.Atoms
        Molecule with transformed atom coordinates
    """
    new_mol = mol.copy()
    new_mol.positions = np.dot(new_mol.positions, np.transpose(M))
    return new_mol

def distance_matrix(positions):
    """
    Calculate the interatomic distance matrix as all pairwise distances between atoms.

    Parameters
    ----------
    mol: np.array
        Numpy array of positions of Molecule.

    Returns
    -------
    np.array
        Interatomic distance matrix, shape(len(mol),len(mol))
    """
    dm = np.zeros((len(positions), len(positions)))
    for i in range(len(positions)):
        for j in range(i, len(positions)):
            dm[i,j] = np.sqrt(sum((positions[i,:]-positions[j,:])**2))
            dm[j,i] = dm[i,j]
    return dm

def find_SEAs(mol):
    """
    Find sets of symmetry equivalent atoms.
    Permutations of the distance matrix reveal which atoms form symmetry equivalent sets.

    Parameters
    ----------
    mol: ase.Atoms
        Molecule object.

    Returns
    -------
    List[SEA]
        List of symmetry equivalent atom sets
    """

    dm = distance_matrix(mol.positions)
    out = []
    for i in range(len(mol)):
        for j in range(i+1, len(mol)):
            a_idx = np.argsort(dm[i, :])
            b_idx = np.argsort(dm[j, :])
            z = dm[i,a_idx] - dm[j,b_idx]
            chk = True
            for k in z:
                if abs(k) < mol.info["tol"]:
                    continue
                else:
                    chk = False
            if chk:
                out.append((i,j))
    skip = []
    SEAs = []
    for i in range(len(mol)):
        if i in skip:
            continue
        else:
            collect = [i]
        
        for k in out:
            if i in k:
                if i == k[0]:
                    collect.append(k[1])
                    skip.append(k[1])
                else:
                    collect.append(k[0])
                    skip.append(k[0])
        SEAs.append(SEA("", collect, np.zeros(3)))
    return SEAs

