import numpy as np
from dataclasses import dataclass
from numba import njit

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
    def __eq__(self, other):
        if len(self.subset) != len(other.subset):
            return False
        return self.label == self.label and (self.subset == other.subset).all() and (self.axis == other.axis).all()

@njit
def transform(positions: "np.array", M: "np.array") -> "np.array":
    """
    Transform coordinates of molecule by matrix M and return new positions.

    Parameters
    ----------
    positions: np.array
        Molecule positions.
    M: np.array
        Transformation matrix (e.g. rotation, reflection, etc.), shape (3,3)

    Returns
    -------
    ase.Atoms
        Molecule with transformed atom coordinates
    """
    return np.dot(positions, np.transpose(M))

@njit
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
            dx = positions[i,0] - positions[j,0]
            dy = positions[i,1] - positions[j,1]
            dz = positions[i,2] - positions[j,2]
            dm[i,j] = np.sqrt(dx*dx + dy*dy + dz*dz)
            dm[j,i] = dm[i,j]
    return dm

@njit
def _jit_find_SEAs(size, positions, tol):
    dm = distance_matrix(positions)
    out = []
    indexs = []

    for i in range(size):
        indexs.append(np.argsort(dm[i, :]))

    for i in range(size):
        a_idx = indexs[i]
        for j in range(i+1, size):
            b_idx = indexs[j]
            z = dm[i,a_idx] - dm[j,b_idx]
            chk = True
            for k in z:
                if abs(k) >= tol:
                    chk = False
                    break
            if chk:
                out.append((i,j))
    skip = np.zeros(size, dtype=np.bool_)
    SEAs = []
    for i in range(size):
        if skip[i]:
            continue
        collect = [i]

        for k in out:
            if i in k:
                if i == k[0]:
                    collect.append(k[1])
                    skip[k[1]] = True
                else:
                    collect.append(k[0])
                    skip[k[0]] = True
        SEAs.append(collect)
    return SEAs

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

    SEAs_parts = _jit_find_SEAs(len(mol), mol.positions, mol.info['tol'])
    SEAs = []
    for collect in SEAs_parts:
        SEAs.append(SEA("", np.asarray(collect, dtype=np.int64), np.zeros(3)))
    return SEAs

