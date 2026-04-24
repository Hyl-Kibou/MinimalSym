import numpy as np
from dataclasses import dataclass
from numba import njit
from .sym_ops import unique_sorted


@dataclass
class SEA():
    """
    Symmetry equivalent atoms (SEA).

    SEAs are atoms that can be swapped with no distinguishable change in the molecule.

    Parameters
    ----------
    label : str or None, optional
        Rotor type of the SEA set (e.g. Single Atom, Linear, Spherical, Regular Polygon, Oblate Symmetric Top).
    subset : np.ndarray
        Atom indices in the molecule that belong to this SEA set, shape (N,).
    axis : np.ndarray or None, optional
        Candidate rotational symmetry axis, shape (3,).
    """
    label: str
    subset: np.ndarray
    axis: np.ndarray
    def __eq__(self, other):
        if len(self.subset) != len(other.subset):
            return False
        return self.label == other.label and (self.subset == other.subset).all() and (self.axis == other.axis).all()

@njit(cache=True)
def transform(positions: "np.ndarray", M: "np.ndarray") -> "np.ndarray":
    """
    Transform coordinates of molecule by matrix M and return new positions.

    Parameters
    ----------
    positions: np.ndarray
        Molecule positions.
    M: np.ndarray
        Transformation matrix (e.g. rotation, reflection, etc.), shape (3,3)

    Returns
    -------
    ase.Atoms
        Molecule with transformed atom coordinates
    """
    return np.dot(positions, np.transpose(M))


@njit(cache=True)
def distance_matrix(positions):
    """
    Calculate the interatomic distance matrix as all pairwise distances between atoms.

    Parameters
    ----------
    positions: np.ndarray
        Numpy array of positions of Molecule.

    Returns
    -------
    np.ndarray
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

@njit(cache=True)
def distance_matrix_pair(posA, posB):
    diff = posA[:, None, :] - posB[None, :, :]
    return np.sum(diff**2, axis=-1)

@njit(cache=True)
def _jit_find_SEAs(size, positions, geom_tol):
    dm = distance_matrix(positions)
    # Pre-sort each row once to avoid storing ragged argsort arrays
    sorted_dm = np.empty_like(dm)
    for i in range(size):
        sorted_dm[i] = np.sort(dm[i])
    # Union-find: group atoms whose sorted distance profiles agree within geom_tol
    parent = np.arange(size, dtype=np.int64)
    for i in range(size):
        for j in range(i + 1, size):
            chk = True
            for k in range(size):
                if abs(sorted_dm[i, k] - sorted_dm[j, k]) >= geom_tol:
                    chk = False
                    break
            if chk:
                ri = i
                while parent[ri] != ri:
                    ri = parent[ri]
                rj = j
                while parent[rj] != rj:
                    rj = parent[rj]
                if ri != rj:
                    parent[rj] = ri
    # Assign a sequential SEA index to each root
    root_to_sea = -1 * np.ones(size, dtype=np.int64)
    sea_id = np.empty(size, dtype=np.int64)
    next_sea = np.int64(0)
    for i in range(size):
        r = i
        while parent[r] != r:
            r = parent[r]
        if root_to_sea[r] == -1:
            root_to_sea[r] = next_sea
            next_sea += 1
        sea_id[i] = root_to_sea[r]
    return sea_id

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
    sea_id = _jit_find_SEAs(len(mol), mol.positions, mol.info["geom_tol"])
    n_seas = int(sea_id.max()) + 1
    SEAs = []
    for k in range(n_seas):
        subset = np.where(sea_id == k)[0].astype(np.int64)
        SEAs.append(SEA("", subset, np.zeros(3)))
    return SEAs

@njit
def _jit_get_SEAs_from_atom_map(atom_map):
    subset_list = []

    for subset in atom_map:
        ordered_subset = unique_sorted(subset)

        found = False
        for j in range(len(subset_list)):
            if np.array_equal(ordered_subset, subset_list[j]):
                found = True
                break

        if not found:
            subset_list.append(ordered_subset)

    return subset_list


def get_SEAs_from_atom_map(atom_map):
    """
    Find sets of symmetry equivalent atoms based on a Symtext.atom_map.
    Form symmetry equivalent sets from who each atom maps to.

    Parameters
    ----------
    atom_map: np.ndarray
        Symtext.atom_map

    Returns
    -------
    List[SEA]
        List of symmetry equivalent atom sets
    """
    SEAs = []
    subset_list = _jit_get_SEAs_from_atom_map(atom_map)

    for subset in subset_list:
        SEAs.append(SEA("", subset, np.zeros(3)))

    return SEAs

@njit(cache=True)
def _isequivalent(A_masses, A_positions, B_masses, B_positions, geom_tol):
    matched = np.zeros(len(B_masses), dtype=np.bool_)
    tol2 = geom_tol*geom_tol
    for i in range(len(A_masses)):
        for j in range(len(B_masses)):
            # Reduce search list so large molecules are a bit faster
            if not matched[j]:
                # Check that masses are equal
                if A_masses[i] == B_masses[j]:
                    # Check if atoms are about at the same Cartesian point
                    zs = A_positions[i,:]-B_positions[j,:]
                    if (zs[0]*zs[0] + zs[1]*zs[1] + zs[2]*zs[2]) < tol2:
                        matched[j]=True
                        break
    # Did we find a match for each atom? If so we win
    if sum(matched) == len(A_masses):
        return True
    return False

@njit(cache=True)
def transform_isequivalent(positions, masses, geom_tol, matrix):
    positions_B = transform(positions, matrix)
    return _isequivalent(masses, positions, masses, positions_B, geom_tol)

@njit(cache=True)
def _jit_calcmoit(positions, masses):
    I = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                for k in range(len(positions)):
                    I[i,i] += masses[k]*(positions[k,(i+1)%3]**2+positions[k,(i+2)%3]**2)
            else:
                for k in range(len(positions)):
                    I[i,j] -= masses[k]*positions[k,i]*positions[k,j]
    return I

def calcmoit(atoms):
    """
    Calculates the moment of inertia tensor for a list of atoms.

    Parameters
    ----------
    atoms: ase.Atoms
        Set of atoms.

    Returns
    -------
    np.ndarray
        Cartesian moment of inertia tensor, shape(3,3).
    """
    atoms.translate(-atoms.get_center_of_mass())
    masses = atoms.get_masses()
    positions = atoms.positions
    
    return _jit_calcmoit(positions, masses)