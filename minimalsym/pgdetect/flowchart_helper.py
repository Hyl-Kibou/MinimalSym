import numpy as np
from ..symtools import reflection_matrix, Cn, transform_isequivalent, calcmoit, vec_norm_axis, normalize, vec_isclose, issame_axis, isfactor
from ..symtext.symtext_helper import rotate_mol_to_symels
from ..molecule import transform
from numba import njit, typed
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List
    from ..molecule import SEA
    from ase import Atoms

class RotationElement():
    """
    Data structure holding rotation axes and orders.
    """
    def __init__(self, axis, order):
        self.axis = axis
        self.order = order

    def __eq__(self, other):
        if isinstance(other, RotationElement):
            return issame_axis(self.axis, other.axis) and self.order == other.order
@njit
def _jit_intersect(a_axis, a_rots, b_axis, b_rots):
    a_size = len(a_axis)
    indexes = np.zeros(a_size, dtype=np.bool_)
    b_size = len(b_axis)
    for ii in range(a_size):
        for jj in range(b_size):
            if issame_axis(a_axis[ii], b_axis[jj]) and a_rots[ii] == b_rots[jj]:
                indexes[ii] = True
                break
    return indexes

def _intersect(a: "List[RotationElement]", b: "List[RotationElement]"):
    """
    Find intersection of sets of RotationElements.

    Parameters
    ----------
    a: List[RotationElement]      
    b: List[RotationElement]        

    Returns
    -------
    List[RotationElement]
        Intersection between the RotationElements
    """
    if len(a) == 0 or len(b) == 0:
        return []

    a_axis = [elem.axis for elem in a]
    a_axis = np.asarray(a_axis, dtype=np.float64)
    b_axis = [elem.axis for elem in b]
    b_axis = np.asarray(b_axis, dtype=np.float64)

    a_rots = [elem.order for elem in a]
    a_rots = np.asarray(a_rots, dtype=np.int64)
    b_rots = [elem.order for elem in b]
    b_rots = np.asarray(b_rots, dtype=np.int64)    

    indexes = _jit_intersect(a_axis, a_rots, b_axis, b_rots)

    return [elem for ii, elem in enumerate(a) if indexes[ii]]

def _rotation_set_intersection(rotation_set: "List[List[RotationElement]]"):
    """Return the intersection of all per-SEA rotation sets."""
    out = rotation_set[0]
    if len(rotation_set) > 1:
        for i in range(len(rotation_set)):
            out = _intersect(out, rotation_set[i])
            if len(out) == 0:
                break      
    return out

def find_rotation_sets(mol: "Atoms", SEAs: "List[SEA]"):
    """
    For each set of symmetry equivalent atoms, find the set of possible RotationElements

    Parameters
    ----------
    mol: ase.Atoms
    SEAs: List[SEA]

    Returns
    -------
    List[List[RotationElement]]
    """
    mol_tol = mol.info['tol']
    out_all_SEAs = []
    for sea in SEAs:
        length = len(sea.subset)
        out_per_SEA = []
        if length < 2:
            sea.label = "Single Atom"
        elif length == 2:
            sea.label = "Linear"
            sea.axis = normalize(mol[sea.subset[0]].position)
        else:
            sea_mol = mol[sea.subset]
            sea_mol.translate(-sea_mol.get_center_of_mass())
            evals, evecs = np.linalg.eigh(calcmoit(mol[sea.subset]))
            idx = evals.argsort()
            Ia, Ib, Ic = evals[idx]
            Iav, Ibv, Icv = [evecs[:,i] for i in idx]
            if np.isclose(Ia, Ib, atol=mol_tol) and np.isclose(Ia, Ic, atol=mol_tol):
                sea.label = "Spherical"
            elif np.isclose(Ia+Ib, Ic, atol=mol_tol):
                axis = Icv
                sea.axis = axis
                if np.isclose(Ia, Ib, atol=mol_tol):
                    sea.label = "Regular Polygon"
                    for i in range(2, length+1):
                        if isfactor(length, i):
                            re = RotationElement(axis, i)
                            out_per_SEA.append(re)
                else:
                    sea.label = "Irregular Polygon"
                    for i in range(2, length):
                        if isfactor(length, i):
                            re = RotationElement(axis, i)
                            out_per_SEA.append(re)
            else:
                if not (np.isclose(Ia, Ib, atol=mol_tol) or np.isclose(Ib, Ic, atol=mol_tol)):
                    sea.label = "Asymmetric Rotor"
                    for ax in [Iav, Ibv, Icv]:
                        re = RotationElement(ax, 2)
                        out_per_SEA.append(re)
                else:
                    if np.isclose(Ia, Ib, atol=mol_tol):
                        sea.label = "Oblate Symmetric Top"
                        axis = Icv
                        sea.axis = Icv
                    else:
                        sea.label = "Prolate Symmetric Top"
                        axis = Iav
                        sea.axis = Iav
                    k = length//2
                    for i in range(2, k+1):
                        if isfactor(k, i):
                            re = RotationElement(axis, i)
                            out_per_SEA.append(re)
            if len(out_per_SEA) > 0:
                out_all_SEAs.append(out_per_SEA)
    return out_all_SEAs

def find_rotations(mol: "Atoms", rotation_set: "List[List[RotationElement]]"):
    """
    Find the RotationElements in rotation_set that leave the molecule indistinguishable.

    Parameters
    ----------
    mol: ase.Atoms
    rotation_set: List[List[RotationElement]]

    Returns
    -------
    List[RotationElement]
    """
    positions = mol.positions
    masses = mol.get_masses()
    mol_tol = mol.info['tol']
    if len(rotation_set) < 1:
        return []
    molmoit = calcmoit(mol)
    evals = np.sort(np.linalg.eigh(molmoit)[0])
    if evals[0] == 0.0 and np.isclose(evals[1], evals[2], atol=mol_tol):
        for i in range(np.shape(positions)[0]):
            if normalize(positions[i,:]) is not None:
                axis = normalize(positions[0,:])
        re = RotationElement(axis, 0)
        return [re]
    rsi = _rotation_set_intersection(rotation_set)
    out = []
    for i in rsi:
        rmat = Cn(i.axis, i.order)
        if transform_isequivalent(positions, masses, mol_tol, rmat):
            out.append(i)
    return out

def linear_mol_axis(mol: "Atoms"):
    """
    Returns the axis that best aligns with a linear molecule.

    Parameters
    ----------
    mol: ase.Atoms

    Returns
    -------
    np.array
        NumPy array of shape (3,) or None

    """
    coords = mol.positions - mol.positions.mean(axis=0)
    _, _, vh = np.linalg.svd(coords, full_matrices=False)
    axis = vh[0]
    axis = normalize(axis)
    return axis

def find_a_c2(mol: "Atoms", SEAs: "List[SEA]"):
    """
    Search for any possible C_2 rotation axes, return the first one found.

    Parameters
    ----------
    mol: ase.Atoms
    SEAs: List[SEA]

    Returns
    -------
    np.array
        NumPy array of shape (3,)
    """
    positions = mol.positions
    masses = mol.get_masses()
    mol_tol = mol.info['tol']
    for sea in SEAs:
        a = _c2a(positions, masses, mol_tol, sea.subset)
        if len(a) != 0:
            return a[0]
        else:
            b = _c2b(positions, masses, mol_tol, sea.subset)
            if len(b) != 0:
                return b[0]
            else:
                if sea.label == "Linear":
                    for sea2 in SEAs:
                        if sea == sea2:
                            continue
                        elif sea2.label == "Linear":
                            c = _c2c(positions, masses, mol_tol, sea.subset, sea2.subset)
                            if not (c == np.zeros(3)).all():
                                return c
    return None

@njit
def _compute_R_max(positions: "np.array", axis: "np.array"):
    """
    Compute the maximum distance of any atom from a given axis.

    Parameters
    ----------
    positions: np.array
        Molecule's positions shape (N,3)
    axis: np.array
        NumPy array of shape (3,), must be normalized

    Returns
    -------
    float
        maximum perpendicular distance
    """
    axis = normalize(axis)
    # projection along axis
    proj = np.dot(positions, axis)[:, np.newaxis] * axis[np.newaxis, :]
    # perpendicular component
    perp = positions - proj
    # distance from axis
    dists = vec_norm_axis(perp)
    # maximum distance
    R_max = np.max(dists)
    return R_max

def is_there_ortho_c2(mol: "Atoms", SEAs: "List[SEA]", paxis: "np.array"):
    """
    Search for any possible C_2 rotation axes that are orthogonal to paxis, return the first one found.

    Parameters
    ----------
    mol: ase.Atoms
    SEAs: List[SEA]
    paxis: np.array
        NumPy array of shape (3,)

    Returns
    -------
    tuple(bool, np.array) or None
        True if found and new C_2 axis of shape (3,)
    """
    ortho_tol = mol.info["tol"] / _compute_R_max(mol.positions, paxis) * 1.10
    positions = mol.positions
    masses = mol.get_masses()
    mol_tol = mol.info['tol']
    for sea in SEAs:
        b = _c2b(positions, masses, mol_tol, sea.subset, axis=paxis)
        if len(b) != 0:
            b = b[0]
            if abs(np.dot(b, paxis)) <= ortho_tol:
                return True, b
        else:
            a = _c2a(positions, masses, mol_tol, sea.subset, axis=paxis)
            if len(a) != 0:
                a = a[0]
                if abs(np.dot(a, paxis)) <= ortho_tol:
                    return True, a
            else:
                if sea.label == "Linear":
                    for sea2 in SEAs:
                        if sea == sea2:
                            continue
                        elif sea2.label == "Linear":
                            c = _c2c(positions, masses, mol_tol, sea.subset, sea2.subset, axis=paxis)
                            if not (c == np.zeros(3)).all() and abs(np.dot(c, paxis)) <= ortho_tol:
                                return True, c
    return False, None

def num_C2(mol: "Atoms", SEAs: "List[SEA]"):
    """
    Find the number of C_2 axis present and the axes defining them.
    
    Parameters
    ----------
    mol: ase.Atoms
    SEAs: List[SEA]

    Returns
    -------
    tuple(int, List[np.array])
        Number of C_2 axes and the C_2 axes of shape (3,)
    """
    axes = []
    positions = mol.positions
    masses = mol.get_masses()
    mol_tol = mol.info['tol']
    for sea in SEAs:
        a = _c2a(positions, masses, mol_tol, sea.subset, all=True)
        if len(a) != 0:
            axes.extend(a)
        b = _c2b(positions, masses, mol_tol, sea.subset, all=True)
        if len(b) != 0:
            axes.extend(b)
    if len(axes) < 1:
        return None
    unique_axes = [axes[0]]
    for i in axes:
        check = True
        for j in unique_axes:
            if issame_axis(i,j):
                check = False
                break
        if check:
            unique_axes.append(i)
    return len(unique_axes), unique_axes

@njit
def _c2a(positions: "np.array", masses: "np.array", mol_tol: "float", sea_subset: "np.array", axis=np.zeros(3), all: bool=False):
    """
    Find C_2 axes by testing vectors formed from the origin and midpoint of all pairs of symmetry equivalent atoms.

    Parameters
    ----------
    positions: np.array
        Positions of molecule
    masses: np.array
        Array of masses of molecule
    mol_tol: float
        Tolerance set for the molecule
    sea_subset: np.array
        SEA.subset
    axis: None or np.array, optional
        If not None, only search for C_2 axes that are not equivalent to axis. Array of shape (3,)
    all: bool, optional
        If False return first C_2 found, if True search all atom pairs for C_2 axes

    Returns
    -------
    np.array or List[np.array]
        C_2 axis or list of C_2 axes (if all=True) of shape (3,)
    """
    length = len(sea_subset)
    out = []
    tol2 = mol_tol*mol_tol
    for i in range(length):
        for j in range(i+1, length):
            midpoint = positions[sea_subset[i],:] + positions[sea_subset[j],:]
            if (midpoint[0]*midpoint[0] + midpoint[1]*midpoint[1] + midpoint[2]*midpoint[2]) < tol2:
            #if (abs(midpoint[0]) < mol_tol and
                #abs(midpoint[1]) < mol_tol and
                #abs(midpoint[2]) < mol_tol):
                continue
            else:
                midpoint = normalize(midpoint)
                if (axis is not None and not (axis == np.zeros(3)).all() ) and issame_axis(midpoint, axis) or midpoint is None or (midpoint == np.zeros(3)).all():
                    continue
                c2 = Cn(midpoint, 2)
                if transform_isequivalent(positions, masses, mol_tol, c2):
                    if all:
                        out.append(midpoint)
                    else:
                        return [midpoint]
    return out

@njit
def _c2b(positions: "np.array", masses: "np.array", mol_tol: "float", sea_subset: "np.array", axis=None, all: bool=False):
    """
    Find C_2 axes by testing vectors which pass through individual symmetry equivalent atoms.

    Parameters
    ----------
    positions: np.array
        Positions of molecule
    masses: np.array
        Array of masses of molecule
    mol_tol: float
        Tolerance set for the molecule
    sea_subset: np.array
        SEA.subset
    axis: None or np.array, optional
        If not None, only search for C_2 axes that are not equivalent to axis. Array of shape (3,)
    all: bool, optional
        If False return first C_2 found, if True search all atom pairs for C_2 axes

    Returns
    -------
    np.array or List[np.array]
        C_2 axis or list of C_2 axes (if all=True) of shape (3,)
    """
    length = len(sea_subset)
    out = []
    for i in range(length):
        c2_axis = normalize(positions[sea_subset[i],:])
        if c2_axis is None or (c2_axis == np.zeros(3)).all():
            continue
        if axis is not None and issame_axis(c2_axis, axis):
            continue
        c2 = Cn(c2_axis, 2)
        if transform_isequivalent(positions, masses, mol_tol, c2):
            if all:
                out.append(c2_axis)
            else:
                return [c2_axis]
    return out

@njit
def _c2c(positions: "np.array", masses: "np.array", mol_tol: "float", sea1_subset: "np.array", sea2_subset: "np.array",  axis=None):
    """
    Find C_2 axes by testing vectors mutually orthogonal to sets of linear SEAs.

    Parameters
    ----------
    positions: np.array
        Positions of molecule
    masses: np.array
        Array of masses of molecule
    mol_tol: float
        Tolerance set for the molecule
    sea1_subset: np.array
        SEA.subset
    sea2_subset: np.array
        SEA.subset
    axis: None or np.array, optional
        If not None, only search for C_2 axes that are not equivalent to axis. Array of shape (3,)    

    Returns
    -------
    np.array
        C_2 axis of shape (3,)
    """
    rij = positions[sea1_subset[0],:] - positions[sea1_subset[1],:]
    rkl = positions[sea2_subset[0],:] - positions[sea2_subset[1],:]
    c2_axis = normalize(np.cross(rij, rkl))
    if c2_axis is None or (c2_axis == np.zeros(3)).all():
        return np.zeros(3)
    if axis is not None and issame_axis(c2_axis, axis):
        return np.zeros(3)
    c2 = Cn(c2_axis,2)
    if transform_isequivalent(positions, masses, mol_tol, c2):
        return c2_axis
    return np.zeros(3)

def highest_order_axis(rotations: "List[RotationElement]"): 
    """
    Sorts rotations by highest-order rotation axis first.

    Parameters
    ----------
    rotations: List[RotationElement]

    Returns
    -------
    List[RotationElemtns]
        RotationElement List order by highest-order rotation axis
    """
    m = rotations[0].order
    for elem_rot in rotations:
        m = max(m, elem_rot.order)
    return m

def is_there_sigmah(mol: "Atoms", paxis: "np.array"):
    """
    Check for reflection plane with same normal axis as paxis.

    Parameters
    ----------
    mol: ase.Atoms
    paxis: np.array
        Array of shape (3,)

    Returns
    -------
    bool
    """
    sigmah = reflection_matrix(paxis)
    return transform_isequivalent(mol.positions, mol.get_masses(), mol.info['tol'], sigmah)

def is_there_sigmav(mol: "Atoms", SEAs: "List[SEA]", paxis: "np.array"):
    """
    Check for reflection planes with normal axis orthogonal to paxis.

    Parameters
    ----------
    mol: ase.Atoms
    SEAs: List[SEA]
    paxis: np.array
        Array of shape (3,)

    Returns
    -------
    bool
    """
    axes = []
    positions = mol.positions
    masses = mol.get_masses()
    mol_tol = mol.info['tol']
    for sea in SEAs:
        length = len(sea.subset)
        if length < 2:
            continue
        A = sea.subset[0]
        for i in range(1, length):
            B = sea.subset[i]
            n = normalize(positions[A,:] - positions[B,:])
            if n is not None and not (n == np.zeros(3)).all():
                sigma = reflection_matrix(n)
                if transform_isequivalent(positions, masses, mol_tol, sigma):
                    axes.append(n)
    if len(axes) < 1:
        if mol_is_planar(mol):
            return True, planar_mol_axis(mol)
        else:
            return False, None
    unique_axes = [axes[0]]
    for i in axes:
        check = True
        for j in unique_axes:
            if issame_axis(i,j):
                check = False
                break
        if check:
            unique_axes.append(i)
    for i in unique_axes:
        if not issame_axis(i, paxis):            
            return True, i
    return False, None

def mol_is_planar(mol: "Atoms"):
    """
    Check if all atoms in the molecule lie in a plane.

    Parameters
    ----------
    mol: ase.Atoms

    Returns
    -------
    bool
    """    
    mol_tol = mol.info['tol']

    rank = np.linalg.matrix_rank(mol.positions, tol=mol_tol)

    if rank < 3:
        axis = planar_mol_axis(mol)
        new_mol, _, _ = rotate_mol_to_symels(mol, axis, np.array([0, 0, 0]))
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        new_positions = new_mol.positions
        positions_B = transform(new_positions, matrix)
        for i in range(len(mol)):
            # Check if atoms are about at the same Cartesian point
            if np.linalg.norm(new_positions[i,:] - positions_B[i,:]) >= mol_tol:
            #if not np.isclose(new_positions[i,:], positions_B[i,:], atol=mol_tol).all():
                return False        
        return True
    return False

def planar_mol_axis(mol: "Atoms"):
    """
    Returns the normal axis to the plane of a planar molecule.

    Parameters
    ----------
    mol: ase.Atoms

    Returns
    -------
    np.array or None
        Array of shape (3,) or None

    """    
    coords = mol.positions - mol.positions.mean(axis=0)
    _, _, vh = np.linalg.svd(coords, full_matrices=False)    
    axis = vh[-1]
    axis = normalize(axis)
    return axis

@njit
def _jit_find_C3s_for_Ih(size, positions, masses, mol_tol):
    c3_axes = []
    for i in range(size):
        for j in range(i+1, size):
            for k in range(j+1, size):
                #if i != j and i != k:
                rij = positions[i,:] - positions[j,:]
                rjk = positions[j,:] - positions[k,:]
                rik = positions[i,:] - positions[k,:]
                nij2 = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]
                njk2 = rjk[0]*rjk[0] + rjk[1]*rjk[1] + rjk[2]*rjk[2]
                nik2 = rik[0]*rik[0] + rik[1]*rik[1] + rik[2]*rik[2]
                if vec_isclose(nij2, njk2, atol=mol_tol) and vec_isclose(nij2, nik2, atol=mol_tol):
                    c3_axis = normalize(np.cross(rij, rjk))
                    if not (c3_axis == np.zeros(3)).all():
                        c3 = Cn(c3_axis, 3)
                        if transform_isequivalent(positions, masses, mol_tol, c3):
                            c3_axes.append(c3_axis)
    unique_axes = []
    unique_axes.append(c3_axes[0])
    for i in c3_axes:
        check = True
        for j in unique_axes:
            if issame_axis(i, j):
                check = False
                break
        if check:
            unique_axes.append(i)
    chk = len(unique_axes)
    if chk != 10:
        raise Exception(f"Unexpected number of C3 axes for Ih point group: Found {chk} unique C3 axes, expected 10")
    return unique_axes

def find_C3s_for_Ih(mol: "Atoms"):
    """
    Finds the twenty C3 axes for an Ih point group so the paxis and saxis can be defined.

    Parameters
    ----------
    mol: ase.Atoms

    Returns
    -------
    List[np.array]
        Array of shape (3,)
    """

    return _jit_find_C3s_for_Ih(len(mol), mol.positions, mol.get_masses(), mol.info['tol'])

@njit
def _check_square(va, vb, a, b, c, d, positions, masses, mol_tol):
    if (vec_isclose(a, b, atol=mol_tol) and
            vec_isclose(c, d, atol=mol_tol) and
            vec_isclose(a, c, atol=mol_tol)):
        c4_axis = normalize(np.cross(va, vb))
        if not (c4_axis == np.zeros(3)).all():
            c4 = Cn(c4_axis, 4)
            if transform_isequivalent(positions, masses, mol_tol, c4):
                return c4_axis
    return np.zeros(3)

@njit
def _jit_find_C4s_for_Oh(size, positions, masses, mol_tol):
    c4_axes = []
    for i in range(size):
        for j in range(i+1, size):
            for k in range(j+1, size):
                for l in range(k+1, size):
                    if i != j and k != l and i != k:
                        rij = positions[i,:] - positions[j,:]
                        rjk = positions[j,:] - positions[k,:]
                        rkl = positions[k,:] - positions[l,:]
                        ril = positions[i,:] - positions[l,:]

                        rik = positions[i,:] - positions[k,:]
                        rjl = positions[j,:] - positions[l,:]

                        nij2 = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]
                        njk2 = rjk[0]*rjk[0] + rjk[1]*rjk[1] + rjk[2]*rjk[2]
                        nkl2 = rkl[0]*rkl[0] + rkl[1]*rkl[1] + rkl[2]*rkl[2]
                        nil2 = ril[0]*ril[0] + ril[1]*ril[1] + ril[2]*ril[2]

                        nik2 = rik[0]*rik[0] + rik[1]*rik[1] + rik[2]*rik[2]
                        njl2 = rjl[0]*rjl[0] + rjl[1]*rjl[1] + rjl[2]*rjl[2]

                        c4_axis = _check_square(rij, rjk, nij2, njk2, nkl2, nil2, positions, masses, mol_tol)
                        if not (c4_axis == np.zeros(3)).all():
                            c4_axes.append(c4_axis)
                        c4_axis = _check_square(rij, rjl, nij2, njl2, nkl2, nik2, positions, masses, mol_tol)
                        if not (c4_axis == np.zeros(3)).all():
                            c4_axes.append(c4_axis)
                        c4_axis = _check_square(rik, rjk, nik2, njk2, njl2, nil2, positions, masses, mol_tol)
                        if not (c4_axis == np.zeros(3)).all():
                            c4_axes.append(c4_axis)

    unique_axes = []
    unique_axes.append(c4_axes[0])
    for i in c4_axes:
        check = True
        for j in unique_axes:
            if issame_axis(i,j):
                check = False
                break
        if check:
            unique_axes.append(i)
    chk = len(unique_axes)
    if chk != 3:
        raise Exception(f"Unexpected number of C4 axes for Oh point group: Found {chk} unique C4 axes, expected 3")
    return unique_axes

def find_C4s_for_Oh(mol: "Atoms"):
    """
    Finds the three C4 axes for an Oh point group so the paxis and saxis can be defined.

    Parameters
    ----------
    mol: ase.Atoms

    Returns
    -------
    List[np.array]
        Array of shape (3,)

    """

    return _jit_find_C4s_for_Oh(len(mol), mol.positions, mol.get_masses(), mol.info['tol'])