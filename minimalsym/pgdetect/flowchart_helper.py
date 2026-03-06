import numpy as np
from ..symtools import reflection_matrix, Cn, isequivalent, calcmoit, normalize, issame_axis, isfactor
from ..symtext.symtext_helper import rotate_mol_to_symels
from ..molecule import Molecule
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

def intersect(a: "List[RotationElement]", b: "List[RotationElement]"):
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
    out = []
    for re_a in a:
        for re_b in b:
            if re_a == re_b:
                out.append(re_a)
                break
    return out

def rotation_set_intersection(rotation_set: "List[List[RotationElement]]"):
    """
    Find intersection of all RotationElement sets in a rotation_set

    Parameters
    ----------
    rotation_set: List[List[RotationElement]]
        Set of sets of RotationElement

    Returns
    -------
    List[RotationElement]
        Intersection of rotation sets
    """
    out = rotation_set[0]
    if len(rotation_set) > 1:
        for i in range(len(rotation_set)):
            out = intersect(out, rotation_set[i])
    return out

def find_rotation_sets(mol: "Atoms", SEAs: "List[SEA]"):
    """
    For each set of symmetry equivalent atoms, find the set of possible RotationElements

    Parameters
    ----------
    mol: ase.Atoms
    SEAs: List[minimalsym.SEA]

    Returns
    -------
    List[List[RotationElement]]
    """
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
            if np.isclose(Ia, Ib, atol=mol.info["tol"]) and np.isclose(Ia, Ic, atol=mol.info["tol"]):
                sea.label = "Spherical"
            elif np.isclose(Ia+Ib, Ic, atol=mol.info["tol"]):
                axis = Icv
                sea.axis = axis
                if np.isclose(Ia, Ib, atol=mol.info["tol"]):
                    sea.label = "Regular Polygon"
                    for i in range(2,length+1):
                        if isfactor(length,i):
                            re = RotationElement(axis,i)
                            out_per_SEA.append(re)
                else:
                    sea.label = "Irregular Polygon"
                    for i in range(2,length):
                        if isfactor(length, i):
                            re = RotationElement(axis, i)
                            out_per_SEA.append(re)
            else:
                if not (np.isclose(Ia, Ib, atol=mol.info["tol"]) or np.isclose(Ib, Ic, atol=mol.info["tol"])):
                    sea.label = "Asymmetric Rotor"
                    for i in [Iav, Ibv, Icv]:
                        re = RotationElement(i, 2)
                        out_per_SEA.append(re)
                else:
                    if np.isclose(Ia, Ib, atol=mol.info["tol"]):
                        sea.label = "Oblate Symmetric Top"
                        axis = Icv
                        sea.axis = Icv
                    else:
                        sea.label = "Prolate Symmetric Top"
                        axis = Iav
                        sea.axis = Iav
                    k = length//2
                    for i in range(2,k+1):
                        if isfactor(k,i):
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
    if len(rotation_set) < 1:
        return []
    molmoit = calcmoit(mol)
    evals = np.sort(np.linalg.eigh(molmoit)[0])
    if evals[0] == 0.0 and np.isclose(evals[1], evals[2], atol=mol.info["tol"]):
        for i in range(np.shape(mol.positions)[0]):
            if normalize(mol.positions[i,:]) is not None:
                axis = normalize(mol.positions[0,:])
        re = RotationElement(axis, 0)
        return [re]
    rsi = rotation_set_intersection(rotation_set)
    out = []
    for i in rsi:
        rmat = Cn(i.axis, i.order)
        molB = Molecule.transform(mol, rmat)
        if isequivalent(mol, molB):
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
    SEAs: List[minimalsym.SEA]

    Returns
    -------
    np.array
        NumPy array of shape (3,)

    """
    for sea in SEAs:
        a = c2a(mol, sea)
        if a is not None:
            return a
        else:
            b = c2b(mol, sea)
            if b is not None:
                return b
            else:
                if sea.label == "Linear":
                    for sea2 in SEAs:
                        if sea == sea2:
                            continue
                        elif sea2.label == "Linear":
                            c = c2c(mol, sea, sea2)
                            if c is not None:
                                return c
    return None

def compute_R_max(mol: "Atoms", axis: "np.array"):
    """
    Compute the maximum distance of any atom from a given axis.

    Parameters
    ----------
    mol: ase.Atoms
    axis: np.array
        NumPy array of shape (3,), must be normalized

    Returns
    -------
    float
        maximum perpendicular distance
    """
    axis = normalize(axis)
    coords = mol.positions  # shape (N,3)
    # projection along axis
    proj = np.dot(coords, axis)[:, np.newaxis] * axis[np.newaxis, :]
    # perpendicular component
    perp = coords - proj
    # distance from axis
    dists = np.linalg.norm(perp, axis=1)
    # maximum distance
    R_max = np.max(dists)
    return R_max

def is_there_ortho_c2(mol: "Atoms", SEAs: "List[SEA]", paxis: "np.array"):
    """
    Search for any possible C_2 rotation axes that are orthogonal to paxis, return the first one found.

    Parameters
    ----------
    mol: ase.Atoms
    SEAs: List[minimalsym.SEA]
    paxis: np.array
        NumPy array of shape (3,)

    Returns
    -------
    tuple(bool, np.array) or None
        True if found and new C_2 axis of shape (3,)
    """

    ortho_tol = mol.info["tol"] / compute_R_max(mol, paxis) * 1.10

    for sea in SEAs:
        b = c2b(mol, sea, axis=paxis)
        if b is not None and abs(np.dot(b, paxis)) <= ortho_tol:
            return True, b
        else:
            a = c2a(mol, sea, axis=paxis)
            if a is not None and abs(np.dot(a, paxis)) <= ortho_tol:
                return True, a
            else:
                if sea.label == "Linear":
                    for sea2 in SEAs:
                        if sea == sea2:
                            continue
                        elif sea2.label == "Linear":
                            c = c2c(mol, sea, sea2, axis=paxis)
                            if c is not None and abs(np.dot(c, paxis)) <= ortho_tol:
                                return True, c
    return False, None

def num_C2(mol: "Atoms", SEAs: "List[SEA]"):
    """
    Find the number of C_2 axis present and the axes defining them.
    
    Parameters
    ----------
    mol: ase.Atoms
    SEAs: List[minimalsym.SEA]

    Returns
    -------
    tuple(int, List[np.array])
        Number of C_2 axes and the C_2 axes of shape (3,)
    """
    axes = []
    for sea in SEAs:
        a = c2a(mol, sea, all=True)
        if a is not None:
            for i in a:
                axes.append(i)
        b = c2b(mol, sea, all=True)
        if b is not None:
            for i in b:
                axes.append(i)
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

def c2a(mol: "Atoms", sea: "SEA", axis=None, all: bool=False):
    """
    Find C_2 axes by testing vectors formed from the origin and midpoint of all pairs of symmetry equivalent atoms.

    Parameters
    ----------
    mol: ase.Atoms
    sea: minimalsym.SEA
    axis: None or np.array, optional
        If not None, only search for C_2 axes that are not equivalent to axis. Array of shape (3,)
    all: bool, optional
        If False return first C_2 found, if True search all atom pairs for C_2 axes

    Returns
    -------
    np.array or List[np.array]
        C_2 axis or list of C_2 axes (if all=True) of shape (3,)
    """
    length = len(sea.subset)
    out = []
    for i in range(length):
        for j in range(i+1,length):
            midpoint = mol.positions[sea.subset[i],:] + mol.positions[sea.subset[j],:]
            if np.isclose(midpoint, [0,0,0], atol=mol.info["tol"]).all():
                continue
            else:
                midpoint = normalize(midpoint)
                if axis is not None and issame_axis(midpoint, axis) or midpoint is None:
                    continue
                c2 = Cn(midpoint, 2)
                molB = Molecule.transform(mol, c2)
                if isequivalent(mol, molB):
                    if all:
                        out.append(midpoint)
                    else:
                        return midpoint
    if len(out) < 1:
        return None
    return out

def c2b(mol: "Atoms", sea: "SEA", axis=None, all: bool=False):
    """
    Find C_2 axes by testing vectors which pass through individual symmetry equivalent atoms.

    Parameters
    ----------
    mol: ase.Atoms
    sea: minimalsym.SEA
    axis: None or np.array, optional
        If not None, only search for C_2 axes that are not equivalent to axis. Array of shape (3,)
    all: bool, optional
        If False return first C_2 found, if True search all atom pairs for C_2 axes

    Returns
    -------
    np.array or List[np.array]
        C_2 axis or list of C_2 axes (if all=True) of shape (3,)
    """
    length = len(sea.subset)
    out = []
    for i in range(length):
        c2_axis = normalize(mol.positions[sea.subset[i],:])
        if c2_axis is None:
            continue
        if axis is not None and issame_axis(c2_axis, axis):
            continue
        c2 = Cn(c2_axis, 2)
        molB = Molecule.transform(mol, c2)
        if isequivalent(mol, molB):
            if all:
                out.append(c2_axis)
            else:
                return c2_axis
    if len(out) < 1:
        return None
    return out

def c2c(mol: "Atoms", sea1: "SEA", sea2: "SEA", axis=None):
    """
    Find C_2 axes by testing vectors mutually orthogonal to sets of linear SEAs.

    Parameters
    ----------
    mol: ase.Atoms
    sea1: minimalsym.SEA
    sea2: minimalsym.SEA
    axis: None or np.array, optional
        If not None, only search for C_2 axes that are not equivalent to axis. Array of shape (3,)    

    Returns
    -------
    np.array or List[np.array]
        C_2 axis or list of C_2 axes (if all=True) of shape (3,)   
    """
    rij = mol.positions[sea1.subset[0],:] - mol.positions[sea1.subset[1],:]
    rkl = mol.positions[sea2.subset[0],:] - mol.positions[sea2.subset[1],:]
    c2_axis = normalize(np.cross(rij, rkl))
    if c2_axis is None:
        return None
    if axis is not None and issame_axis(c2_axis, axis):
        return None
    c2 = Cn(c2_axis,2)
    molB = Molecule.transform(mol, c2)
    if isequivalent(mol, molB):
        return c2_axis
    return None

def highest_order_axis(rotations: "List[RotationElement]"): 
    """
    Sorts rotations by highest order rotation axis first.

    Parameters
    ----------
    rotations: List[RotationElement]

    Returns
    -------
    List[RotationElemtns]
    """
    ns = []
    for i in range(len(rotations)):
        ns.append(rotations[i].order)
    return np.sort(ns)[-1]

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
    molB = Molecule.transform(mol, sigmah)
    return isequivalent(mol, molB)

def is_there_sigmav(mol: "Atoms", SEAs: "List[SEA]", paxis: "np.array"):
    """
    Check for reflection planes with normal axis orthogonal to paxis.

    Parameters
    ----------
    mol: ase.Atoms
    SEAs: List[minimalsym.SEA]
    paxis: np.array
        Array of shape (3,)

    Returns
    -------
    bool
    """
    axes = []
    for sea in SEAs:
        length = len(sea.subset)
        if length < 2:
            continue
        A = sea.subset[0]
        for i in range(1,length):
            B = sea.subset[i]
            #n = normalize(mol[A].xyz - mol[B].xyz)
            n = normalize(mol.positions[A,:] - mol.positions[B,:])
            if n is not None:
                sigma = reflection_matrix(n)
                molB = Molecule.transform(mol, sigma)
                if isequivalent(mol, molB):
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
        if issame_axis(i, paxis):
            continue
        else:
            return True, i
    return False, None

def mol_is_planar(mol: "Atoms"):
    """
    Check if all atoms in the molecue lie in a plane.

    Parameters
    ----------
    mol: ase.Atoms

    Returns
    -------
    bool
    """    
    rank = np.linalg.matrix_rank(mol.positions, tol=mol.info["tol"])

    if rank < 3:
        axis = planar_mol_axis(mol)
        new_mol, _, _ = rotate_mol_to_symels(mol, axis, np.array([0, 0, 0]))
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

        molB = Molecule.transform(new_mol, matrix)
        for i in range(len(mol)):
            # Check if atoms are about at the same Cartesian point
            if not np.isclose(new_mol.positions[i,:], molB.positions[i,:], atol=mol.info["tol"]).all():                
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
    c3_axes = []
    for i in range(len(mol)):
        for j in range(len(mol)):
            for k in range(len(mol)):
                if i != j and i != k:
                    rij = mol.positions[i,:] - mol.positions[j,:]
                    rjk = mol.positions[j,:] - mol.positions[k,:]
                    rik = mol.positions[i,:] - mol.positions[k,:]
                    nij = np.linalg.norm(rij)
                    njk = np.linalg.norm(rjk)
                    nik = np.linalg.norm(rik)
                    if np.isclose(nij, njk, atol=mol.info["tol"]) and np.isclose(nij, nik, atol=mol.info["tol"]):
                        c3_axis = normalize(np.cross(rij, rjk))
                        if c3_axis is not None:
                            c3 = Cn(c3_axis, 3)
                            molB = Molecule.transform(mol, c3)
                            if isequivalent(mol, molB):
                                c3_axes.append(c3_axis)
    unique_axes = [c3_axes[0]]
    for i in c3_axes:
        check = True
        for j in unique_axes:
            if issame_axis(i,j):
                check = False
                break
        if check:
            unique_axes.append(i)
    chk = len(unique_axes)
    if chk != 10:
        raise Exception(f"Unexpected number of C3 axes for Ih point group: Found {chk} unique C3 axes")
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
    c4_axes = []
    for i in range(len(mol)):
        for j in range(len(mol)):
            for k in range(len(mol)):
                for l in range(len(mol)):
                    if i != j and k != l and i != k:
                        rij = mol.positions[i,:] - mol.positions[j,:]
                        rjk = mol.positions[j,:] - mol.positions[k,:]
                        rkl = mol.positions[k,:] - mol.positions[l,:]
                        ril = mol.positions[i,:] - mol.positions[l,:]
                        nij = np.linalg.norm(rij)
                        njk = np.linalg.norm(rjk)
                        nkl = np.linalg.norm(rkl)
                        nil = np.linalg.norm(ril)
                        if np.isclose(nij, njk, atol=mol.info["tol"]) and np.isclose(nkl, nil, atol=mol.info["tol"]) and np.isclose(nij, nkl, atol=mol.info["tol"]):
                            c4_axis = normalize(np.cross(rij, rjk))
                            if c4_axis is not None:
                                c4 = Cn(c4_axis, 4)
                                molB = Molecule.transform(mol, c4)
                                if isequivalent(mol, molB):
                                    c4_axes.append(c4_axis)
    unique_axes = [c4_axes[0]]
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
        raise Exception("Unexpected number of C4 axes for Oh point group: Found $(chk) unique C4 axes")
    return unique_axes