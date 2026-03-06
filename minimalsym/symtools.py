import numpy as np
from .molecule import global_tol

def rotation_matrix(axis, theta):    
    """
    Create rotation matrix about an axis by theta in radians.

    Parameters
    ----------
    axis: np.array
        Cartesian vector defining rotation axis, shape(3,).
    theta: float
        Angle of rotation in radians.

    Returns
    -------
    np.array
        Matrix defining rotation on column vector, shape (3,3).
    """
    kmat1 = np.array([[0.0, -axis[2], axis[1]], 
                      [axis[2], 0.0, -axis[0]], 
                      [-axis[1], axis[0], 0.0]])
    kmat2 = np.matmul(kmat1,kmat1)
    rodriguesrm = np.eye(3) + np.sin(theta)*kmat1 + (1.0 - np.cos(theta))*kmat2
    return rodriguesrm

def reflection_matrix(axis):    
    """
    Create reflection matrix about a plane defined by its normal vector.

    Parameters
    ----------
    axis: np.array
        Cartesian vector defining the plane normal vector, shape(3,).

    Returns
    -------
    np.array
        Matrix defining reflection on column vector, shape (3,3).
    """
    M = np.zeros((3,3))
    for i in range(3):
        for j in range(i,3):
            if i == j:
                M[i,i] = 1 - 2*(axis[i]**2)
            else:
                M[i,j] = -2 * axis[i] * axis[j]
                M[j,i] = M[i,j]
    return M

def inversion_matrix():
    """
    Create cartesian inversion matrix.

    Returns
    -------
    np.array
        Matrix defining inversion, shape(3,3).
    """
    return -1*np.eye(3)

def Cn(axis, n):
    """
    Wrapper around rotation_matrix for producing a C_n rotation about axis.

    Parameters
    ----------
    axis: np.array
        Cartesian vector defining rotation axis, shape(3,).
    n: int
        Defines rotation angle by theta = 2 pi / n.
    
    Returns
    -------
    np.array
        Matrix defining proper rotation on column vector, shape(3,3).
    """
    theta = 2*np.pi/n
    return rotation_matrix(axis, theta)

def Sn(axis, n):
    """
    Improper rotation S_n about an axis.
    
    Parameters
    ----------
    axis: np.array
        Cartesian vector defining rotation axis, shape(3,).
    n: int
        Defines rotation angle by theta = 2 pi / n.

    Returns
    -------
    np.array
        Matrix defining improper rotation on column vector, shape(3,3)
    """
    return np.dot(reflection_matrix(axis), Cn(axis, n))

def isequivalent(A,B):
    """
    Returns True if molecule A and B are equivalent with respect to permutation of like atoms.

    Parameters
    ----------
    A: ase.Atoms
        Molecule A.
    B: ase.Atoms
        Molecule B.

    Returns
    -------
    bool
        True if equivalent, False if not.
    """
    if A.info["tol"] >= B.info["tol"]:
        eq_tol = A.info["tol"]
    else:
        eq_tol = B.info["tol"]
    matched_already = []
    for i in range(len(A)):
        for j in range(len(B)):
            # Reduce search list so large molecules are a bit faster
            if j not in matched_already:
                # Check that masses are equal
                if A.get_masses()[i] == B.get_masses()[j]:
                    # Check if atoms are about at the same Cartesian point
                    zs = abs(A.positions[i,:]-B.positions[j,:])
                    if np.allclose(zs, [0,0,0], atol=eq_tol):
                        matched_already.append(j)
                        break
    # Did we find a match for each atom? If so we win
    if len(matched_already) == len(A):
        return True
    return False

def calcmoit(atoms):
    """
    Calculates the moment of inertia tensor for a list of atoms.

    Parameters
    ----------
    atoms: ase.Atoms
        Set of atoms.

    Returns
    -------
    np.array
        Cartesian moment of inertia tensor, shape(3,3).
    """
    I = np.zeros((3,3))
    atoms.translate(-atoms.get_center_of_mass())
    for i in range(3):
        for j in range(3):
            if i == j:
                for k in range(len(atoms)):
                    I[i,i] += atoms.get_masses()[k]*(atoms.positions[k,(i+1)%3]**2+atoms.positions[k,(i+2)%3]**2)
            else:
                for k in range(len(atoms)):
                    I[i,j] -= atoms.get_masses()[k]*atoms.positions[k,i]*atoms.positions[k,j]
    return I

def normalize(a):
    """
    Normalize vector a to unit length, return None if the input vector is of zero length.

    Parameters
    ----------
    a: np.array
        Vector of arbitrary magnitude, shape(n,).

    Returns
    -------
    np.array or None
        Normalized vector shape(n,) or None if the magnitude of ``a`` is less than the global tolerance.
    """
    n = np.linalg.norm(a)
    if n <= global_tol:
        return None
    return a / np.linalg.norm(a)

def issame_axis(a, b, tol=global_tol):
    """
    Return True if vectors a and b are colinear within the global tolerance.

    Paremeters
    ----------
    a: np.array
        Vector a, shape(n,).
    b: np.array
        Vector b, shape(n,).
    tol: float
        Tolerance for error, default is ``global_tol``.

    Returns
    -------
    bool
        True if vectors are colinear, False if not colinear or if either vector has zero length.
    """
    A = normalize(a)
    B = normalize(b)
    if A is None or B is None:
        return False
    d = abs(np.dot(A,B))
    return np.isclose(d, 1.0, atol=tol)

def isfactor(n,a):
    """
    Return True if a divides n.

    Parameters
    ----------
    n: int
        Dividend.
    a: int
        Divisor.

    Returns
    -------
    bool
        True if ``a`` divides ``n`` with remainder 0.
    """
    if n % a == 0:
        return True
    else:
        return False

def reduce(n, i):
    """
    Divide n and i by their greatest common divisor g.

    Parameters
    ----------
    n: int
    i: int

    Returns
    -------
    Tuple
        Tuple of n/g and i/g, shape(int, int).
    """
    g = gcd(n, i)
    return n//g, i//g # floor divide to get an int, there should never be a remainder since we are dividing by the gcd

def gcd(A, B):
    """
    A quick implementation of the Euclid algorithm for finding the greatest common divisor between A and B.
    
    Parameters
    ----------
    A: int
    B: int

    Returns
    -------
    int
        Greatest common divisor between A and B.
    """
    a = max(A,B)
    b = min(A,B)
    if a == 0:
        return b
    elif b == 0:
        return a
    else:
        r = a % b
        return gcd(b, r)