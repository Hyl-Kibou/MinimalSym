import numpy as np
from .constants import NUMERICAL_TOL as global_tol
from numba import njit

@njit
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
    kmat2 = kmat1 @ kmat1
    rodriguesrm = np.eye(3) + np.sin(theta)*kmat1 + (1.0 - np.cos(theta))*kmat2
    return rodriguesrm

@njit
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
    M = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            if i == j:
                M[i,i] = 1 - 2*(axis[i]**2)
            else:
                M[i,j] = -2 * axis[i] * axis[j]
                M[j,i] = M[i,j]
    return M

@njit
def inversion_matrix():
    """
    Create cartesian inversion matrix.

    Returns
    -------
    np.array
        Matrix defining inversion, shape(3,3).
    """
    return -1*np.eye(3)

@njit
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

@njit
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
        Matrix defining improper rotation on column vector, shape (3,3).
    """
    return np.dot(reflection_matrix(axis), Cn(axis, n))

@njit
def vec_norm_axis(v):
    """Compute Euclidean norm (Numba-compatible)."""
    # Vectorize the norm calculation over rows
    return np.sqrt(np.sum(v**2, axis=1))

@njit
def vec_norm(v):
    """Compute Euclidean norm (Numba-compatible)."""
    #return np.sqrt(np.sum(v**2))
    return np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

@njit
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
    n = vec_norm(a)
    if n <= global_tol:
        return np.zeros(3)
    return a / n

@njit
def vec_isclose(a, b, rtol=1e-05, atol=1e-08):
    """Compare two floats (Numba-compatible)."""
    return np.abs(a - b) <= (atol + rtol * np.abs(b))

@njit
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
        True if vectors are collinear, False if not collinear or if either vector has zero length.
    """
    A_vector = normalize(a)
    B_vector = normalize(b)
    if A_vector is None or B_vector is None:
        return False
    if (A_vector == np.zeros(3)).all() or (B_vector == np.zeros(3)).all():
        return False
    d = np.abs(np.dot(A_vector, B_vector))
    return vec_isclose(d, 1.0, atol=tol)

@njit
def isfactor(n, a):
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
    return n % a == 0

@njit
def reduce(n, i):
    """
    Divide n and i by their greatest common divisor g.

    Parameters
    ----------
    n: int
    i: int

    Returns
    -------
    tuple
        Tuple of n/g and i/g.
    """
    g = gcd(n, i)
    return n//g, i//g

@njit
def gcd(A, B):
    """
    Euclid algorithm for finding the greatest common divisor between A and B.

    Parameters
    ----------
    A: int
    B: int

    Returns
    -------
    int
        Greatest common divisor between A and B.
    """
    a = max(A, B)
    b = min(A, B)
    if a == 0:
        return b
    elif b == 0:
        return a
    else:
        r = a % b
        return gcd(b, r)
