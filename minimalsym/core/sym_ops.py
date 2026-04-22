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
    Normalize vector a to unit length, return zero vector shape(n, ) if the input vector is of zero length.

    Parameters
    ----------
    a: np.array
        Vector of arbitrary magnitude, shape(n,).

    Returns
    -------
    np.array
        Normalized vector shape(n,) or zero vector shape(n, ) if the magnitude of ``a`` is less than the global tolerance.
    """
    n = vec_norm(a)
    if n <= global_tol:
        return np.zeros(3)
    return a / n

@njit
def float_isclose(a: float, b: float, rtol:float=1e-05, atol:float=1e-08) -> bool:
    """Compare two floats (Numba-compatible)."""
    return np.abs(a - b) <= (atol + rtol * max(np.abs(b), np.abs(a)))

@njit
def inertia_isclose(a: float, b: float, rtol:float=1e-05, atol:float=1e-08) -> bool:
    """Compare two inertia (Numba-compatible)."""
    if a == 0.0 or b == 0.0:
        return float_isclose(a, b, rtol=0.0, atol=atol)
    else:
        return float_isclose(a, b, rtol=rtol, atol=0.0)

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
    return float_isclose(d, 1.0, atol=tol)

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
    g = _gcd(n, i)
    return n//g, i//g

@njit
def _gcd(A, B):
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
        return _gcd(b, r)

@njit
def unique_sorted(arr):
    """
    Return the sorted unique elements of a 1D array.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D array of comparable elements.

    Returns
    -------
    np.ndarray
        Sorted array containing the unique elements of `arr`.

    Notes
    -----
    This is a Numba-compatible alternative to `np.unique`, implemented
    using sorting followed by linear duplicate removal.

    Equality is tested using exact comparison (`!=`), so this function
    is best suited for integer or discretized floating-point data.
    """
    if len(arr) == 0:
        return arr

    sorted_arr = np.sort(arr)

    # allocate output (max possible size)
    out = np.empty_like(sorted_arr)
    count = 0

    out[count] = sorted_arr[0]
    count += 1

    for i in range(1, len(sorted_arr)):
        if sorted_arr[i] != out[count-1]:
            out[count] = sorted_arr[i]
            count += 1

    return out[:count]

@njit
def canonical(v):
    """
    Returns canonical form of vector.
    Ensures that the first non-zero element is positive.

    Parameters
    ----------
    v : np.ndarray, shape (3,)
        Vector to canonicalize

    Returns
    -------
    np.ndarray, shape (3,)
        Canonical vector
    """
    v = normalize(v)
    for i in range(len(v)):
        if abs(v[i]) < 1e-08:
            continue
        if v[i] < 0:
            v = -v
        break
    return v

@njit
def generate_cyclic_axes(static_axis:np.array, saxis:np.array, n:int, num_elem_generate:int = -1):
    """
    Generate a set of axes by rotating a reference axis around a fixed axis.

    Parameters
    ----------
    static_axis : np.ndarray, shape (3,)
        Unit vector defining the rotation axis (e.g., principal symmetry axis).
    saxis : np.ndarray, shape (3,)
        Reference axis to be rotated about `static_axis`.
    n : int
        Order of the rotation symmetry (Cn). Defines the angular step 2π/n.
        If `n` is even, it is internally doubled to ensure full coverage of
        distinct orientations.
    num_elem_generate : int, optional
        Number of rotated axes to generate. If <= 0, defaults to `n`.

    Returns
    -------
    np.ndarray, shape (num_elem_generate, 3)
        Array of rotated axes. The first row is `saxis`, and subsequent rows
        are obtained by successive rotations about `static_axis`.

    Notes
    -----
    This function is typically used to generate symmetry-equivalent axes
    (e.g., C2 axes perpendicular to a principal axis in dihedral groups).
    """
    if num_elem_generate <= 0:
        num_elem_generate = n
    if n % 2 == 0:
        n *= 2
    rotated_axes = np.empty((num_elem_generate, 3), dtype=np.float64)
    rotated_axes[0, :] = saxis

    for ii in range(1, num_elem_generate):
        theta = 2 * np.pi * ii / n
        R = rotation_matrix(static_axis, theta)
        rotated_axis = R @ saxis
        #rotated_axis = rotated_axis - np.dot(rotated_axis, static_axis) * static_axis
        rotated_axes[ii, :] = normalize(rotated_axis)

    return rotated_axes