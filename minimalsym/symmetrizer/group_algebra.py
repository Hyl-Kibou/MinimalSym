"""
group_algebra.py — Pure arithmetic for point-group element products.

Functions here have no side effects and depend only on integer arithmetic
and numba JIT. They are used by cyclic_dihedral.py and symel_gen.py.
"""

import numpy as np
from numba import njit


@njit
def _omega(m, n):
    """Reduce the power m of an S_n element to its canonical symbol index and axis order."""
    gcd_val = np.gcd(m, n)
    l = (m / gcd_val) + (n / gcd_val) * (1 - ((m / gcd_val) % 2))
    return int(l), int(n / gcd_val)


@njit
def _mult_iCnm(m, n):
    """Return the canonical power and axis order of i * C_n^m."""
    a = (2 * m + n) % (2 * n)
    return _omega(a, 2 * n)


@njit
def _mult_sigmahCnm(m, n):
    """Return the canonical power and axis order of sigma_h * C_n^m."""
    return _omega(m, n)

@njit
def _mult_CSC2sigma(m, n, pre, post):
    """Return the symbol of the product of a principal-axis element with a C_2' or sigma element."""
    if pre == "C":
        even_odd = ["'", "''"] if post == "C_2" else ["_v", "_d"]
    else:
        if post == "C_2":
            post = "sigma"
        even_odd = ["_v", "_d"]
    if n % 2 == 0:
        if pre == "iC":
            if n % 4 == 0:
                label = even_odd[m % 2]
                a = ((n >> 2) + (m >> 1)) % (n >> 1)
            else:
                label = even_odd[(m + 1) % 2]
                a = ((n >> 2) + ((m + 1) >> 1)) % (n >> 1)
            return post + label + f"({a})"
        else:
            label = even_odd[m % 2]
            a = (m >> 1) % (n >> 1)
            return post + label + f"({a})"
    else:
        if pre == "iC":
            label = "_d"
            a = ((n >> 1) + m) % n
        else:
            label = even_odd[0]
            a = ((((n >> 1) + 1) * (m % 2)) + (m >> 1)) % n
        return post + label + f"({a})"
