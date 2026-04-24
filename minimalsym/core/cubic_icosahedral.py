"""
cubic_icosahedral.py — Generators for cubic (T, O) and icosahedral (I) point groups.

Precomputed tuples T_SYMELS, TD_SYMELS, TH_SYMELS, O_SYMELS, OH_SYMELS,
I_SYMELS, IH_SYMELS are built once at import time and reused by symel_gen.py.
"""

import numpy as np
from numpy.linalg import matrix_power

from .symel import Symel
from .sym_ops import reflection_matrix, inversion_matrix, Cn, Sn, normalize


# ── Icosahedron geometry ──────────────────────────────────────────────────────

def _icosahedron_vectors():
    """
    Vectors defining the faces, vertices, and edge centers of a regular icosahedron/dodecahedron.

    Returns
    -------
    : tuple(List[np.ndarray], List[np.ndarray], List[np.ndarray])
        Face axes (6), vertex axes (10), and edge-center axes (15).
    """
    s5   = np.sqrt(5)
    phi  = (1 + s5) / 2           # golden ratio phi = (1+sqrt(5))/2

    r5   = 1 / s5                 # 1/sqrt(5), appears 7 times
    r3   = 1 / np.sqrt(3)         # 1/sqrt(3), appears 4 times

    sp10 = np.sqrt((5 + s5) / 10) # sqrt((5+sqrt(5))/10), appears 10 times
    sm10 = np.sqrt((5 - s5) / 10) # sqrt((5-sqrt(5))/10), appears 10 times
    vp   = np.sqrt((5 + 2*s5) / 15) # sqrt((5+2*sqrt(5))/15), appears 5 times
    vm   = np.sqrt((5 - 2*s5) / 15) # sqrt((5-2*sqrt(5))/15), appears 5 times
    sp30 = np.sqrt((5 + s5) / 30) # sqrt((5+sqrt(5))/30), appears 2 times
    sm30 = np.sqrt((5 - s5) / 30) # sqrt((5-sqrt(5))/30), appears 2 times
    t3p  = np.sqrt((3*s5 + 5) / (6*s5)) # sqrt((3*sqrt(5)+5)/(6*sqrt(5))), appears 2 times
    t3m  = np.sqrt((3*s5 - 5) / (6*s5)) # sqrt((3*sqrt(5)-5)/(6*sqrt(5))), appears 2 times

    phim = 1 / (2*phi)            # (sqrt(5)-1)/4, appears 4 times
    phip = phi / 2                # (sqrt(5)+1)/4, appears 4 times
    sp2h = np.sqrt((5 + s5) / 2) / 2  # sqrt((5+sqrt(5))/2)/2, appears 2 times
    sm2h = np.sqrt((5 - s5) / 2) / 2  # sqrt((5-sqrt(5))/2)/2, appears 2 times
    a1p  = np.sqrt(1 + 2*r5) / 2 # sqrt(1+2/sqrt(5))/2, appears 2 times
    a1m  = np.sqrt(1 - 2*r5) / 2 # sqrt(1-2/sqrt(5))/2, appears 2 times

    faces = [
        np.array([0.0,   0.0,        1.0  ]),
        np.array([0.0,   2*r5,       r5   ]),
        np.array([-sp10, (5-s5)/10,  r5   ]),
        np.array([ sp10, (5-s5)/10,  r5   ]),
        np.array([-sm10, -(5+s5)/10, r5   ]),
        np.array([ sm10, -(5+s5)/10, r5   ]),
    ]
    vertices = [
        np.array([0.0,  -np.sqrt(2*(5+s5)/15), vm  ]),
        np.array([0.0,  -np.sqrt(2*(5-s5)/15), vp  ]),
        np.array([-r3,   vp,                   vm  ]),
        np.array([-r3,  -vm,                   vp  ]),
        np.array([ r3,   vp,                   vm  ]),
        np.array([ r3,  -vm,                   vp  ]),
        np.array([-t3p, -sm30,                 vm  ]),
        np.array([ t3m,  sp30,                 vp  ]),
        np.array([-t3m,  sp30,                 vp  ]),
        np.array([ t3p, -sm30,                 vm  ]),
    ]
    edgecenters = [
        np.array([0.0,   sp10,  -sm10]),
        np.array([0.0,   sm10,   sp10]),
        np.array([0.5,  -a1p,   -sm10]),
        np.array([0.5,   a1m,    sp10]),
        np.array([0.5,  -a1m,   -sp10]),
        np.array([0.5,   a1p,    sm10]),
        np.array([1.0,   0.0,    0.0 ]),
        np.array([phim, -sp10/2, sp10]),
        np.array([phim,  sp10/2,-sp10]),
        np.array([phim, -sp2h,   0.0 ]),
        np.array([phim,  sp2h,   0.0 ]),
        np.array([phip, -sm10/2, sm10]),
        np.array([phip,  sm10/2,-sm10]),
        np.array([phip, -sm2h,   0.0 ]),
        np.array([phip,  sm2h,   0.0 ]),
    ]
    return faces, vertices, edgecenters


def _checked_normalize(vec, label):
    """Normalize *vec* and raise ValueError if the result is a zero vector."""
    result = normalize(vec)
    if result is None or (result == np.zeros(3)).all():
        raise ValueError(f"Normalization of axis '{label}' produced a zero vector: {vec}")
    return result


FACE_VEC, VERTEX_VEC, EDGE_VEC = _icosahedron_vectors()


# ── Cubic group generators ────────────────────────────────────────────────────

def _generate_T():
    """
    Generate symmetry elements for the T point group.
    Assumes a tetrahedron contained in a cube.

    Returns
    -------
    List[Symel]
    """
    symels = [Symel("E", None, np.eye(3))]
    C3_1v = _checked_normalize(np.array([1.0, 1.0, 1.0]),  "C3(alpha)")
    C3_2v = _checked_normalize(np.array([-1.0, 1.0, -1.0]), "C3(beta)")
    C3_3v = _checked_normalize(np.array([-1.0, -1.0, 1.0]), "C3(gamma)")
    C3_4v = _checked_normalize(np.array([1.0, -1.0, -1.0]), "C3(delta)")
    C3list = [C3_1v, C3_2v, C3_3v, C3_4v]
    namelist = ("alpha", "beta", "gamma", "delta")
    for i in range(4):
        C3 = Cn(C3list[i], 3)
        C3s = matrix_power(C3, 2)
        symels.append(Symel(f"C_3({namelist[i]})", C3list[i], C3))
        symels.append(Symel(f"C_3^2({namelist[i]})", C3list[i], C3s))
    C2list = np.eye(3)
    namelist = ["x", "y", "z"]
    for i in range(3):
        C2 = Cn(C2list[i], 2)
        symels.append(Symel(f"C_2({namelist[i]})", C2list[i], C2))
    return symels


def _generate_Td():
    """
    Generate symmetry elements for the Td point group.
    Assumes a tetrahedron contained in a cube.

    Returns
    -------
    List[Symel]
    """
    symels = _generate_T()
    sigmas = [
        normalize(np.array([1.0, 1.0, 0.0])), normalize(np.array([1.0, -1.0, 0.0])),
        normalize(np.array([1.0, 0.0, 1.0])), normalize(np.array([1.0, 0.0, -1.0])),
        normalize(np.array([0.0, 1.0, 1.0])), normalize(np.array([0.0, 1.0, -1.0])),
    ]
    namelist = ["xyp", "xym", "xzp", "xzm", "yzp", "yzm"]
    for i in range(6):
        symels.append(Symel(f"sigma_d({namelist[i]})", sigmas[i], reflection_matrix(sigmas[i])))
    S4list = np.eye(3)
    namelist = ["x", "y", "z"]
    for i in range(3):
        S4 = Sn(S4list[i], 4)
        symels.append(Symel(f"S_4({namelist[i]})", S4list[i], S4))
        symels.append(Symel(f"S_4^3({namelist[i]})", S4list[i], matrix_power(S4, 3)))
    return symels


def _generate_Th():
    """
    Generate symmetry elements for the Th point group.
    Assumes a tetrahedron contained in a cube.

    Returns
    -------
    List[Symel]
    """
    symels = _generate_T()
    symels.append(Symel("i", None, inversion_matrix()))
    S6list = [
        normalize(np.array([1.0, 1.0, 1.0])), normalize(np.array([-1.0, 1.0, -1.0])),
        normalize(np.array([-1.0, -1.0, 1.0])), normalize(np.array([1.0, -1.0, -1.0])),
    ]
    namelist = ["alpha", "beta", "gamma", "delta"]
    for i in range(4):
        S6 = Sn(S6list[i], 6)
        symels.append(Symel(f"S_6({namelist[i]})", S6list[i], S6))
        symels.append(Symel(f"S_6^5({namelist[i]})", S6list[i], matrix_power(S6, 5)))
    sigma_list = np.eye(3)
    namelist = ["x", "y", "z"]
    for i in range(3):
        symels.append(Symel(f"sigma_h({namelist[i]})", sigma_list[i], reflection_matrix(sigma_list[i])))
    return symels


def _generate_O():
    """
    Generate symmetry elements for the O point group.
    Assumes operations on a cube.

    Returns
    -------
    List[Symel]
    """
    symels = [Symel("E", None, np.eye(3))]
    C4list = np.eye(3)
    namelist = ["x", "y", "z"]
    for i in range(3):
        C4 = Cn(C4list[i], 4)
        symels.append(Symel(f"C_4({namelist[i]})", C4list[i], C4))
        symels.append(Symel(f"C_2({namelist[i]})", C4list[i], matrix_power(C4, 2)))
        symels.append(Symel(f"C_4^3({namelist[i]})", C4list[i], matrix_power(C4, 3)))
    C3list = [
        normalize(np.array([1.0, 1.0, 1.0])), normalize(np.array([1.0, -1.0, 1.0])),
        normalize(np.array([1.0, 1.0, -1.0])), normalize(np.array([1.0, -1.0, -1.0])),
    ]
    namelist = ["alpha", "beta", "gamma", "delta"]
    for i in range(4):
        C3 = Cn(C3list[i], 3)
        symels.append(Symel(f"C_3({namelist[i]})", C3list[i], C3))
        symels.append(Symel(f"C_3^2({namelist[i]})", C3list[i], matrix_power(C3, 2)))
    C2list = [
        normalize(np.array([1.0, 0.0, 1.0])), normalize(np.array([1.0, 0.0, -1.0])),
        normalize(np.array([1.0, 1.0, 0.0])), normalize(np.array([1.0, -1.0, 0.0])),
        normalize(np.array([0.0, 1.0, 1.0])), normalize(np.array([0.0, -1.0, 1.0])),
    ]
    namelist = ["xzp", "xzm", "xyp", "xym", "yzp", "yzm"]
    for i in range(6):
        symels.append(Symel(f"C_2({namelist[i]})", C2list[i], Cn(C2list[i], 2)))
    return symels


def _generate_Oh():
    """
    Generate symmetry elements for the Oh point group.
    Assumes operations on a cube.

    Returns
    -------
    List[Symel]
    """
    symels = _generate_O()
    symels.append(Symel("i", None, inversion_matrix()))
    S4list = np.eye(3)
    namelist = ["x", "y", "z"]
    for i in range(3):
        S4 = Sn(S4list[i], 4)
        symels.append(Symel(f"S_4({namelist[i]})", S4list[i], S4))
        symels.append(Symel(f"sigma_h({namelist[i]})", S4list[i], reflection_matrix(S4list[i])))
        symels.append(Symel(f"S_4^3({namelist[i]})", S4list[i], matrix_power(S4, 3)))
    S6list = [
        normalize(np.array([1.0, 1.0, 1.0])), normalize(np.array([1.0, -1.0, 1.0])),
        normalize(np.array([1.0, 1.0, -1.0])), normalize(np.array([1.0, -1.0, -1.0])),
    ]
    namelist = ["alpha", "beta", "gamma", "delta"]
    for i in range(4):
        S6 = Sn(S6list[i], 6)
        symels.append(Symel(f"S_6({namelist[i]})", S6list[i], S6))
        symels.append(Symel(f"S_6^5({namelist[i]})", S6list[i], matrix_power(S6, 5)))
    sigma_dlist = [
        normalize(np.array([1.0, 0.0, 1.0])), normalize(np.array([1.0, 0.0, -1.0])),
        normalize(np.array([1.0, 1.0, 0.0])), normalize(np.array([1.0, -1.0, 0.0])),
        normalize(np.array([0.0, 1.0, 1.0])), normalize(np.array([0.0, -1.0, 1.0])),
    ]
    namelist = ["xzp", "xzm", "xyp", "xym", "yzp", "yzm"]
    for i in range(6):
        symels.append(Symel(f"sigma_d({namelist[i]})", sigma_dlist[i], reflection_matrix(sigma_dlist[i])))
    return symels


# ── Icosahedral group generators ──────────────────────────────────────────────

def _generate_I():
    """
    Generate symmetry elements for the I point group.

    Returns
    -------
    List[Symel]
    """
    symels = [Symel("E", None, np.eye(3))]
    faces, vertices, edgecenters = FACE_VEC, VERTEX_VEC, EDGE_VEC
    for i in range(6):
        C5 = Cn(faces[i], 5)
        symels.append(Symel(f"C_5({i})", faces[i], C5))
        symels.append(Symel(f"C_5^2({i})", faces[i], matrix_power(C5, 2)))
        symels.append(Symel(f"C_5^3({i})", faces[i], matrix_power(C5, 3)))
        symels.append(Symel(f"C_5^4({i})", faces[i], matrix_power(C5, 4)))
    for i in range(10):
        C3 = Cn(vertices[i], 3)
        symels.append(Symel(f"C_3({i})", vertices[i], C3))
        symels.append(Symel(f"C_3^2({i})", vertices[i], matrix_power(C3, 2)))
    for i in range(15):
        symels.append(Symel(f"C_2({i})", edgecenters[i], Cn(edgecenters[i], 2)))
    return symels


def _generate_Ih():
    """
    Generate symmetry elements for the Ih point group.

    Returns
    -------
    List[Symel]
    """
    symels = _generate_I()
    faces, vertices, edgecenters = FACE_VEC, VERTEX_VEC, EDGE_VEC
    symels.append(Symel("i", None, inversion_matrix()))
    for i in range(6):
        S10 = Sn(faces[i], 10)
        symels.append(Symel(f"S_10({i})", faces[i], S10))
        symels.append(Symel(f"S_10^3({i})", faces[i], matrix_power(S10, 3)))
        symels.append(Symel(f"S_10^7({i})", faces[i], matrix_power(S10, 7)))
        symels.append(Symel(f"S_10^9({i})", faces[i], matrix_power(S10, 9)))
    for i in range(10):
        S6 = Sn(vertices[i], 6)
        symels.append(Symel(f"S_6({i})", vertices[i], S6))
        symels.append(Symel(f"S_6^5({i})", vertices[i], matrix_power(S6, 5)))
    for i in range(15):
        symels.append(Symel(f"sigma({i})", edgecenters[i], reflection_matrix(edgecenters[i])))
    return symels


# ── Precomputed element tuples (built once at import time) ────────────────────

T_SYMELS  = tuple(_generate_T())
TD_SYMELS = tuple(_generate_Td())
TH_SYMELS = tuple(_generate_Th())
O_SYMELS  = tuple(_generate_O())
OH_SYMELS = tuple(_generate_Oh())
I_SYMELS  = tuple(_generate_I())
IH_SYMELS = tuple(_generate_Ih())
