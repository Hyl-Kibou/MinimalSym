import numpy as np
from numpy.linalg import matrix_power
from ..symtools import reflection_matrix, inversion_matrix, Cn, Sn, normalize
from dataclasses import dataclass

# New Symel definition!
@dataclass(frozen=True, slots=True)
class Symel():
    """
    Symmetry element: a single symmetry operation with its matrix representation and metadata.
    
    Parameters
    ----------
    symbol: str
        Schoenflies symbol of the symmetry element (e.g. "C_3", "sigma_h", "i").
    vector: np.array or None
        Axis vector for Cn and Sn; plane normal vector for sigma; None for E and i.
    rrep: np.array
        3x3 real-space matrix representation of the operation.
    m: int or None
        Power of the generator (e.g. m=2 for C_3^2).
    n: int or None
        Order of the principal axis of the generating cycle.
    O: str or None
        Origin class of the element. Options: "E", "sigma_v", "C_2'", "i", "sigma_h".
    """
    symbol: str
    vector: np.ndarray | None # Not defined for E or i, axis vector for Cn and Sn, plane normal vector for sigma
    rrep: np.ndarray
    m: int | None = None
    n: int | None = None
    O: str | None = None # str Options: E, sigma_v, C_2', i, sigma_h

    def __str__(self) -> str:
        with np.printoptions(precision=5, suppress=True, formatter={"all":lambda x: f"{x:8.5f}"}):
            return f"\nSymbol: {self.symbol:>10s}: [{self.rrep[0,:]},{self.rrep[1,:]},{self.rrep[2,:]}]"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        return self.symbol == other.symbol and np.isclose(self.rrep, other.rrep, atol=1e-10).all()

# ── Cubic and icosahedral group generators ────────────────────────────────────    

def _generate_T():
    """
    Generate symmetry elements for the T point group.
    Assume a tetrahedron contained in a cube, then we can easily generate
    the vectors for the rotation elements.

    Returns
    -------
    List[Symel]
    """
    # Generate C3's
    symels = [Symel("E", None, np.eye(3))]
    C3_1v = normalize(np.array([1.0, 1.0, 1.0]))
    C3_2v = normalize(np.array([-1.0, 1.0, -1.0]))
    C3_3v = normalize(np.array([-1.0, -1.0, 1.0]))
    C3_4v = normalize(np.array([1.0, -1.0, -1.0]))
    C3list = [C3_1v, C3_2v, C3_3v, C3_4v]
    namelist = ("alpha", "beta", "gamma", "delta")
    for i in range(4):
        C3 = Cn(C3list[i], 3)
        C3s = matrix_power(C3, 2)
        symels.append(Symel(f"C_3({namelist[i]})", C3list[i], C3))
        symels.append(Symel(f"C_3^2({namelist[i]})", C3list[i], C3s))
    # Generate C2's    
    C2list = np.eye(3)
    namelist = ["x", "y", "z"]
    for i in range(3):
        C2 = Cn(C2list[i], 2)
        symels.append(Symel(f"C_2({namelist[i]})", C2list[i], C2))
    return symels

def _generate_Td():
    """
    Generate symmetry elements for the Td point group.
    Assume a tetrahedron contained in a cube, then we can easily generate
    the vectors for the rotation elements.

    Returns
    -------
    List[Symel]
    """
    symels = _generate_T()
    # σd's
    sigmas = [
        normalize(np.array([1.0, 1.0, 0.0])), normalize(np.array([1.0, -1.0, 0.0])),
        normalize(np.array([1.0, 0.0, 1.0])), normalize(np.array([1.0, 0.0, -1.0])),
        normalize(np.array([0.0, 1.0, 1.0])), normalize(np.array([0.0, 1.0, -1.0])),
    ]
    namelist = ["xyp", "xym", "xzp", "xzm", "yzp", "yzm"]
    for i in range(6):
        sigma_d = reflection_matrix(sigmas[i])
        symels.append(Symel(f"sigma_d({namelist[i]})", sigmas[i], sigma_d))
    # S4's    
    S4vlist = np.eye(3)
    namelist = ["x","y","z"]
    for i in range(3):
        S4 = Sn(S4vlist[i], 4)
        S43 = matrix_power(S4, 3)
        symels.append(Symel(f"S_4({namelist[i]})", S4vlist[i], S4))
        symels.append(Symel(f"S_4^3({namelist[i]})", S4vlist[i], S43))
    return symels

def _generate_Th():
    """
    Generate symmetry elements for the Th point group.
    Assume a tetrahedron contained in a cube, then we can easily generate
    the vectors for the rotation elements.

    Returns
    -------
    List[Symel]
    """
    symels = _generate_T()
    # i
    symels.append(Symel("i", None, inversion_matrix()))
    # S6
    S6list = [
        normalize(np.array([1.0, 1.0, 1.0])), normalize(np.array([-1.0, 1.0, -1.0])),
        normalize(np.array([-1.0, -1.0, 1.0])), normalize(np.array([1.0, -1.0, -1.0])),
    ]
    namelist = ["alpha", "beta", "gamma", "delta"]
    for i in range(4):
        S6 = Sn(S6list[i], 6)
        S65 = matrix_power(S6, 5)
        symels.append(Symel(f"S_6({namelist[i]})", S6list[i], S6))
        symels.append(Symel(f"S_6^5({namelist[i]})", S6list[i], S65))
    # 3sigma_h    
    sigma_list = np.eye(3)
    namelist = ["x", "y", "z"]
    for i in range(3):
        sigma_h = reflection_matrix(sigma_list[i])
        symels.append(Symel(f"sigma_h({namelist[i]})", sigma_list[i], sigma_h))
    return symels

def _generate_O():
    """
    Generate symmetry elements for the O point group.
    Assume operations on a cube.

    Returns
    -------
    List[Symel]
    """
    symels = [Symel("E", None, np.eye(3))]
    # C4    
    C4list = np.eye(3)
    namelist = ["x", "y", "z"]
    for i in range(3):
        C4 = Cn(C4list[i], 4)
        C42 = matrix_power(C4, 2)
        C43 = matrix_power(C4, 3)
        symels.append(Symel(f"C_4({namelist[i]})", C4list[i], C4))
        symels.append(Symel(f"C_2({namelist[i]})", C4list[i], C42))
        symels.append(Symel(f"C_4^3({namelist[i]})", C4list[i], C43))
    # C3
    C3list = [
        normalize(np.array([1.0, 1.0, 1.0])), normalize(np.array([1.0, -1.0, 1.0])),
        normalize(np.array([1.0, 1.0, -1.0])), normalize(np.array([1.0, -1.0, -1.0])),
    ]
    namelist = ["alpha", "beta", "gamma", "delta"]
    for i in range(4):
        C3 = Cn(C3list[i], 3)
        C32 = matrix_power(C3, 2)
        symels.append(Symel(f"C_3({namelist[i]})", C3list[i], C3))
        symels.append(Symel(f"C_3^2({namelist[i]})", C3list[i], C32))
    # C2
    C2list = [
        normalize(np.array([1.0, 0.0, 1.0])), normalize(np.array([1.0, 0.0, -1.0])),
        normalize(np.array([1.0, 1.0, 0.0])), normalize(np.array([1.0, -1.0, 0.0])),
        normalize(np.array([0.0, 1.0, 1.0])), normalize(np.array([0.0, -1.0, 1.0])),
    ]
    namelist = ["xzp", "xzm", "xyp", "xym", "yzp", "yzm"]
    for i in range(6):
        C2 = Cn(C2list[i], 2)
        symels.append(Symel(f"C_2({namelist[i]})", C2list[i], C2))
    return symels

def _generate_Oh():
    """
    Generate symmetry elements for the Oh point group.
    Assume operations on a cube.

    Returns
    -------
    List[Symel]
    """
    symels = _generate_O()
    symels.append(Symel("i", None, inversion_matrix()))
    # S4 and σh    
    S4list = np.eye(3)
    namelist = ["x", "y", "z"]
    for i in range(3):
        S4 = Sn(S4list[i], 4)
        sigma_h = reflection_matrix(S4list[i])
        S43 = matrix_power(S4, 3)
        symels.append(Symel(f"S_4({namelist[i]})", S4list[i], S4))
        symels.append(Symel(f"sigma_h({namelist[i]})", S4list[i], sigma_h))
        symels.append(Symel(f"S_4^3({namelist[i]})", S4list[i], S43))
    # S6
    S6list = [
        normalize(np.array([1.0, 1.0, 1.0])), normalize(np.array([1.0, -1.0, 1.0])),
        normalize(np.array([1.0, 1.0, -1.0])), normalize(np.array([1.0, -1.0, -1.0])),
    ]
    namelist = ["alpha", "beta", "gamma", "delta"]
    for i in range(4):
        S6 = Sn(S6list[i], 6)
        S65 = matrix_power(S6, 5)
        symels.append(Symel(f"S_6({namelist[i]})", S6list[i], S6))
        symels.append(Symel(f"S_6^5({namelist[i]})", S6list[i], S65))
    # C2
    sigma_dlist = [
        normalize(np.array([1.0, 0.0, 1.0])), normalize(np.array([1.0, 0.0, -1.0])),
        normalize(np.array([1.0, 1.0, 0.0])), normalize(np.array([1.0, -1.0, 0.0])),
        normalize(np.array([0.0, 1.0, 1.0])), normalize(np.array([0.0, -1.0, 1.0])),
    ]
    namelist = ["xzp", "xzm", "xyp", "xym", "yzp", "yzm"]
    for i in range(6):
        sigma_d = reflection_matrix(sigma_dlist[i])
        symels.append(Symel(f"sigma_d({namelist[i]})", sigma_dlist[i], sigma_d))
    return symels

def _generate_I():
    """
    Generate symmetry elements for the I point group.

    Returns
    -------
    List[Symel]
    """
    symels = [Symel("E", None, np.eye(3))]
    faces, vertices, edgecenters = FACE_VEC, VERTEX_VEC, EDGE_VEC
    # C5 (face vectors)
    for i in range(6):
        C5 = Cn(faces[i],5)
        C52 = matrix_power(C5,2)
        C53 = matrix_power(C5,3)
        C54 = matrix_power(C5,4)
        symels.append(Symel(f"C_5({i})", faces[i], C5))
        symels.append(Symel(f"C_5^2({i})", faces[i], C52))
        symels.append(Symel(f"C_5^3({i})", faces[i], C53))
        symels.append(Symel(f"C_5^4({i})", faces[i], C54))
    
    # C3 (vertex vectors)
    for i in range(10):
        C3 = Cn(vertices[i], 3)
        C32 = matrix_power(C3, 2)
        symels.append(Symel(f"C_3({i})", vertices[i], C3))
        symels.append(Symel(f"C_3^2({i})", vertices[i], C32))

    # C2 (edge vectors)
    for i in range(15):
        C2 = Cn(edgecenters[i], 2)
        symels.append(Symel(f"C_2({i})", edgecenters[i], C2))
    
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
    # S10 (face vectors)
    for i in range(6):
        S10 = Sn(faces[i],10)
        S103 = matrix_power(S10, 3)
        S107 = matrix_power(S10, 7)
        S109 = matrix_power(S10, 9)
        symels.append(Symel(f"S_10({i})", faces[i], S10))
        symels.append(Symel(f"S_10^3({i})", faces[i], S103))
        symels.append(Symel(f"S_10^7({i})", faces[i], S107))
        symels.append(Symel(f"S_10^9({i})", faces[i], S109))

    # S6 (vertex vectors)
    for i in range(10):
        S6 = Sn(vertices[i],6)
        S65 = matrix_power(S6,5)
        symels.append(Symel(f"S_6({i})", vertices[i], S6))
        symels.append(Symel(f"S_6^5({i})", vertices[i], S65))

    # σ (edge vectors)
    for i in range(15):
        sigma_i = reflection_matrix(edgecenters[i])
        symels.append(Symel(f"sigma({i})", edgecenters[i], sigma_i))
    
    return symels

def _generate_I_vectors():
    """
    Vectors defining the faces, vertices, and edges of a regular dodecahedron.

    Returns
    -------
    tuple(List[np.array], List[np.array], List[np.array])
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

    face_vectors = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 2*r5, r5]),
        np.array([-sp10, (5-s5)/10, r5]),
        np.array([ sp10, (5-s5)/10, r5]),
        np.array([-sm10,-(5+s5)/10, r5]),
        np.array([ sm10,-(5+s5)/10, r5])]
    vertex_vectors = [
        np.array([0.0, -np.sqrt((2*(5+s5))/15), vm]),
        np.array([0.0, -np.sqrt((2*(5-s5))/15), vp]),
        np.array([-r3, vp, vm]),
        np.array([-r3,-vm, vp]),
        np.array([ r3, vp, vm]),
        np.array([ r3,-vm, vp]),
        np.array([-t3p,-sm30, vm]),
        np.array([ t3m, sp30, vp]),
        np.array([-t3m, sp30, vp]),
        np.array([ t3p,-sm30, vm]),
    ]
    edgecenters = [
        np.array([0.0, sp10,-sm10]),
        np.array([0.0, sm10, sp10]),
        np.array([0.5,-a1p,-sm10]),
        np.array([0.5, a1m, sp10]),
        np.array([0.5,-a1m,-sp10]),
        np.array([0.5, a1p, sm10]),
        np.array([1.0, 0.0, 0.0]),
        np.array([phim,-sp10/2, sp10]),
        np.array([phim, sp10/2,-sp10]),
        np.array([phim,-sp2h, 0.0]),
        np.array([phim, sp2h, 0.0]),
        np.array([phip,-sm10/2, sm10]),
        np.array([phip, sm10/2,-sm10]),
        np.array([phip,-sm2h, 0.0]),
        np.array([phip, sm2h, 0.0]),
    ]

    return (face_vectors, vertex_vectors, edgecenters)

FACE_VEC, VERTEX_VEC, EDGE_VEC = _generate_I_vectors()

T_SYMELS  = tuple(_generate_T())
TD_SYMELS = tuple(_generate_Td())
TH_SYMELS = tuple(_generate_Th())
O_SYMELS  = tuple(_generate_O())
OH_SYMELS = tuple(_generate_Oh())
I_SYMELS  = tuple(_generate_I())
IH_SYMELS = tuple(_generate_Ih())