import numpy as np
from numpy.linalg import matrix_power
from ..symtools import *
from dataclasses import dataclass
from .point_group import PointGroup

@dataclass
class Symel():
    """
    Deprecated data structure for symmetry elements.
    Still hanging around because of cubic/icosahedral groups and tests.
    """
    symbol:str
    vector:np.array # Not defined for E or i, axis vector for Cn and Sn, plane normal vector for sigma
    rrep:np.array
    def __str__(self) -> str:
        with np.printoptions(precision=5, suppress=True, formatter={"all":lambda x: f"{x:8.5f}"}):
            return f"\nSymbol: {self.symbol:>10s}: [{self.rrep[0,:]},{self.rrep[1,:]},{self.rrep[2,:]}]"
    def __repr__(self) -> str:
        return self.__str__()
    def __eq__(self, other):
        return self.symbol == other.symbol and np.isclose(self.rrep,other.rrep,atol=1e-10).all()
    
def generate_T():
    """
    Generate symmetry elements for the T point group.
    Assume a tetrahedron contained in a cube, then we can easily generate
    the vectors for the rotation elements.

    :rtype: List[molsym.Symel]
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
        C3s = matrix_power(C3,2)
        symels.append(Symel(f"C_3({namelist[i]})", C3list[i], C3))
        symels.append(Symel(f"C_3^2({namelist[i]})", C3list[i], C3s))
    # Generate C2's
    C2_x = np.array([1.0, 0.0, 0.0])
    C2_y = np.array([0.0, 1.0, 0.0])
    C2_z = np.array([0.0, 0.0, 1.0])
    C2list = [C2_x, C2_y, C2_z]
    namelist = ["x", "y", "z"]
    for i in range(3):
        C2 = Cn(C2list[i], 2)
        symels.append(Symel(f"C_2({namelist[i]})", C2list[i], C2))
    return symels

def generate_Td():
    """
    Generate symmetry elements for the Td point group.
    Assume a tetrahedron contained in a cube, then we can easily generate
    the vectors for the rotation elements.

    :rtype: List[molsym.Symel]
    """
    symels = generate_T()
    # σd's
    sigma_d_1v = normalize(np.array([1.0, 1.0, 0.0]))
    sigma_d_2v = normalize(np.array([1.0,-1.0, 0.0]))
    sigma_d_3v = normalize(np.array([1.0, 0.0, 1.0]))
    sigma_d_4v = normalize(np.array([1.0, 0.0,-1.0]))
    sigma_d_5v = normalize(np.array([0.0, 1.0, 1.0]))
    sigma_d_6v = normalize(np.array([0.0, 1.0,-1.0]))
    sigmas = [sigma_d_1v,sigma_d_2v,sigma_d_3v,sigma_d_4v,sigma_d_5v,sigma_d_6v]
    namelist = ["xyp","xym","xzp","xzm","yzp","yzm"]
    for i in range(6):
        sigma_d = reflection_matrix(sigmas[i])
        symels.append(Symel(f"sigma_d({namelist[i]})", sigmas[i], sigma_d))
    # S4's
    S4_1v = np.array([1.0, 0.0, 0.0])
    S4_2v = np.array([0.0, 1.0, 0.0])
    S4_3v = np.array([0.0, 0.0, 1.0])
    S4vlist = [S4_1v, S4_2v, S4_3v]
    namelist = ["x","y","z"]
    for i in range(3):
        S4 = Sn(S4vlist[i], 4)
        S43 = matrix_power(S4,3)
        symels.append(Symel(f"S_4({namelist[i]})", S4vlist[i], S4))
        symels.append(Symel(f"S_4^3({namelist[i]})", S4vlist[i], S43))
    return symels

def generate_Th():
    """
    Generate symmetry elements for the Th point group.
    Assume a tetrahedron contained in a cube, then we can easily generate
    the vectors for the rotation elements.

    :rtype: List[molsym.Symel]
    """
    symels = generate_T()
    # i
    symels.append(Symel("i", None, inversion_matrix()))
    # S6
    S6_1v = normalize(np.array([ 1.0, 1.0, 1.0]))
    S6_2v = normalize(np.array([-1.0, 1.0,-1.0]))
    S6_3v = normalize(np.array([-1.0,-1.0, 1.0]))
    S6_4v = normalize(np.array([ 1.0,-1.0,-1.0]))
    S6list = [S6_1v, S6_2v, S6_3v, S6_4v]
    namelist = ["alpha", "beta", "gamma", "delta"]
    for i in range(4):
        S6 = Sn(S6list[i], 6)
        S65 = matrix_power(S6, 5)
        symels.append(Symel(f"S_6({namelist[i]})", S6list[i], S6))
        symels.append(Symel(f"S_6^5({namelist[i]})", S6list[i], S65))
    # 3sigma_h
    sigma_h_xv = np.array([1.0, 0.0, 0.0])
    sigma_h_yv = np.array([0.0, 1.0, 0.0])
    sigma_h_zv = np.array([0.0, 0.0, 1.0])
    sigma_list = [sigma_h_xv,sigma_h_yv,sigma_h_zv]
    namelist = ["x", "y", "z"]
    for i in range(3):
        sigma_h = reflection_matrix(sigma_list[i])
        symels.append(Symel(f"sigma_h({namelist[i]})", sigma_list[i], sigma_h))
    return symels

def generate_O():
    """
    Generate symmetry elements for the O point group.
    Assume operations on a cube.

    :rtype: List[molsym.Symel]
    """
    symels = [Symel("E", None, np.eye(3))]
    # C4
    C4_xv = np.array([1.0, 0.0, 0.0])
    C4_yv = np.array([0.0, 1.0, 0.0])
    C4_zv = np.array([0.0, 0.0, 1.0])
    C4list = [C4_xv, C4_yv, C4_zv]
    namelist = ["x", "y", "z"]
    for i in range(3):
        C4 = Cn(C4list[i], 4)
        C42 = matrix_power(C4,2)
        C43 = matrix_power(C4,3)
        symels.append(Symel(f"C_4({namelist[i]})", C4list[i], C4))
        symels.append(Symel(f"C_2({namelist[i]})", C4list[i], C42))
        symels.append(Symel(f"C_4^3({namelist[i]})", C4list[i], C43))
    # C3
    C3_1v = normalize(np.array([1.0, 1.0, 1.0]))
    C3_2v = normalize(np.array([1.0,-1.0, 1.0]))
    C3_3v = normalize(np.array([1.0, 1.0,-1.0]))
    C3_4v = normalize(np.array([1.0,-1.0,-1.0]))
    C3list = [C3_1v, C3_2v, C3_3v, C3_4v]
    namelist = ["alpha", "beta", "gamma", "delta"]
    for i in range(4):
        C3 = Cn(C3list[i], 3)
        C32 = matrix_power(C3,2)
        symels.append(Symel(f"C_3({namelist[i]})", C3list[i], C3))
        symels.append(Symel(f"C_3^2({namelist[i]})", C3list[i], C32))
    # C2
    C2_1v = normalize(np.array([1.0, 0.0, 1.0]))
    C2_2v = normalize(np.array([1.0, 0.0,-1.0]))
    C2_3v = normalize(np.array([1.0, 1.0, 0.0]))
    C2_4v = normalize(np.array([1.0,-1.0, 0.0]))
    C2_5v = normalize(np.array([0.0, 1.0, 1.0]))
    C2_6v = normalize(np.array([0.0,-1.0, 1.0]))
    
    C2list = [C2_1v, C2_2v, C2_3v, C2_4v, C2_5v, C2_6v]
    namelist = ["xzp", "xzm", "xyp", "xym", "yzp", "yzm"]
    for i in range(6):
        C2 = Cn(C2list[i],2)
        symels.append(Symel(f"C_2({namelist[i]})", C2list[i], C2))
    return symels

def generate_Oh():
    """
    Generate symmetry elements for the Oh point group.
    Assume operations on a cube.

    :rtype: List[molsym.Symel]
    """
    symels = generate_O()
    symels.append(Symel("i", None, inversion_matrix()))
    # S4 and σh
    S4_xv = np.array([1.0, 0.0, 0.0])
    S4_yv = np.array([0.0, 1.0, 0.0])
    S4_zv = np.array([0.0, 0.0, 1.0])
    S4list = [S4_xv, S4_yv, S4_zv]
    namelist = ["x", "y", "z"]
    for i in range(3):
        S4 = Sn(S4list[i], 4)
        sigma_h = reflection_matrix(S4list[i])
        S43 = matrix_power(S4,3)
        symels.append(Symel(f"S_4({namelist[i]})", S4list[i], S4))
        symels.append(Symel(f"sigma_h({namelist[i]})", S4list[i], sigma_h))
        symels.append(Symel(f"S_4^3({namelist[i]})", S4list[i], S43))
    # S6
    S6_1v = normalize(np.array([1.0, 1.0, 1.0]))
    S6_2v = normalize(np.array([1.0,-1.0, 1.0]))
    S6_3v = normalize(np.array([1.0, 1.0,-1.0]))
    S6_4v = normalize(np.array([1.0,-1.0,-1.0]))
    S6list = [S6_1v, S6_2v, S6_3v, S6_4v]
    namelist = ["alpha", "beta", "gamma", "delta"]
    for i in range(4):
        S6 = Sn(S6list[i], 6)
        S65 = matrix_power(S6,5)
        symels.append(Symel(f"S_6({namelist[i]})", S6list[i], S6))
        symels.append(Symel(f"S_6^5({namelist[i]})", S6list[i], S65))
    # C2
    sigma_d_1v = normalize(np.array([1.0, 0.0, 1.0]))
    sigma_d_2v = normalize(np.array([1.0, 0.0,-1.0]))
    sigma_d_3v = normalize(np.array([1.0, 1.0, 0.0]))
    sigma_d_4v = normalize(np.array([1.0,-1.0, 0.0]))
    sigma_d_5v = normalize(np.array([0.0, 1.0, 1.0]))
    sigma_d_6v = normalize(np.array([0.0,-1.0, 1.0]))
    
    sigma_dlist = [sigma_d_1v, sigma_d_2v, sigma_d_3v, sigma_d_4v, sigma_d_5v, sigma_d_6v]
    namelist = ["xzp", "xzm", "xyp", "xym", "yzp", "yzm"]
    for i in range(6):
        sigma_d = reflection_matrix(sigma_dlist[i])
        symels.append(Symel(f"sigma_d({namelist[i]})", sigma_dlist[i], sigma_d))
    return symels

def generate_I():
    """
    Generate symmetry elements for the I point group.

    :rtype: List[molsym.Symel]
    """
    symels = [Symel("E", None, np.eye(3))]
    faces, vertices, edgecenters = generate_I_vectors()
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
        C3 = Cn(vertices[i],3)
        C32 = matrix_power(C3,2)
        symels.append(Symel(f"C_3({i})", vertices[i], C3))
        symels.append(Symel(f"C_3^2({i})", vertices[i], C32))

    # C2 (edge vectors)
    for i in range(15):
        C2 = Cn(edgecenters[i],2)
        symels.append(Symel(f"C_2({i})", edgecenters[i], C2))
    
    return symels

def generate_Ih():
    """
    Generate symmetry elements for the Ih point group.

    :rtype: List[molsym.Symel]
    """
    symels = generate_I()
    faces, vertices, edgecenters = generate_I_vectors()
    symels.append(Symel("i", None, inversion_matrix()))
    # S10 (face vectors)
    for i in range(6):
        S10 = Sn(faces[i],10)
        S103 = matrix_power(S10,3)
        S107 = matrix_power(S10,7)
        S109 = matrix_power(S10,9)
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

def generate_I_vectors():
    """
    Vectors defining the faces, vertices, and edges of a regular dodecahedron.

    :rtype: (List[NumPy array of shape (3,)], List[NumPy array of shape (3,)], List[NumPy array of shape (3,)])
    """
    face_vectors = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 2/np.sqrt(5), 1/np.sqrt(5)]),
        np.array([-np.sqrt((5+np.sqrt(5))/10), (5-np.sqrt(5))/10, 1/np.sqrt(5)]),
        np.array([ np.sqrt((5+np.sqrt(5))/10), (5-np.sqrt(5))/10, 1/np.sqrt(5)]),
        np.array([-np.sqrt((5-np.sqrt(5))/10),-(5+np.sqrt(5))/10, 1/np.sqrt(5)]),
        np.array([ np.sqrt((5-np.sqrt(5))/10),-(5+np.sqrt(5))/10, 1/np.sqrt(5)])]
    vertex_vectors = [
        np.array([0.0, -np.sqrt((2*(5+np.sqrt(5)))/15), np.sqrt((5-2*np.sqrt(5))/15)]),
        np.array([0.0, -np.sqrt((2*(5-np.sqrt(5)))/15), np.sqrt((5+2*np.sqrt(5))/15)]),
        np.array([-1/np.sqrt(3), np.sqrt((5+2*np.sqrt(5))/15), np.sqrt((5-2*np.sqrt(5))/15)]),
        np.array([-1/np.sqrt(3),-np.sqrt((5-2*np.sqrt(5))/15), np.sqrt((5+2*np.sqrt(5))/15)]),
        np.array([ 1/np.sqrt(3), np.sqrt((5+2*np.sqrt(5))/15), np.sqrt((5-2*np.sqrt(5))/15)]),
        np.array([ 1/np.sqrt(3),-np.sqrt((5-2*np.sqrt(5))/15), np.sqrt((5+2*np.sqrt(5))/15)]),
        np.array([-np.sqrt((3*np.sqrt(5)+5)/(6*np.sqrt(5))),-np.sqrt((5-np.sqrt(5))/30), np.sqrt((5-2*np.sqrt(5))/15)]),
        np.array([ np.sqrt((3*np.sqrt(5)-5)/(6*np.sqrt(5))), np.sqrt((5+np.sqrt(5))/30), np.sqrt((5+2*np.sqrt(5))/15)]),
        np.array([-np.sqrt((3*np.sqrt(5)-5)/(6*np.sqrt(5))), np.sqrt((5+np.sqrt(5))/30), np.sqrt((5+2*np.sqrt(5))/15)]),
        np.array([ np.sqrt((3*np.sqrt(5)+5)/(6*np.sqrt(5))),-np.sqrt((5-np.sqrt(5))/30), np.sqrt((5-2*np.sqrt(5))/15)]),
    ]
    edgecenters = [
        np.array([0.0, np.sqrt((5+np.sqrt(5))/10),-np.sqrt((5-np.sqrt(5))/10)]),
        np.array([0.0, np.sqrt((5-np.sqrt(5))/10), np.sqrt((5+np.sqrt(5))/10)]),
        np.array([0.5,-np.sqrt(1+(2/np.sqrt(5)))/2,-np.sqrt((5-np.sqrt(5))/10)]),
        np.array([0.5, np.sqrt(1-(2/np.sqrt(5)))/2, np.sqrt((5+np.sqrt(5))/10)]),
        np.array([0.5,-np.sqrt(1-(2/np.sqrt(5)))/2,-np.sqrt((5+np.sqrt(5))/10)]),
        np.array([0.5, np.sqrt(1+(2/np.sqrt(5)))/2, np.sqrt((5-np.sqrt(5))/10)]),
        np.array([1.0, 0.0, 0.0]),
        np.array([(np.sqrt(5)-1)/4,-np.sqrt((5+np.sqrt(5))/10)/2, np.sqrt((5+np.sqrt(5))/10)]),
        np.array([(np.sqrt(5)-1)/4, np.sqrt((5+np.sqrt(5))/10)/2,-np.sqrt((5+np.sqrt(5))/10)]),
        np.array([(np.sqrt(5)-1)/4,-np.sqrt((5+np.sqrt(5))/2)/2, 0.0]),
        np.array([(np.sqrt(5)-1)/4, np.sqrt((5+np.sqrt(5))/2)/2, 0.0]),
        np.array([(np.sqrt(5)+1)/4,-np.sqrt((5-np.sqrt(5))/10)/2, np.sqrt((5-np.sqrt(5))/10)]),
        np.array([(np.sqrt(5)+1)/4, np.sqrt((5-np.sqrt(5))/10)/2,-np.sqrt((5-np.sqrt(5))/10)]),
        np.array([(np.sqrt(5)+1)/4,-np.sqrt((5-np.sqrt(5))/2)/2, 0.0]),
        np.array([(np.sqrt(5)+1)/4, np.sqrt((5-np.sqrt(5))/2)/2, 0.0]),
    ]

    return (face_vectors, vertex_vectors, edgecenters)

