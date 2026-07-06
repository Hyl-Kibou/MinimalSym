#===== EXAMPLE 2 =====
from molsympy.collections import symmetrized
from molsympy import get_inequivalent

mol = symmetrized['C3h_1']
unique, parent = get_inequivalent(mol, geom_tol=0.05, eigen_tol=0.01)
print(unique)    # [0, 1, 2, 9, 10]
print(parent)    # [0, 1, 2, 1, 0, 2, 1, 0, 2, 9, 10, 9, 10, 9, 10]