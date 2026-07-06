#===== EXAMPLE 1 =====
from ase.build import molecule
from molsympy import get_point_group

ciclopropane = molecule('C3H6_D3h')
print(get_point_group(ciclopropane)) #D3h