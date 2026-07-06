#===== DATABASE ACCESS =====
from molsympy.collections import symmetrized, unsymmetrized

# Lists of molecule IDs
print(unsymmetrized.names)         #['C0v_1', 'C0v_2', …, 'Td_3']
print(symmetrized.names)           #['C0v_1', 'C0v_2', …, 'Td_3']
# Equivalent lists asking for molecular formulas (alphabetical order):
print(unsymmetrized.formulas)      # ['AgC68N4H76O4', 'AuC18P2N6H24', …, 'ZrSi7C24H52']
print(symmetrized.formulas)        # ['AgC68N4H76O4', 'AuC18P2N6H24', …, 'ZrSi7C24H52']
# ASE Atoms objects can be called from their molecule IDs or molecular formulas:
mol1 = symmetrized['C3h_1']
mol2 = symmetrized['C6H6O3']