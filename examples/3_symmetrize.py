#===== EXAMPLE 3 =====
from molsympy.collections import unsymmetrized
from molsympy import symmetrize
atoms_U = unsymmetrized['C0v_1']     # unsymmetrized ase.Atoms
print(atoms_U.positions)             # [[-1.332400e-02  1.132466e+00  8.276000e-03]
                                     #  [ 2.311000e-03 -1.915900e-02  1.929000e-03]
                                     #  [-2.780300e-02  2.198949e+00  1.415400e-02]]
atoms_S = symmetrize(atoms_U)        # symmetrize ase.Atoms
print(atoms_S.positions)             # [[ 0.          0.         -0.55714544]
                                     #  [ 0.          0.          0.59460318]
                                     #  [ 0.          0.         -1.62374292]]