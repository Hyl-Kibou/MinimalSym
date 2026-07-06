#===== EXAMPLE 4 =====
from ase.build import molecule
from molsympy import generate_symmetry_candidates

benzene = molecule('C6H6')                    # D6h
for s in generate_symmetry_candidates(benzene):
    # Access Atom object with s.mol
    print(s.pg, s.rmsd)                       # D6h 1.6032259480710427e-07, D3h 1.4451283412026997e-07, ...