"""
Core symmetry engine.

This package implements the full machinery for molecular symmetry
detection, classification, and symmetrization. It contains the internal
algorithms and data structures used by the public API.

Scope
-----
- Point group detection (pg_detect)
- Symmetry element generation (symel, symel_gen)
- Group construction and algebra (cyclic_dihedral, cubic_icosahedral, group_algebra)
- Symmetry operation matrices (sym_ops)
- Atom mapping and equivalence (atom_mapping, mol_ops)
- Molecular orientation and geometry handling (mol_orient, special_geometry)
- Point group decomposition (pg_decompose)
- High-level symmetry representation (symtext)

These modules work together to:
1. Detect the point group of a molecule
2. Construct the corresponding symmetry elements
3. Build atom mappings under symmetry operations
4. Apply symmetry constraints to generate symmetrized geometries

Stability
---------
This is an internal package. Its API is not guaranteed to be stable
and may change between releases.

Users should rely on the public interface in `api.py`.

Notes
-----
- Many routines are optimized for numerical robustness and may use
  Numba-compatible implementations.
- Tolerances (geometric and eigenvalue) propagate through most
  algorithms and strongly influence classification behavior.
"""