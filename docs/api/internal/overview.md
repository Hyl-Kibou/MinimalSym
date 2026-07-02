# Internal API Overview

This section documents the internal architecture of MolSymPy.

The internal API implements the full symmetry pipeline, from **point group detection**
to **symmetry-aware geometry transformation** and **candidate generation**.

!!! warning
    These interfaces are not stable and may change between versions.

---

## Architecture Overview

MolSymPy is organized into three conceptual layers:

* [Public API](../public.md)
* [API Helpers (validation, symmetrization workflow)](api_helpers.md)
* [Core Engine (symmetry algorithms)](#module-organization)


## Module Organization

Internal modules are grouped by functionality:


### [API Helpers](api_helpers.md)

Bridge public API and core engine

* Input validation and tolerance handling
* Symmetrization workflow
* Candidate evaluation

See: [API helpers](api_helpers.md)

---

### [Symmetry Detection](detection.md)

Determine the point group of a molecule

* [pg_detect][molsympy.core.pg_detect]
* [rotation_detection][molsympy.core.rotation_detection]
* [reflection_detection][molsympy.core.reflection_detection]
* [special_geometry][molsympy.core.special_geometry]

---

### [Symmetry Representation](representation.md)

Encode symmetry in a structured form

* [symel][molsympy.core.symel]
* [symel_gen][molsympy.core.symel_gen]
* [point_group][molsympy.core.point_group]
* [symtext][molsympy.core.symtext]

---

### [Group Construction](group_construction.md)

Build symmetry elements from group definitions

* [cyclic_dihedral][molsympy.core.cyclic_dihedral]
* [cubic_icosahedral][molsympy.core.cubic_icosahedral]
* [group_algebra][molsympy.core.group_algebra]

---

### [Geometry & Molecular Operations](geometry.md)

Manipulate and analyze molecular coordinates

* [mol_ops][molsympy.core.mol_ops]
* [mol_orient][molsympy.core.mol_orient]
* [atom_mapping][molsympy.core.atom_mapping]

---

### [Decomposition & Candidate Generation](decomposition.md)

Generate compatible subgroups and symmetry candidates

* [pg_decompose][molsympy.core.pg_decompose]

---

### [Utilities](utilities.md)

Low-level numerical and algebraic helpers

* [sym_ops][molsympy.core.sym_ops]
* [constants][molsympy.core.constants]

---

## See Also

* [Public API](../public.md)
* [Data flow diagram](../../flowchart.md)
