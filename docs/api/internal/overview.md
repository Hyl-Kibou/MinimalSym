# Internal API Overview

This section documents the internal architecture of MinimalSym.

The internal API implements the full symmetry pipeline, from **point group detection**
to **symmetry-aware geometry transformation** and **candidate generation**.

!!! warning
    These interfaces are not stable and may change between versions.

---

## Architecture Overview

MinimalSym is organized into three conceptual layers:

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

* [pg_detect][minimalsym.core.pg_detect]
* [rotation_detection][minimalsym.core.rotation_detection]
* [reflection_detection][minimalsym.core.reflection_detection]
* [special_geometry][minimalsym.core.special_geometry]

---

### [Symmetry Representation](representation.md)

Encode symmetry in a structured form

* [symel][minimalsym.core.symel]
* [symel_gen][minimalsym.core.symel_gen]
* [point_group][minimalsym.core.point_group]
* [symtext][minimalsym.core.symtext]

---

### [Group Construction](group_construction.md)

Build symmetry elements from group definitions

* [cyclic_dihedral][minimalsym.core.cyclic_dihedral]
* [cubic_icosahedral][minimalsym.core.cubic_icosahedral]
* [group_algebra][minimalsym.core.group_algebra]

---

### [Geometry & Molecular Operations](geometry.md)

Manipulate and analyze molecular coordinates

* [mol_ops][minimalsym.core.mol_ops]
* [mol_orient][minimalsym.core.mol_orient]
* [atom_mapping][minimalsym.core.atom_mapping]

---

### [Decomposition & Candidate Generation](decomposition.md)

Generate compatible subgroups and symmetry candidates

* [pg_decompose][minimalsym.core.pg_decompose]

---

### [Utilities](utilities.md)

Low-level numerical and algebraic helpers

* [sym_ops][minimalsym.core.sym_ops]
* [constants][minimalsym.core.constants]

---

## See Also

* [Public API](../public.md)
* [Data flow diagram](../../flowchart.md)
