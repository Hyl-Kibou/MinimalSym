# Concepts and Conventions

This page introduces the core concepts, assumptions, and numerical
conventions used throughout MolSymPy.

Understanding these is important for correctly interpreting results,
especially when working with symmetry detection and symmetrization.

If you're looking for quick usage examples, start with
[Examples](examples.md).

## Tolerance

### Geometric Tolerance

Symmetry and point group detection utilize a geometric tolerance set in
`Atoms.info["geom_tol"]` with units in Å. Typical value is `0.05`.

This value is used to determine if positional variation is low enough
after symmetry operations to be considered equivalent.
Larger tolerances make the algorithm more permissive when identifying
symmetry operations.

When using `symmetrize()`, `get_point_group()`, `get_inequivalent()`,
the tolerance is set via the `geom_tol` argument. If not specified, the
default value is `0.05`.

### Relative eigen tolerance

!!! warning
    By default, `eigen_tol` is derived from `geom_tol` and the
    molecular size, ensuring that inertia comparisons are consistent
    with the allowed geometric perturbation.
    Be careful when overriding this parameter, as values that are
    inconsistent with the geometric tolerance will result in
    unexpected behaviour.

    In most cases, the automatically estimated value is recommended.

Symmetry and point group detection utilize a relative tolerance set in
`Atoms.info["eigen_tol"]`. Typical value is in the range of `1e-02` and
`1e-05`.

This value is used to compare the relative difference between
eigenvalues (moments of inertia).
Used for molecular classification (linear, spherical top, etc.)
Larger tolerances make the algorithm more permissive when identifying
degeneracies in the moments of inertia, which affects molecular
classification.

When using `symmetrize()`, `get_point_group()`, `get_inequivalent()`
and `generate_symmetry_candidates()`
the tolerance is set via the `eigen_tol` argument. If not specified,
the default value is `None` and a function will automatically try to
determine an adequate float given the input molecule and geometric
tolerance.

The relative eigen tolerance is estimated from the geometric tolerance
by propagating positional uncertainty into the expected relative
variation of the moments of inertia.
Roughly, `eigen_tol ∝ geom_tol / molecular_size`, so larger molecules
require tighter relative tolerances.

---

## Warnings and debug output

When using [`symmetrize()`][molsympy.api.symmetrize], [`get_point_group()`][molsympy.api.get_point_group], [`get_inequivalent()`][molsympy.api.get_inequivalent], and
[`generate_symmetry_candidates`][molsympy.api.generate_symmetry_candidates], warnings and debug output may be triggered.
Printing is controlled by the python logging module.

By default the debug and warning output will be hidden, the user can choose to enable the output by adding this snippet in their calling script.

```python
import logging

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(name)s - %(levelname)s - %(message)s"
))

logger = logging.getLogger("molsympy")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False
```

---

## Core Data Structures

### [PointGroup][molsympy.core.point_group.PointGroup]
::: molsympy.core.point_group.PointGroup
    options:    
        show_root_heading: false
        show_root_toc_entry: false
        backlinks: false

---

### [SymmetryResult][molsympy.api.SymmetryResult]
::: molsympy.api.SymmetryResult
    options:
        show_root_heading: false
        show_root_toc_entry: false
        backlinks: false

---

### [PointGroupResult][molsympy.core.pg_detect.PointGroupResult]
::: molsympy.core.pg_detect.PointGroupResult
    options:
        show_root_heading: false
        show_root_toc_entry: false
        backlinks: false

---

### [Symel][molsympy.core.symel.Symel]

::: molsympy.core.symel.Symel
    options:
        show_root_heading: false
        show_root_toc_entry: false
        backlinks: false

---

### [Symtext][molsympy.core.symtext.Symtext]

::: molsympy.core.symtext.Symtext
    options:
        members: false
        show_root_heading: false
        show_root_toc_entry: false
        backlinks: false


---

### [SEA][molsympy.core.mol_ops.SEA] (Symmetry Equivalent Atoms)

::: molsympy.core.mol_ops.SEA
    options:
        show_root_heading: false
        show_root_toc_entry: false
        backlinks: false

---

## Output metadata

MolSymPy stores symmetry information in the `Atoms.info` dictionary
of the returned `Atoms` object.

| Key | Description |
|-----|-------------|
| `"pg"` | detected point group |
| `"geom_tol"` | positional tolerance |
| `"eigen_tol"` | eigenvalues relative tolerance |

---

See the [API reference](../api/index.md) for full documentation.