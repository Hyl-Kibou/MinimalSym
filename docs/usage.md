# Usage

All molecule objects are expected to be ASE `Atoms` objects. 

!!! note
    MinimalSym does not require ASE as a dependency,
    but public functions operate on `Atoms` objects from ASE. 
    To use the examples below, ASE must be installed.

These functions do **not modify the input molecule**.

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

When using `symmetrize()`, `get_point_group()`, `get_inequivalent()`,
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

## Symmetrize a molecule's geometry and detect its point group

An ASE `Atoms` object is passed to the function and a new symmetrized
`Atoms` object is returned.

```python
from ase import Atoms
import numpy as np
from minimalsym import symmetrize

## Set positions for molecule
theta = np.radians(104.5)

positions = np.array([
    [0.0, 0.0, 0.0],
    [0.958, 0.0, 0.0],
    [0.958 * np.cos(theta), 0.958 * np.sin(theta), 0.0]
])

## Create Atoms object
mol = Atoms(symbols=["O", "H", "H"], positions=positions)

## Symmetrize molecule
mol_symmetric = symmetrize(mol, geom_tol=0.05)

## Check output
print("Detected Point group: ", mol_symmetric.info["pg"])
# Example output: "Detected Point group: C2v"
# Symmetrized positions:
# [[ 0.     0.     0.066]
#  [-0.    -0.757 -0.521]
#  [ 0.     0.757 -0.521]]
```

---

## Output metadata

MinimalSym stores symmetry information in the `Atoms.info` dictionary
of the returned `Atoms` object.

| Key | Description |
|-----|-------------|
| `"pg"` | detected point group |
| `"geom_tol"` | positional tolerance |
| `"eigen_tol"` | eigenvalues relative tolerance |

---

## Detecting a point group

An ASE `Atoms` object is passed to the function and a `string` with the
point group of the molecule is returned.

```python
from minimalsym import get_point_group

## Detect point group for molecule
pg_str = get_point_group(mol, geom_tol=0.05)

## Check output
print("Detected Point group: ", pg_str)
# Example output: "Detected Point group: C2v"
```

---

## Checking planarity

An ASE `Atoms` object is passed to the function and a `bool` is returned.
True if the molecule passed has planarity.

```python
from minimalsym import is_planar

## Check planarity for molecule
mol_is_planar = is_planar(mol, geom_tol=0.05)

## Check output
print("Mol is planar: ", mol_is_planar)
# Example output: "Mol is planar: True"
```

---

## Get symmetry-inequivalent atoms

Find symmetry-inequivalent atoms using all symmetry operations.

Two atoms are in the same equivalence class if any symmetry operation (proper or improper) maps one onto the other.

An ASE `Atoms` object is passed to the function and a `tuple` is returned.

`tuple(unique, parent)`

&nbsp; &nbsp; `unique` : sorted representative atom indices (one per class). 

&nbsp; &nbsp; `parent` : `parent[i]` is the representative of atom `i`.

```python
from minimalsym import get_inequivalent

## Get symmetry-inequivalent atoms for molecule
atom_indices_list, representatives = get_inequivalent(mol, geom_tol=0.3)

## Check output
print("Inequivalent indices list: ", atom_indices_list)
# Example output: "Inequivalent indices list: [0 1 2]"

print("Representatives: ", representatives)
# Example output: "Inequivalent indices list: [0 1 2 0 0 1 1 1 1 2 2 1]"
```

---

See the [API reference](api/public.md) for full documentation.