
All molecule objects are expected to be ASE `Atoms` objects. 

These functions do **not modify the input molecule**.

!!! note
    These examples use default tolerances and standard conventions.
    For details on geometric tolerance, eigenvalue tolerance, and
    internal data structures, see [Concepts](concepts.md).

## Symmetrize a molecule's geometry and detect its point group

An ASE `Atoms` object is passed to the function and a new symmetrized
`Atoms` object is returned.

The returned object includes metadata such as the detected point group.

```python hl_lines="18"
from ase import Atoms
import numpy as np
from molsympy import symmetrize

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

For a deeper understanding of how symmetrization works internally,
see the [Symmetrize Algorithm Flowchart](../flowchart.md#symmetrize).

See: [`symmetrize` API][molsympy.api.symmetrize]

---

## Generate alternative symmetry candidates

An ASE `Atoms` object is passed to the function and a list of [SymmetryResult][molsympy.api.SymmetryResult] is returned.

The data in [SymmetryResult][molsympy.api.SymmetryResult] can be accessed through the attributes:

* `mol`
* `pg`
* `rmsd`

```python hl_lines="4"
from molsympy import generate_symmetry_candidates

## Get candidate symetries of mol
candidate_symmetries = generate_symmetry_candidates(mol, geom_tol=0.5)

for elem in candidate_symmetries:
    print(f"Symmetrized molecule: {elem.mol}")
    print(f"Point group: {elem.pg}")
    print(f"RMSD: {elem.rmsd}")

    # Example output:
    # Symmetrized molecule: Atoms(symbols='B12', pbc=False)
    # Point group: C3v
    # RMSD: 0.00011057236018531259

    # Symmetrized molecule: Atoms(symbols='B12', pbc=False)
    # Point group: C3
    # ...
```

See: [`generate_symmetry_candidates` API][molsympy.api.generate_symmetry_candidates]

---

## Detecting a point group

An ASE `Atoms` object is passed to the function and a `string` with the
point group of the molecule is returned.

```python hl_lines="4"
from molsympy import get_point_group

## Detect point group for molecule
pg_str = get_point_group(mol, geom_tol=0.05)

## Check output
print("Detected Point group: ", pg_str)
# Example output: "Detected Point group: C2v"
```

See: [`get_point_group` API][molsympy.api.get_point_group]

---

## Checking planarity

An ASE `Atoms` object is passed to the function and a `bool` is returned.
True if the molecule passed has planarity.

```python hl_lines="4"
from molsympy import is_planar

## Check planarity for molecule
mol_is_planar = is_planar(mol, geom_tol=0.05)

## Check output
print("Mol is planar: ", mol_is_planar)
# Example output: "Mol is planar: True"
```

See: [`is_planar` API][molsympy.api.is_planar]

---

## Get symmetry-inequivalent atoms

Find symmetry-inequivalent atoms using all symmetry operations.

Two atoms are in the same equivalence class if any symmetry operation (proper or improper) maps one onto the other.

An ASE `Atoms` object is passed to the function and a `tuple` is returned.

Returns: `(unique, parent)`

&nbsp; &nbsp; `unique` : sorted representative atom indices (one per equivalence class).

&nbsp; &nbsp; `parent` : array where `parent[i]` gives the representative of atom `i`

```python hl_lines="4"
from molsympy import get_inequivalent

## Get symmetry-inequivalent atoms for molecule
atom_indices_list, parent_mapping = get_inequivalent(mol, geom_tol=0.3)

## Check output
print("Inequivalent indices list: ", atom_indices_list)
# Example output: "Inequivalent indices list: [0 1 2]"

print("Parent mapping: ", parent_mapping)
# Example output: Parent mapping: [0 1 2 0 0 1 1 1 1 2 2 1]"
```

See: [`get_inequivalent` API][molsympy.api.get_inequivalent]

---

See the [API Public Reference](../api/public.md) for full documentation.