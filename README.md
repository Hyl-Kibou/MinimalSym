# MinimalSym

**MinimalSym** is a Python package for handling molecular symmetry in **ASE** `Atoms` objects.

---

## Features

* **Point group detection:** Detect common point groups for molecules based on their geometry.
* **Molecule symmetrization:** Apply symmetry operations to molecules, aligning them to the detected point group.
* **Get symmetry-inequivalent points:** Atoms are grouped if any symmetry operation (proper or improper) maps one onto the other.
* **ASE Atoms-native workflow:** Directly integrates with ASE Atoms objects, enabling smooth use in existing workflows.

---

## Supported point groups

Point groups are classifications of molecules based on their symmetry operations.
MinimalSym can detect the following common molecular point groups:

- C₁, Cₛ, Cᵢ
- Cₙ, Cₙᵥ, Cₙₕ
- Dₙ, Dₙₕ, Dₙd
- Sₙ
- T, T_h, T_d
- O, O_h
- I, I_h
- C0v, D0h

Detection depends on the symmetry present in the input geometry
and the tolerance used during symmetry detection.

---

## Installation

MinimalSym is tested with **Python 3.12–3.13**, but should also work with **Python 3.9–3.13**.

```bash
pip install minimalsym
```

---

## Quick Examples

### Symmetrize a molecule's geometry and detect its point group

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
mol_symmetric = symmetrize(mol, asym_tol=0.05)

## Check output
print("Detected Point group: ", mol_symmetric.info["pg"])
# Example output: "Detected Point group: C2v"
# Symmetrized positions:
# [[ 0.     0.     0.066]
#  [-0.    -0.757 -0.521]
#  [ 0.     0.757 -0.521]]
```

---

### Detecting a point group

An ASE `Atoms` object is passed to the function and a `string` with the
point group of the molecule is returned.

```python
from minimalsym import get_point_group

## Detect point group for molecule
pg_str = get_point_group(mol, asym_tol=0.05)

## Check output
print("Detected Point group: ", pg_str)
# Example output: "Detected Point group: C2v"
```

---

### Checking planarity

An ASE `Atoms` object is passed to the function and a `bool` is returned.
True if the molecule passed has planarity.

```python
from minimalsym import is_planar

## Check planarity for molecule
mol_is_planar = is_planar(mol, tol=0.05)

## Check output
print("Mol is planar: ", mol_is_planar)
# Example output: "Mol is planar: True"
```

---

### Get symmetry-inequivalent atoms

Find symmetry-inequivalent atoms using all symmetry operations.

Two atoms are in the same equivalence class if any symmetry operation (proper or improper) maps one onto the other.

An ASE `Atoms` object is passed to the function and a `tuple` is returned.

`tuple(unique, parent)`

&nbsp; &nbsp; `unique` : sorted representative atom indices (one per class). 

&nbsp; &nbsp; `parent` : `parent[i]` is the representative of atom `i`.

```python
from minimalsym import get_inequivalent

## Get symmetry-inequivalent atoms for molecule
atom_indices_list, representatives = get_inequivalent(mol, asym_tol=0.3)

## Check output
print("Inequivalent indices list: ", atom_indices_list)
# Example output: "Inequivalent indices list: [0 1 2]"

print("Representatives: ", representatives)
# Example output: "Inequivalent indices list: [0 1 2 0 0 1 1 1 1 2 2 1]"
```

---

## Acknowledgments

This package is based on and inspired by
[NASymmetry / MolSym](https://github.com/NASymmetry/MolSym),
modified to focus on core symmetry functionality.

---

## License

Distributed under the MIT License. See `LICENSE` for details.
