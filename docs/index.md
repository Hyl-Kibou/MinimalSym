# MolSymPy

![PyPI](https://img.shields.io/pypi/v/molsympy)
![Python](https://img.shields.io/pypi/pyversions/molsympy)
![License](https://img.shields.io/badge/license-MIT-green)

**MolSymPy** is a Python package for molecular symmetry analysis in **Atomic Simulation Environment** **(ASE)** `Atoms` objects, including structure symmetrization.
It provides fast, geometry-based symmetry detection and manipulation,
designed to integrate seamlessly into existing ASE workflows.

Internally, MolSymPy constructs symmetry operations and atom mappings,
then projects atomic positions onto symmetry elements to enforce exact symmetry.

---

## Installation

```bash
pip install molsympy
```
See the [Installation](installation.md) page for details.

---

## Getting started

MolSymPy is a Python package that provides essential symmetry 
functionality for molecules represented as ASE `Atoms` objects.

Follow the steps below to quickly set up and use MolSymPy with ASE.

### Install the necessary libraries
```bash
pip install molsympy ase
```

### Symmetrize a molecule
```python
from ase.build import molecule
from molsympy import symmetrize

## Create Atoms object.
mol = molecule("H2O")

## Symmetrize molecule.
mol_symmetric = symmetrize(mol)

## Check output
print(mol_symmetric.info["pg"])
```

For further examples and details on how to use MolSymPy,
see the [Usage](usage/examples.md) and [API](api/index.md) pages.

---

## Features

* **Point group detection:** Detect common point groups for molecules based on their geometry.
* **Molecule symmetrization:** Apply symmetry operations to molecules, aligning them to the detected point group.
* **Find symmetry-inequivalent atoms:** Atoms are grouped if any symmetry operation (proper or improper) maps one onto the other.
* **ASE Atoms-native workflow:** Directly integrates with ASE Atoms objects, enabling smooth use in existing workflows.

---

## Scope and limitations

- Works on **finite molecules** (no periodic structure support)
- Uses **geometric tolerance-based** symmetry detection
- Results depend on the chosen `geom_tol`

---

## Supported point groups

Point groups are classifications of molecules based on their symmetry operations.
MolSymPy can detect the following common molecular point groups:

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

## API Overview

- `symmetrize(mol, geom_tol=...) -> Atoms`
- `get_point_group(mol, geom_tol=...) -> str`
- `is_planar(mol, geom_tol=...) -> bool`
- `get_inequivalent(mol, geom_tol=...) -> (unique, parent)`

See the full [API Reference](api/index.md).

---

## Acknowledgments

This package is based on and inspired by
[NASymmetry / MolSym](https://github.com/NASymmetry/MolSym),
modified to focus on core symmetry functionality.

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

## Contributing

If you find MolSymPy useful or have suggestions for improvement,
please feel free to open an issue or pull request on 
[GitHub](https://github.com/hyl-kibou/molsympy).