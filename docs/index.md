# MinimalSym

![PyPI](https://img.shields.io/pypi/v/minimalsym)
![Python](https://img.shields.io/pypi/pyversions/minimalsym)
![License](https://img.shields.io/badge/license-MIT-blue)

**MinimalSym** is a Python package for handling molecular symmetry in **ASE** `Atoms` objects.
It is a minimal, modified version of [MolSym](https://github.com/NASymmetry/MolSym), focused on core symmetry functionality.

---

## Features

* Point group detection: Detect common point groups for molecules based on their geometry.
* Molecule symmetrization: Apply symmetry operations to molecules, aligning them to the detected point group.
* ASE Atoms-native workflow: Directly integrates with ASE Atoms objects, enabling smooth use in existing workflows.

---

## Supported point groups

Point groups are classifications of molecules based on their symmetry operations.
MinimalSym can detect the following common molecular point groups:

- C₁, Cₛ, Cᵢ
- Cₙ, Cₙᵥ, Cₙₕ,
- Dₙ, Dₙₕ, Dₙd
- Sₙ
- I_h, T_h, T_d, O_h
- C0v, D0h

Detection depends on the symmetry present in the input geometry
and the tolerance used during symmetry detection.

---

## Installation

```bash
pip install minimalsym
```
See the [Installation](installation.md) page for details.

---

## Getting started

MinimalSym is a Python package that provides essential symmetry 
functionality for molecules represented as ASE `Atoms` objects.

Follow the steps below to quickly set up and use MinimalSym with ASE.

!!! note
    MinimalSym does not require ASE as a dependency,
    but public functions operate on `Atoms` objects from ASE. 
    To use the examples below, ASE must be installed.

### Install the necessary libraries
```bash
pip install minimalsym ase
```

### Symmetrize a molecule
```python
from ase.build import molecule
from minimalsym import symmetrize

## Create Atoms object.
mol = molecule("H2O")

## Symmetrize molecule.
mol_symmetric = symmetrize(mol)

## Check output
print(mol_symmetric.info["pg"])
```

For further examples and details on how to use MinimalSym,
see the [Usage](usage.md) and [API](api/index.md) pages.

---

## API

See the full [API Reference](api/index.md).

---

## Acknowledgments

This package is based on and inspired by
[NASymmetry / MolSym](https://github.com/NASymmetry/MolSym).

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

## Contributing

If you find MinimalSym useful or have suggestions for improvement,
please feel free to open an issue or pull request on 
[GitHub](https://github.com/yourusername/minimalsym).