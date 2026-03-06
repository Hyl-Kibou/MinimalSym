# MinimalSym

**MinimalSym** is a Python package for handling molecular symmetry in **ASE** `Atoms` objects.
It is a minimal, modified version of [MolSym](https://github.com/NASymmetry/MolSym), focused on core symmetry functionality.

---

## Features

* Point group detection
* Molecule symmetrization
* ASE Atoms-native workflow

---

## Installation

MinimalSym is tested with **Python 3.12–3.13**, but should also work with **Python 3.9–3.13**.

```bash
pip install minimalsym
```

---

## Quick Example

Symmetrizing a water molecule and detecting its point group:

```python
from ase import Atoms
import numpy as np
from minimalsym import symmetrize

theta = np.radians(104.5)

positions = np.array([
    [0.0, 0.0, 0.0],
    [0.958, 0.0, 0.0],
    [0.958 * np.cos(theta), 0.958 * np.sin(theta), 0.0]
])

mol = Atoms(symbols=["O", "H", "H"], positions=positions)

mol_symmetric = symmetrize(mol, asym_tol=0.05)

print("Detected Point group: ", mol_symmetric.info["pg"])
```
---
## API Overview

### `symmetrize(atoms, asym_tol=0.05)`

Returns a symmetrized ASE `Atoms` object and determines its molecular point group.

**Parameters**

* `atoms` (`ase.Atoms`): Molecule to symmetrize
* `asym_tol` (`float`): Asymmetry tolerance (Å)

**Returns**

* `ase.Atoms`: Symmetrized molecule

  * Detected point group stored in `atoms.info["pg"]`

---

## Acknowledgments

This package is based on and inspired by
[NASymmetry / MolSym](https://github.com/NASymmetry/MolSym).

---

## License

Distributed under the MIT License. See `LICENSE` for details.
