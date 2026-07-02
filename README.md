# MolSymPy

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI](https://img.shields.io/pypi/v/molsympy)

**MolSymPy** is an open-source Python package for molecular symmetry analysis in atomistic simulations. It is built natively on top of the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) and operates directly on `ase.Atoms` objects, enabling seamless integration with atomistic simulation workflows.

## Features

- **Geometric idealization** — project nearly-symmetric structures onto their exact symmetry elements and orient the molecular frame by aligning principal and secondary symmetry axes with the Cartesian coordinate system.
- **Point group detection** — identify molecular point groups in Schoenflies notation, including the two infinite-order linear groups (C∞v and D∞h).
- **Symmetry-inequivalent atoms** — determine symmetry-inequivalent atoms using a path-compressed union-find algorithm operating on the complete symmetry permutation map.
- **Reference database** — companion collection of molecular and atomic-cluster geometries spanning the principal point groups, available in both raw and idealized forms.

Performance-critical routines are accelerated through [Numba](https://numba.readthedocs.io/) just-in-time compilation, providing efficient execution with no compilation requirements at install time.

## Installation

```bash
pip install molsympy
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24, Numba ≥ 0.61, ASE ≥ 3.22.

## Quick start

### Point group detection

```python
from ase.build import molecule
from molsympy import get_point_group

cyclopropane = molecule('C3H6_D3h')
print(get_point_group(cyclopropane))  # D3h
```

### Symmetry-inequivalent atoms

```python
from molsympy.collections import symmetrized
from molsympy import get_inequivalent

mol = symmetrized['C3h_1']
unique, parent = get_inequivalent(mol, geom_tol=0.05, eigen_tol=0.01)
print(unique)   # [0, 1, 2, 9, 10]
print(parent)   # [0, 1, 2, 1, 0, 2, 1, 0, 2, 9, 10, 9, 10, 9, 10]
```

### Symmetrization

```python
from molsympy.collections import unsymmetrized
from molsympy import symmetrize, get_point_group

atoms = unsymmetrized['C0v_1']   # by point-group key
atoms = unsymmetrized['CNH']     # equivalent: lookup by molecular formula
print(atoms.positions)

sym = symmetrize(atoms)
print(sym.positions)          # Lst of atomic coordinates
print(get_point_group(sym))   # C0v
```

### Symmetry candidates (subgroup fan-out)

```python
from ase.build import molecule
from molsympy import generate_symmetry_candidates

benzene = molecule('C6H6')                      # D6h
for s in generate_symmetry_candidates(benzene):
    print(s.pg, s.rmsd)                         # D6h, D3h, C6h, C6v, ...
```

## API reference

### `get_point_group(mol, geom_tol=0.05, eigen_tol=None) → str`

Determine the point group of a molecule.

| Parameter | Type | Description |
|-----------|------|-------------|
| `mol` | `ase.Atoms` | Input molecule. |
| `geom_tol` | `float` | Geometric tolerance in Å (default `0.05`). |
| `eigen_tol` | `float \| None` | Relative tolerance for moment-of-inertia eigenvalues. Estimated automatically when `None`. |

Returns the Schoenflies symbol as a string (e.g. `"C2v"`, `"D3h"`, `"Oh"`).

---

### `symmetrize(mol, geom_tol=0.05, eigen_tol=None) → ase.Atoms`

Project a nearly-symmetric structure onto exact point-group symmetry.

Each set of symmetry-equivalent atoms (SEA) is handled by projecting a representative atom onto the symmetry element that fixes it, then mapping the remaining SEA members via the stored matrix representation of the connecting symmetry operation.

---

### `get_inequivalent(mol, geom_tol=0.3, eigen_tol=None) → (np.ndarray, np.ndarray)`

Find symmetry-inequivalent atoms.

Returns `(unique, parent)` where `unique` contains one representative index per equivalence class and `parent[i]` is the representative of atom `i`.

---

### `is_planar(mol, geom_tol=0.05) → bool`

Check whether all atoms lie in a common plane.

---

### `generate_symmetry_candidates(mol, geom_tol=0.05, eigen_tol=None, sort_by=0) → list[SymmetryResult]`

Generate symmetry-consistent geometries for all subgroups compatible with the detected point group.

Each `SymmetryResult` contains:
- `mol` — the symmetrized `ase.Atoms` object
- `pg` — the Schoenflies symbol
- `rmsd` — RMSD (Å) between original and symmetrized structure

`sort_by=0` sorts by group size (descending) then RMSD (ascending); `sort_by=1` sorts by RMSD first.

## Reference database

MolSymPy ships a companion database of molecular and atomic-cluster geometries indexed by point group:

```python
from molsympy.collections import symmetrized, unsymmetrized

# List available structures
print(unsymmetrized.names)   # ['C0v_1', 'C0v_2', ..., 'Td_3']
print(symmetrized.names)     # ['C0v_1', 'C0v_2', ..., 'Td_3']

# Load a structure by key  ('<PointGroup>_<index>')
mol_u = unsymmetrized['C3h_1']
mol_s = symmetrized['C3h_1']

# Iterate over all structures in a collection
for name in symmetrized.names:
    atoms = symmetrized[name]
```

The two collections are:

| Object | Content |
|--------|---------|
| `symmetrized` | Idealized geometries with exact point-group symmetry |
| `unsymmetrized` | Raw geometries with slight numerical distortions |

### Lookup by molecular formula

Every structure in the database can also be retrieved by its molecular formula.
The key and the formula are interchangeable in all collection operations:

```python
from molsympy.collections import symmetrized, unsymmetrized

# These two calls return the same structure
atoms = unsymmetrized['C0v_1']
atoms = unsymmetrized['CNH']     # equivalent

# Works with both collections
mol_u = unsymmetrized['H2O']     # same as unsymmetrized['C2v_1']
mol_s = symmetrized['H2O']       # same as symmetrized['C2v_1']

# List all available formulas
print(unsymmetrized.formulas)    # ['AgC68N4H76O4', 'AuC18P2N6H24', …, 'ZrSi7C24H52']
```

The formula stored for each entry is the one embedded in the ``.npz`` database.
Each structure carries exactly one formula alias; looking up by key always works
regardless of whether a formula is defined.

## License

MIT — see [LICENSE](LICENSE).

## Authors

- Sebastian Hernandez-Gutierrez — Departamento de Física Aplicada, Cinvestav-IPN, Mérida, México
- Diego Roman-Montalvo — Departamento de Física Aplicada, Cinvestav-IPN, Mérida, México
- Gabriel Merino — Departamento de Física Aplicada, Cinvestav-IPN, Mérida, México
- Filiberto Ortiz-Chi — Secihti-Departamento de Física Aplicada, Cinvestav-IPN, Mérida, México

If you use MolSymPy in your research, please cite the associated manuscript
