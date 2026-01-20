# MinimalSym
A python package for handling molecular symmetry for ASE's Atoms object. Minimal modified version of <a href=https://github.com/NASymmetry/MolSym>Molsym</a>.


## Capabilities
- [Point group detection](https://github.com/NASymmetry/MolSym/wiki/Point-group-detection)
- [Molecule symmetrization](https://github.com/NASymmetry/MolSym/wiki/Symmetrizing-a-molecule)
- Symmetry element generation
- Character table generation


## Installing
MinimalSym is tested with Python 3.12-3.13, but should work fro 3.9-3.13.

## Use example
```python
from ase import Atoms
import numpy as np
from minimalsym.symmetrize import symmetrize

theta =np.radians(104.5)
positions = np.array(
    [[0.0, 0.0, 0.0],
    [0.958, 0.0, 0.0],
    [0.958 * np.cos(theta), 0.958 * np.sin(theta), 0.0]])
mol = Atoms(symbols=["O", "H", "H"], positions=positions)
mol_symmetric =symmetrize(mol, asym_tol=0.05)            
```