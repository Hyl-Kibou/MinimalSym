from ase.io import read, write
from ase import Atoms
import os
import numpy as np
from minimalsym import symmetrize, get_point_group, is_planar, get_inequivalent

PATH = os.path.dirname(os.path.realpath(__file__))
folder_path = os.path.join(PATH, "new_xyz")

def test_symmetrize():
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
    assert mol_symmetric.info["pg"] == "C2v", f"mol_symmetric.info[\"pg\"]: {mol_symmetric.info["pg"]}"
    assert np.allclose(mol_symmetric.positions, [[ 0., 0., 0.066],[-0., -0.757, -0.521], [0., 0.757, -0.521]], atol=1e-3), f"mol.positions: {mol_symmetric.positions}"

def test_get_point_group():
    mol = read(os.path.join(folder_path, "minimos.xyz"), index=":")[94]

    ## Detect point group for molecule
    pg_str = get_point_group(mol, geom_tol=0.05)

    ## Check output
    print("Detected Point group: ", pg_str)
    # Example output: "Detected Point group: C2v"

def test_is_planar():
    mol = read(os.path.join(folder_path, "minimos.xyz"), index=":")[94]

    ## Check planarity for molecule
    mol_is_planar = is_planar(mol, geom_tol=0.05)

    ## Check output
    print("Mol is planar: ", mol_is_planar)
    # Example output: "Mol is planar: True"
    assert mol_is_planar == True, f"mol_is_planar: {mol_is_planar}"


def test_get_inequivalent():
    mol = read(os.path.join(folder_path, "minimos.xyz"), index=":")[0]

    ## Get symmetry-inequivalent atoms for molecule
    atom_indices_list, representatives = get_inequivalent(mol, geom_tol=0.3)

    ## Check output
    print("Inequivalent indices list: ", atom_indices_list)
    # Example output: "Inequivalent indices list: [0 1 2]"
    assert (atom_indices_list == np.array([0, 1, 2])).all(), f"atom_indices_list: {atom_indices_list}"

    print("Representatives: ", representatives)
    # Example output: "Inequivalent indices list: [0 1 2 0 0 1 1 1 1 2 2 1]"
    assert (representatives == np.array([0, 1, 2, 0, 0, 1, 1, 1, 1, 2, 2, 1])).all(), f"representatives: {representatives}"
