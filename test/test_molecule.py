import pytest
import os
import numpy as np
from copy import deepcopy
import ase
from ._test_helper import read_file

mass_tol = 1e-8
coord_tol = 1e-6

PATH = os.path.dirname(os.path.realpath(__file__))
h2o_file = os.path.join(PATH, "xyz", "water.xyz")

atoms = ["O", "H", "H"]
coords = np.array([
    [1.2091536548, 1.7664118189,-0.0171613972],
    [2.1984800075, 1.7977100627, 0.0121161719],
    [0.9197881882, 2.4580185570, 0.6297938832]])
masses = [15.99491461957, 1.00782503223, 1.00782503223]
molsym_mol = ase.Atoms(symbols=atoms, positions=coords)
molsym_mol.info["tol"] = 0.05
com = np.array([1.24832167, 1.80686373, 0.02067886])

"""
schema = {"symbols": atoms,
          "geometry": coords
          }

psi_schema = {"elem": atoms,
              "geom": coords
          }
"""

def mols_are_same(mol1, mol2):
    for i in range(len(mol1)):
        assert mol1.get_chemical_symbols()[i] == mol2.get_chemical_symbols()[i]
        assert np.isclose(mol1.get_masses()[i], mol2.get_masses()[i], atol=mass_tol)
    assert np.isclose(mol1.positions, mol2.positions, atol=coord_tol).all()

"""
def _test_mol_from_schema():
    newmol = molsym.Molecule.from_schema(schema)
    mols_are_same(molsym_mol, newmol)
"""

def test_mol_from_file():
    newmol = read_file(h2o_file)
    mols_are_same(molsym_mol, newmol)

"""
def _test_mol_from_psi4_schema():
    newmol = molsym.Molecule.from_psi4_schema(psi_schema)
    mols_are_same(molsym_mol, newmol)
"""

def test_find_com():
    assert np.isclose(molsym_mol.get_center_of_mass(), com, atol=coord_tol).all

def is_at_com(mol):
        """
        Check if molecule is centered at its center of mass already.

        Parameters
        ----------
        mol: ase.Atoms
            Molecule object.

        Returns
        -------
        bool
            True if the center of mass is within the tolerance defined by
            ``mol.info["tol"]``, False otherwise.

        """
        if sum(abs(mol.get_center_of_mass())) < mol.info["tol"]:
            return True
        else:
            return False


def test_is_at_com():
    newmol = deepcopy(molsym_mol)
    newmol.positions -= np.repeat(com, 3).reshape((3,3)).T
    assert is_at_com(newmol)

"""
def _test_eq():
    assert molsym.Molecule.from_file(h2o_file, keep_angstrom=True) == molsym_mol
    assert molsym.Molecule.from_schema(schema) == molsym_mol
    assert molsym.Molecule.from_psi4_schema(psi_schema) == molsym_mol
"""