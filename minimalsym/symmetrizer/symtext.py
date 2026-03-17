"""
symtext.py — Full symmetry characterization of a molecule (Symtext class).

Dependencies:
  pg_detect   — find_point_group
  point_group — PointGroup
  symel_gen   — pg_to_symels
  mol_orient  — rotate_mol_to_symels
  atom_mapping — _get_atom_mapping, _get_linear_atom_mapping
"""

import numpy as np

from .pg_detect import find_point_group
from .point_group import PointGroup
from .symel_gen import pg_to_symels
from .mol_orient import rotate_mol_to_symels
from .atom_mapping import _get_atom_mapping, _get_linear_atom_mapping


class Symtext():
    """
    Full symmetry characterization of a molecule.

    Holds the point group, list of symmetry elements, atom permutation map,
    and the rotation matrices that relate the molecule to the canonical
    orientation defined by the symmetry elements.
    """

    def __init__(self, mol, rotate_to_std, reverse_rotate, pg, symels, atom_map):
        self.mol = mol
        self.rotate_to_std = rotate_to_std
        self.reverse_rotate = reverse_rotate
        self.pg = pg
        self.symels = symels
        self.atom_map = atom_map
        if pg.is_linear:
            # Linear groups have infinitely many symmetry elements; the finite
            # atom_map covers only the operations that actually permute atoms.
            self.order = atom_map.shape[1]
        else:
            self.order = len(symels)

    def __len__(self):
        return len(self.symels)

    def __repr__(self):
        return f"\n{self.mol}\n{self.symels}\nAtom map:\n{self.atom_map}"

    @classmethod
    def empty(cls):
        return Symtext(None, None, None, PointGroup.from_string("C1"), [], [])

    @classmethod
    def from_molecule(cls, mol):
        """
        Build a Symtext from an ase.Atoms object.

        Parameters
        ----------
        mol : ase.Atoms

        Returns
        -------
        Symtext
        """
        mol.translate(-mol.get_center_of_mass())
        pgr = find_point_group(mol)
        pg = PointGroup.from_string(pgr.pg)
        mol, reverse_rotate, rotate_to_std = rotate_mol_to_symels(mol, pgr.paxis, pgr.saxis)
        mol.info["pg"] = pgr.pg
        symels = pg_to_symels(pg.str)
        if pg.is_linear:
            atom_map = _get_linear_atom_mapping(mol, pg)
        else:
            atom_map = _get_atom_mapping(mol, symels)
        return Symtext(mol, rotate_to_std, reverse_rotate, pg, symels, atom_map)
