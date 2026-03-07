import numpy as np
from minimalsym import find_point_group
from .point_group import PointGroup
from .general_irrep_mats import pg_to_symels
from .symtext_helper import get_atom_mapping, rotate_mol_to_symels, get_linear_atom_mapping

class Symtext():
    """
    Fundamental object of MolSym, holds most of the symmetry information of the molecule.
    """
    def __init__(self, mol, rotate_to_std, reverse_rotate, pg, symels, atom_map) -> None:
        self.mol = mol
        self.rotate_to_std = rotate_to_std
        self.reverse_rotate = reverse_rotate
        self.pg = pg
        self.symels = symels
        self.atom_map = atom_map
        if pg.is_linear:
            # Haar measure
            if pg.family == "C":
                self.order = 4*np.pi
            elif pg.family == "D":
                self.order = 8*np.pi
        else:
            self.order = len(symels)        

    def __len__(self):
        return len(self.symels)

    def __repr__(self):
        return f"\n{self.mol}\n{self.symels}\nAtom map:\n{self.atom_map}"

    @classmethod
    def empty(cls):
        mol = None
        rotate_to_std = None
        reverse_rotate = None
        pg = PointGroup.from_string("C1")
        symels = []
        atom_map = []        
        return Symtext(mol, rotate_to_std, reverse_rotate, pg, symels, atom_map)

    @classmethod
    def from_molecule(cls, mol):
        """
        Class method for creating a Symtext from a ase.Atoms

        :type mol: Atoms object from ASE
        :rtype: molsym.Symtext
        """
        mol.translate(-mol.get_center_of_mass())
        pg_str, (paxis, saxis) = find_point_group(mol)
        pg = PointGroup.from_string(pg_str)
        # Return transformation matrix so properties can be rotated to original configuration
        mol, reverse_rotate, rotate_to_std = rotate_mol_to_symels(mol, paxis, saxis)
        mol.info["pg"] = pg_str
        symels = pg_to_symels(pg.str)
        if pg.is_linear:
            atom_map = get_linear_atom_mapping(mol, pg)
            return Symtext(mol, rotate_to_std, reverse_rotate, pg, symels, atom_map)
        atom_map = get_atom_mapping(mol, symels)
        return Symtext(mol, rotate_to_std, reverse_rotate, pg, symels, atom_map)