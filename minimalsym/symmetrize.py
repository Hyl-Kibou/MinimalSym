import numpy as np
from .molecule import Molecule
from .symtext.symtext import Symtext

def symmetrize(mol_in, asym_tol=0.05):
    """
    Symmetrizes a molecule to a detectable low tolerance point group.
    Constructs a molsym.Symtext for mol_in with tolerance asym_tol.
    The atoms are then projected onto the symmetry elements in the Symtext.
    The tolerance of the returned molecule is set to 1e-12.

    :param mol_in: Molecule to be symmetrized.
    :param asym_tol: Tolerance for asymmetry in mol_in, default is 0.05
    :type mol_in: Atoms object from ASE
    :type asym_tol: float
    :return: Symmetrized molecule
    :rtype: Atoms object from ASE
    """    
    mol_in = mol_in.copy()

    mol_in.info["tol"] = asym_tol
    seas = Molecule.find_SEAs(mol_in)
    asym_symtext = Symtext.from_molecule(mol_in)
    mol = asym_symtext.mol
    for sea in seas:
        atom_i = sea.subset[0]
        # Linear path
        if asym_symtext.pg.is_linear:
            z = np.array([0,0,1])
            if asym_symtext.pg.family == "C":
                # Project onto z-axis
                for atom_j in sea.subset[:]:
                    mol.positions[atom_j,:] = np.array([0.0, 0.0, np.dot(mol.positions[atom_j,:], z)])
            elif asym_symtext.pg.family == "D":
                if atom_i == asym_symtext.atom_map[atom_i, 1]:
                    # Atom i must be at origin
                    mol.positions[atom_i,:] = np.array([0.0,0.0,0.0])
                else:
                    mol.positions[atom_i,:] = np.array([0.0, 0.0, np.dot(mol.positions[atom_i,:], z)])
                for atom_j in sea.subset[1:]:
                    mol.positions[atom_j,:] = np.array([0.0, 0.0, np.dot(mol.positions[atom_j,:], z)])
                    if atom_j == asym_symtext.atom_map[atom_i, 1]:
                        mol.positions[atom_j,:] = np.dot(-1*np.eye(3), mol.positions[atom_i,:])
            mol.info["tol"] = 1e-12
            return mol
        for g in range(1, asym_symtext.order):
            if atom_i == asym_symtext.atom_map[atom_i, g]:
                # Atom i invariant under g
                if asym_symtext.symels[g].symbol == "i":
                    # Atom i must be at origin
                    mol.positions[atom_i,:] = np.array([0.0,0.0,0.0])
                    break
                elif asym_symtext.symels[g].symbol[0] == "S":
                    # Atom i must be at origin
                    mol.positions[atom_i,:] = np.array([0.0,0.0,0.0])
                    break
                elif asym_symtext.symels[g].symbol[0] == "C":
                    # Project atom i onto rotation axis
                    l = np.dot(mol.positions[atom_i,:],asym_symtext.symels[g].vector)
                    mol.positions[atom_i,:] = l * asym_symtext.symels[g].vector
                elif asym_symtext.symels[g].symbol[:5] == "sigma":
                    # Project atom i onto plane
                    l = np.dot(mol.positions[atom_i,:],asym_symtext.symels[g].vector)
                    mol.positions[atom_i,:] -= l*asym_symtext.symels[g].vector
                else:
                    raise Exception("Wut")
        # Place other atoms in SEA based off of atom i
        for atom_j in sea.subset[1:]:
            for g in range(1,asym_symtext.order):
                if atom_j == asym_symtext.atom_map[atom_i, g]:
                    mol.positions[atom_j,:] = np.dot(asym_symtext.symels[g].rrep, mol.positions[atom_i,:])
                    break

    mol.info["tol"] = 1e-12
    return mol
