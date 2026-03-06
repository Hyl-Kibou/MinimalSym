import numpy as np
from dataclasses import dataclass
from copy import deepcopy
import sys
global_tol = 1e-8 # TODO It would be nice to get rid of this...

@dataclass
class SEA():
    """
    Symmetry equivalent atoms (SEA).

    SEAs are atoms that can be swapped with no distinguishable change in the molecule.

    Parameters
    ----------
    label : str or None, optional
        Optionally defines rotor type of SEA set (e.g. Single Atom, Linear, Spherical, Regular Polygon, Oblate Symmetric Top, etc.)
    subset: np.array
        Sublist of atom indices in molecule that define the SEA set, shape (N,)
    axis: np.array or None, optional
        Optional possible rotational symmetry vector, shape (3,)
    """
    label:str
    subset:np.array
    axis:np.array



class Molecule():
    """
    Molecular structure and geometry utilities.

    This class provides static helper methods for computing ase.Atoms
    molecular properties such as the center of mass, distance matrices,
    symmetry equivalent atoms, and coordinate transformations.
    """

    @staticmethod
    def find_com(mol): # deprecate!
        """
        Get center of mass of molecule.

        Parameters
        ----------
        mol: ase.Atoms
            Molecule object.

        Returns
        -------
        np.array
            Center of mass vector of shape (3,)

        Warnings
        --------
        Deprecated since version 0.1.0.
        This method will be removed in Minimalsym 1.0.0.
        Use ``ase.Atoms.get_center_of_mass`` instead.
        """

        return mol.get_center_of_mass()

    @staticmethod
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

    @staticmethod
    def transform(mol, M):
        """
        Transform coordinates of molecule by matrix M and return new molecule.

        Parameters
        ----------
        mol: ase.Atoms
            Molecule object.
        M: np.array
            Transformation matrix (e.g. rotation, reflection, etc.), shape (3,3)

        Returns
        -------
        ase.Atoms
            Molecule with transformed atom coordinates
        """
        new_mol = deepcopy(mol)
        new_mol.positions = np.dot(new_mol.positions, np.transpose(M))
        return new_mol

    @staticmethod
    def distance_matrix(mol):
        """
        Calculate the interatomic distance matrix as all pairwise distances between atoms.

        Parameters
        ----------
        mol: ase.Atoms
            Molecule object.

        Returns
        -------
        np.array
            Interatomic distance matrix, shape(len(mol),len(mol))
        """
        dm = np.zeros((len(mol),len(mol)))
        for i in range(len(mol)):
            for j in range(i,len(mol)):
                dm[i,j] = np.sqrt(sum((mol.positions[i,:]-mol.positions[j,:])**2))
                dm[j,i] = dm[i,j]
        return dm

    @staticmethod
    def find_SEAs(mol):
        """
        Find sets of symmetry equivalent atoms.
        Permutations of the distance matrix reveal which atoms form symmetry equivalent sets.

        Parameters
        ----------
        mol: ase.Atoms
            Molecule object.

        Returns
        -------
        List[molsym.SEA]
            List of symmetry equivalent atom sets
        """
        dm = Molecule.distance_matrix(mol)
        out = []
        for i in range(len(mol)):
            for j in range(i+1,len(mol)):
                a_idx = np.argsort(dm[i,:])
                b_idx = np.argsort(dm[j,:])
                z = dm[i,a_idx] - dm[j,b_idx]
                chk = True
                for k in z:
                    if abs(k) < mol.info["tol"]:
                        continue
                    else:
                        chk = False
                if chk:
                    out.append((i,j))
        skip = []
        SEAs = []
        for i in range(len(mol)):
            if i in skip:
                continue
            else:
                collect = [i]
            
            for k in out:
                if i in k:
                    if i == k[0]:
                        collect.append(k[1])
                        skip.append(k[1])
                    else:
                        collect.append(k[0])
                        skip.append(k[0])
            SEAs.append(SEA("", collect, np.zeros(3)))
        return SEAs

