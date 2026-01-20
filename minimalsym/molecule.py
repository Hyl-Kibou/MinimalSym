import numpy as np
from dataclasses import dataclass
from copy import deepcopy
import sys
global_tol = 1e-8 # TODO It would be nice to get rid of this...

@dataclass
class SEA():
    """
    SEA: symmetry equivalent atoms.
    SEAs are atoms that can be swapped with no distinguishable change in the molecule.

    :param label: Optionally defines rotor type of SEA set (e.g. Single Atom, Linear, Spherical, Regular Polygon, Oblate Symmetric Top, etc.)
    :param subset: Sublist of atom indices in molecule that constitute the SEA set
    :param axis: Optionally defines possible rotational symmetry vector
    :type label: str or None
    :type subset: NumPy array of integers
    :type axis: NumPy array of shape (3,) or None
    """
    label:str
    subset:np.array
    axis:np.array



class Molecule():
    """
    Class dealing with molecule relevant information.
    Typically initiated from a QCSchema object.
    """

    @staticmethod
    def find_com(self): # deprecate!
        """
        Get center of mass of molecule.

        :return: Center of mass
        :rtype: NumPy array of shape (3,)
        """

        return self.get_center_of_mass()

    @staticmethod
    def is_at_com(self):
        """
        Checks if molecule is at center of mass already.

        :rtype: bool
        """
        if sum(abs(self.get_center_of_mass())) < self.info["tol"]:
            return True
        else:
            return False    

    @staticmethod 
    def transform(self, M):
        """
        Transform coordinates of molecule by matrix M and return new molecule.

        :param M: Transformation matrix (e.g. rotation, reflection, etc.)
        :type M: NumPy array (3,3)
        :return: Molecule with transformed atom coordinates
        :rtype: molsym.Molecule
        """
        new_mol = deepcopy(self)
        new_mol.positions = np.dot(new_mol.positions, np.transpose(M))
        return new_mol

    @staticmethod
    def distance_matrix(self):
        """
        Calculates the interatomic distance matrix as all pairwise distances between atoms.

        :return: Interatomic distance matrix
        :rtype: NumPy array of shape (len(self),len(self))
        """
        dm = np.zeros((len(self),len(self)))
        for i in range(len(self)):
            for j in range(i,len(self)):
                dm[i,j] = np.sqrt(sum((self.positions[i,:]-self.positions[j,:])**2))
                dm[j,i] = dm[i,j]
        return dm

    @staticmethod
    def find_SEAs(self):
        """
        Find sets of symmetry equivalent atoms.
        Permutations of the distance matrix reveal which atoms form symmetry equivalent sets.

        :return: List of symmetry equivalent atom sets
        :rtype: List[molsym.SEA]
        """
        dm = Molecule.distance_matrix(self)
        out = []
        for i in range(len(self)):
            for j in range(i+1,len(self)):
                a_idx = np.argsort(dm[i,:])
                b_idx = np.argsort(dm[j,:])
                z = dm[i,a_idx] - dm[j,b_idx]
                chk = True
                for k in z:
                    if abs(k) < self.info["tol"]:
                        continue
                    else:
                        chk = False
                if chk:
                    out.append((i,j))
        skip = []
        SEAs = []
        for i in range(len(self)):
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

