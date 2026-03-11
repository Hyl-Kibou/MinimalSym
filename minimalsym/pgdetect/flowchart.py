import numpy as np
from ..symtools import rotation_matrix, inversion_matrix, Sn, isequivalent, calcmoit, normalize
from .flowchart_helper import find_rotation_sets, find_rotations, linear_mol_axis, find_a_c2, is_there_ortho_c2, num_C2, highest_order_axis, is_there_sigmah, is_there_sigmav, mol_is_planar, planar_mol_axis, find_C3s_for_Ih, find_C4s_for_Oh
from ..molecule import transform, find_SEAs
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms

def find_point_group(mol: "Atoms"):
    """
    Find the point group of a molecule.

    Returns the point group as a string, and primary and secondary axis
    in order to define an orientation of the molecule with respect to
    the symmetry elements generated later.

    Based on the algorithm developed by:
        Beruski, Otávio; Vidal, Luciano N. Algorithms for computer detection of 
        symmetry elements in molecular systems, J. Comp. Chem, 2013 doi:10.1002/jcc.23493
        
    
    Parameters
    ----------
    mol: ase.Atoms
        Molecule which to find point group.

    Returns
    -------
    tuple(str, (np.array, np.array))
        Schoenflies point group string, primary axis, and secondary axis of shape(3,).
    """
    try:
        mol.info['tol']
    except KeyError:
        raise Exception("Atoms object tolerance hasn't been set. Set it with Atoms.info['tol']=.")

    mol = mol.copy()

    paxis = [0,0,0]
    saxis = [0,0,0]
    moit = calcmoit(mol)
    Ia_mol, Ib_mol, Ic_mol = np.sort(np.linalg.eigh(moit)[0])
    # Linear tops
    if np.isclose(Ia_mol, 0.0, atol=mol.info["tol"]):
        paxis = linear_mol_axis(mol)
        if isequivalent(mol, transform(mol, inversion_matrix())):
            pg = "D0h"
        else:
            pg = "C0v"
    # Spherical tops
    elif np.isclose(Ia_mol, Ib_mol, atol=mol.info["tol"]) and np.isclose(Ia_mol, Ic_mol, atol=mol.info["tol"]):
        seas = find_SEAs(mol)
        n, axes = num_C2(mol, seas)
        invertable = isequivalent(mol, transform(mol, inversion_matrix()))
        # Icosahedral
        if n == 15:
            # tempaxis is any C2 axis 
            # saxis is defined as the C2 axis that is coplanar with tempaxis and the nearest C3 axis to tempaxis
            tempaxis = axes[0]
            c3s = find_C3s_for_Ih(mol)
            for c3 in c3s:
                if np.isclose(np.arccos(abs(np.dot(c3, tempaxis))), 0.36486382647383764, atol=1e-4):
                    taxis = normalize(np.cross(c3, tempaxis))
                    saxis = normalize(np.cross(taxis, tempaxis))
                    break
            # Reorienting vectors such that one face is on the z-axis with "pentagon" pointing at the POSITIVE y-axis
            phi = (1+np.sqrt(5.0))/2
            # Weirdness here, negative or positive???
            theta = np.arccos(phi/np.sqrt(1+(phi**2)))
            rmat = rotation_matrix(saxis, theta)
            paxis = np.dot(rmat, tempaxis)
            if invertable:
                pg = "Ih"
            else:
                pg = "I"
        # Octahedral
        elif n == 9:
            # paxis and saxis are orthogonal C4 axes
            c4s = find_C4s_for_Oh(mol)
            paxis = c4s[0]
            saxis = c4s[1]
            if invertable:
                pg = "Oh"
            else:
                pg = "O"
        # Tetrahedral, TODO: No path for T
        elif n == 3:
            paxis = axes[0]
            saxis = axes[1]
            if invertable:
                pg = "Th"
            else:
                pg = "Td"
    else:
        seas = find_SEAs(mol)
        rot_set = find_rotation_sets(mol, seas)
        rots = find_rotations(mol, rot_set)
        if len(rots) >= 1:
            Cn_order = highest_order_axis(rots)
            paxis = rots[0].axis
        else:
            c2 = find_a_c2(mol, seas)
            if c2 is None:
                molB = transform(mol, inversion_matrix())
                if isequivalent(mol, molB):
                    return "Ci", (paxis, saxis)
                else:
                    sigmav_chk, sigmav = is_there_sigmav(mol, seas, np.asarray([0,0,0]))
                    if sigmav_chk:
                        if sigmav is not None:
                            paxis = sigmav
                        return "Cs", (paxis, saxis)
                    else:
                        return "C1", (paxis, saxis)
            paxis = c2
            Cn_order = 2
        
        ortho_c2_chk, c2_ortho = is_there_ortho_c2(mol, seas, paxis)
        sigmav_chk, sigmav = is_there_sigmav(mol, seas, paxis)
        sigmah_chk = is_there_sigmah(mol, paxis)
        if ortho_c2_chk:
            saxis = c2_ortho
            if sigmah_chk:
                pg = "D"+str(Cn_order)+"h"
            elif sigmav_chk:
                pg = "D"+str(Cn_order)+"d"
            else:
                pg = "D"+str(Cn_order)
        elif sigmah_chk:
            pg = "C"+str(Cn_order)+"h"
        elif sigmav_chk:
            pg = "C"+str(Cn_order)+"v"
            if mol_is_planar(mol):
                saxis = planar_mol_axis(mol)
            elif sigmav.any():
                saxis = normalize(np.cross(paxis, sigmav))
        else:
            S2n = Sn(paxis, Cn_order*2)
            molB = transform(mol, S2n)
            if isequivalent(mol, molB):
                pg = "S"+str(2*Cn_order)
            else:
                pg = "C"+str(Cn_order)
    return pg, (paxis, saxis)
