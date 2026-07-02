"""
    Put old bugged cases here so we can make sure we don't reintroduce them!
"""
import os
import numpy as np
import molsympy
from ase.io import read
from ._helper import read_file

PATH = os.path.dirname(os.path.realpath(__file__))

# Test formaldehyde. Bug was planar molecules not being assigned a secondary axis
def test_formaldehyde():
    file_path = os.path.join(PATH, "xyz", f"formaldehyde.xyz")
    mol = read_file(file_path)
    pg_obj = molsympy.core.pg_detect.find_point_group(mol)
    pg = pg_obj.pg
    paxis = pg_obj.paxis
    saxis = pg_obj.saxis

    assert pg == "C2v"
    x,y,z = np.eye(3)
    assert (np.isclose(paxis, z).all() or np.isclose(paxis, -z).all())
    assert (np.isclose(saxis, x).all() or np.isclose(saxis, -x).all())

# Linear molecules could collapse to center.
def test_collapse():
    file_path = os.path.join(PATH, "new_xyz", f"Malo_CC2Si2.xyz")
    mol = read(file_path)
    smol = molsympy.symmetrize(mol)

    for ii in range(len(smol.positions)):
        for jj in range(ii + 1, len(smol.positions)):
            assert not np.allclose(smol.positions[ii], smol.positions[jj])    

# Some D family point groups were assigned without proper validation,
# resulting in mapping errors.
def test_overestimate_D_family():
    file_path1 = os.path.join(PATH, "new_xyz", f"overestimation_D_A.xyz")
    file_path2 = os.path.join(PATH, "new_xyz", f"overestimation_D_B.xyz")

    file_path_list = [file_path1, file_path2]

    for path in file_path_list:
        mol = read(path)
        smol = molsympy.symmetrize(mol)

# C0v molecules weren't properly symmetrized, they should have two rows of 0.
def test_c0v():
    file_path = os.path.join(PATH, "new_xyz", f"error.xyz")
    listmol = read(file_path, index=':')
    smol = molsympy.symmetrize(listmol[0])

    assert smol.info["pg"]=="C0v", f"Wrong point group, got: {smol.info["pg"]} expected: C0v"
    for ii in range(len(smol.positions)):
        assert np.allclose(smol.positions[ii][:2], np.zeros(2), atol=0., rtol=0.), f"Positions not aligned with z-axis, got: {smol.positions}{smol.positions[ii][:2]} expected: {np.zeros(2)}"

# Some molecules failed atom mapping for `geom_tol` = 0.01. Caused by:
# - no orthogonalization check for ortho c2 axes and vertical mirror planes
# - numerical error
# - not checking suborder rotations
def test_different_geom_tol():
    file_path1 = os.path.join(PATH, "new_xyz", f"variate_geom_tol.xyz")

    file_path_list = [file_path1]

    for path in file_path_list:
        for tol in range(1, 100, 5):
            mol = read(path)
            smol = molsympy.symmetrize(mol, geom_tol=tol/100)