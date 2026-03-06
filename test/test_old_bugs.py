"""
    Put old bugged cases here so we can make sure we don't reintroduce them!
"""
import os
import numpy as np
import minimalsym
from ase.io import read
from ._test_helper import read_file

PATH = os.path.dirname(os.path.realpath(__file__))

# Test formaldehyde. Bug was planar molecules not being assigned a secondary axis
def test_formaldehyde():
    file_path = os.path.join(PATH, "xyz", f"formaldehyde.xyz")
    mol = read_file(file_path)
    pg, (paxis, saxis) = minimalsym.find_point_group(mol)

    assert pg == "C2v"
    x,y,z = np.eye(3)
    assert (np.isclose(paxis, z).all() or np.isclose(paxis, -z).all())
    assert (np.isclose(saxis, x).all() or np.isclose(saxis, -x).all())

def test_collapse():
    file_path = os.path.join(PATH, "new_xyz", f"Malo_CC2Si2.xyz")
    mol = read(file_path)
    smol = minimalsym.symmetrize(mol)

    for ii in range(len(smol.positions)):
        for jj in range(ii + 1, len(smol.positions)):
            assert not np.allclose(smol.positions[ii], smol.positions[jj])    

