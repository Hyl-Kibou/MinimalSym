#!/usr/bin/python3 -u
import sys
import glob
import pytest
import os.path
from ase.io import read, write
from collections import defaultdict
from minimalsym import get_point_group

PATH = os.path.dirname(os.path.realpath(__file__))

#------------------------------------------------------------------------------------------
symmetry_number = {
    # Cn families
    'C1': 1,
    'Cs': 1,
    'Ci': 1,

    'C2': 2,
    'C3': 3,
    'C4': 4,
    'C5': 5,
    'C6': 6,

    'C2v': 2,
    'C3v': 3,
    'C4v': 4,
    'C5v': 5,
    'C6v': 6,

    'C2h': 2,
    'C3h': 3,
    'C4h': 4,
    'C5h': 5,
    'C6h': 6,

    # Dn families
    'D2': 4,
    'D3': 6,
    'D4': 8,
    'D5': 10,
    'D6': 12,

    'D2h': 4,
    'D3h': 6,
    'D4h': 8,
    'D5h': 10,
    'D6h': 12,

    'D2d': 4,
    'D3d': 6,
    'D4d': 8,
    'D5d': 10,

    # cubic groups
    'T': 12,
    'Td': 12,
    'Th': 12,

    'O': 24,
    'Oh': 24,

    'I': 60,
    'Ih': 60
}
#------------------------------------------------------------------------------------------

list_paths = []

for i in range(5,100+1,1):
    complete_path = os.path.join(PATH, "LJxyz")
    prefix='LJ'+str(i+1).zfill(3)
    prefix = os.path.join(complete_path, prefix)
    filexyz=prefix+'ALL.xyz'
    if not os.path.isfile(filexyz): continue
    list_paths.append(filexyz)

mol_list = []

for path in list_paths:
    moleculelist=read(path, index=":")
    mol_list.extend(moleculelist)

#mol_list = [] ##

# Test the runtime performance of get_point_group function
@pytest.mark.slow
@pytest.mark.parametrize("imol", [imol for imol in mol_list])
def test_pg_LJ(imol):
    pg =get_point_group(imol, 0.01)
    assert type(pg) is str, f"pg is {type(pg)}"
    sn=symmetry_number[pg]    
    if sn < 4: return