import os
import minimalsym
import pytest
from ._test_helper import read_file

PATH = os.path.dirname(os.path.realpath(__file__))

names = ["C1", "Ci", "Cs", "Cs_nonplanar", "C2", "C3", "S4", "C2v", "C3v", "C2h", "C3h", "D2", "D2d_b", "D3", "D2d", "D4d", "D3h", "D4h", "D5h", "D6h", "D12h", 
         "Cinfv", "Dinfh", "T", "Th", "Td", "cube", "octahedron", "dodecahedron", "icosahedron"]
pgs = ["C1", "Ci", "Cs", "Cs", "C2", "C3", "S4", "C2v", "C3v", "C2h", "C3h", "D2", "D2d", "D3", "D2d", "D4d", "D3h", "D4h", "D5h", "D6h", "D12h", 
       "C0v", "D0h", "T", "Th", "Td", "Oh", "Oh", "Ih", "Ih"]

@pytest.mark.parametrize("name, pg_ans", [(names[i], pgs[i]) for i in range(len(names))])
def test_find_point_group(name, pg_ans):
    file_path = os.path.join(PATH, "sxyz", f"{name}.xyz")
    with open(file_path, "r") as fn:
        strang = fn.read()
    #schema = qcel.models.Molecule.from_data(strang).dict()
    mol = read_file(file_path)
    pg_obj = minimalsym.core.pg_detect.find_point_group(mol)
    pg = pg_obj.pg
    paxis = pg_obj.paxis
    saxis = pg_obj.saxis

    print("Ans: ", pg)
    assert pg_ans == pg, f"Expected {pg_ans} got {pg} pa:{paxis} sa:{saxis}"

"""
@pytest.mark.slow
def test_find_point_group_D100h():
    with open("test/sxyz/"+"D100h"+".xyz", "r") as fn:
        strang = fn.read()
    schema = qcel.models.Molecule.from_data(strang).dict()
    mol = Molecule.from_schema(schema)
    pg, (paxis, saxis) = minimalsym.find_point_group(mol)
    print("Ans: ", pg)
    assert "D100h" == pg

"""