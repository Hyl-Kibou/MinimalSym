from ase.io import read, write
from minimalsym import symmetrize
import numpy as np
import os
import pytest

PATH = os.path.dirname(os.path.realpath(__file__))
ASYM_TOL = 0.05

def write_tests(file_test, file_check, file_sym):
  listmol = read(file_test, index=":")
  pg_list = []
  smol_list = []
  for imol in listmol:
    smol = symmetrize(imol, asym_tol=ASYM_TOL)
    smol_list.append(smol)
    pg_list.append(smol.info['pg'])
  with open(file_check, "w") as f:
    for pg in pg_list:
      f.write(pg + '\n')
  write(file_sym, smol_list)

def check_xyz_folder(folder_path):

  path_list = [
      os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".xyz") and f[0] != '_'
      ]

  current_list_path = []

  for file in path_list:
    test_path = file
    split_path = os.path.splitext(test_path)
    check_path = split_path[0] + "_check.out"
    sym_path = os.path.join(folder_path, ("_sym_" + os.path.basename(file)))

    #write_tests(test_path, check_path, sym_path)

    current_list_path.append((test_path, check_path, sym_path))

  return current_list_path

list_folders = ["new_xyz"]
list_paths = []

for folder_name in list_folders:
  complete_path = os.path.join(PATH, folder_name)
  list_paths.extend(check_xyz_folder(complete_path))

def atoms_are_close(a1, a2, tol=1e-7):
    if len(a1) != len(a2):
        return False
    if a1.get_chemical_symbols() != a2.get_chemical_symbols():
        return False
    return np.allclose(a1.get_positions(), a2.get_positions(), atol=tol)

@pytest.mark.parametrize("file_test, file_check, file_sym", [(list_paths[i][0], list_paths[i][1], list_paths[i][2]) for i in range(len(list_paths))])
def test_pg_file(file_test, file_check, file_sym):
  listmol = read(file_test, index=":")
  pg_list = []
  sym_list = []
  for imol in listmol:
    smol = symmetrize(imol, asym_tol=ASYM_TOL)
    sym_list.append(smol)
    pg_list.append(smol.info['pg'])
  correct_pg_list = []
  with open(file_check, 'r') as f:
      correct_pg_list = [line.rstrip('\n') for line in f]
  pg_is_same = (pg_list == correct_pg_list)
  old_sym_list = read(file_sym, index=":")
  nummol_is_same = (len(sym_list) == len(old_sym_list))
  for ii in range(len(sym_list)):
      assert atoms_are_close(sym_list[ii], old_sym_list[ii]), f"geometry subset: {ii}"
  geometry_is_same = all(
    atoms_are_close(a, b)
    for a, b in zip(sym_list, old_sym_list)
  )
  assert pg_is_same, f"pg_is_same: {pg_is_same}"
  assert nummol_is_same, f"nummol_is_same: {nummol_is_same}"
  assert geometry_is_same, f"geometry_is_same: {geometry_is_same}"