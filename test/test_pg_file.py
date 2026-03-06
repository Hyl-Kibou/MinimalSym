from ase.io import read
from minimalsym import symmetrize
import os
import pytest

PATH = os.path.dirname(os.path.realpath(__file__))
ASYM_TOL = 0.05

def write_tests(file_test, file_check):
  listmol = read(file_test, index=":")
  pg_list = []
  for imol in listmol:
    smol = symmetrize(imol, asym_tol=ASYM_TOL)
    pg_list.append(smol.info['pg'])
  with open(file_check, "w") as f:
    for pg in pg_list:
      f.write(pg + '\n')

def check_xyz_folder(folder_path):

  path_list = [
      os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".xyz")
      ]
  
  current_list_path = []

  for file in path_list:
    test_path = file
    split_path = os.path.splitext(test_path)
    check_path = split_path[0] + "_check.out"      

    #write_tests(test_path, check_path)

    current_list_path.append((test_path, check_path))
  
  return current_list_path
  
list_folders = ["new_xyz"]
list_paths = []

for folder_name in list_folders:
  complete_path = os.path.join(PATH, folder_name)
  list_paths.extend(check_xyz_folder(complete_path))

@pytest.mark.parametrize("file_test, file_check", [(list_paths[i][0], list_paths[i][1]) for i in range(len(list_paths))])
def test_pg_file(file_test, file_check):
  listmol = read(file_test, index=":")
  pg_list = []
  for imol in listmol:
    smol = symmetrize(imol, asym_tol=ASYM_TOL)
    pg_list.append(smol.info['pg'])
  correct_pg_list = []
  with open(file_check, 'r') as f:
      correct_pg_list = [line.rstrip('\n') for line in f]
  test_passed = (pg_list == correct_pg_list)
  assert test_passed