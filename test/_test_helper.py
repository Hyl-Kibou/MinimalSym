import numpy as np
import ase

def read_file(file):

    symbols = []
    positions = []

    with open(file, "r") as f:
        for line in f:
            if line.strip():
                coord = []
                split_line = line.split()
                if len(split_line) != 4:
                    continue
                symbols.append(split_line[0])
                coord.append(float(split_line[1]))
                coord.append(float(split_line[2]))
                coord.append(float(split_line[3]))

                positions.append(np.array(coord))

    positions = np.array(positions)

    atoms = ase.Atoms(symbols=symbols, positions=positions)
    atoms.info["geom_tol"] = 0.05
    atoms.info['eigen_tol'] = 1e-05

    return atoms