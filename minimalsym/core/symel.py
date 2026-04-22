import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Symel():
    """
    Symmetry element (SymEl):
        Symmetry operation with its symbol, representative axis,
        matrix representation and metadata.

    Parameters
    ----------
    symbol: str
        Schoenflies symbol of the symmetry element (e.g. "C_3", "sigma_h", "i").
    vector: np.array or None
        Axis vector for Cn and Sn; plane normal vector for sigma; None for E and i.
    rrep: np.array
        3x3 real-space matrix representation of the operation.
    m: int or None
        Power of the generator (e.g. m=2 for C_3^2).
    n: int or None
        Order of the principal axis of the generating cycle.
    O: str or None
        Origin class of the element. Options: "E", "sigma_v", "C_2'", "i", "sigma_h".
    """
    symbol: str
    vector: np.ndarray | None
    rrep: np.ndarray
    m: int | None = None
    n: int | None = None
    O: str | None = None

    def __str__(self) -> str:
        if self.rrep is None:
            return f"\nSymbol: {self.symbol:>10s}: [no matrix — linear group placeholder]"
        with np.printoptions(precision=5, suppress=True, formatter={"all": lambda x: f"{x:8.5f}"}):
            return f"\nSymbol: {self.symbol:>10s}: [{self.rrep[0,:]},{self.rrep[1,:]},{self.rrep[2,:]}]"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        if self.rrep is None or other.rrep is None:
            return self.symbol == other.symbol and self.rrep is other.rrep
        return self.symbol == other.symbol and np.isclose(self.rrep, other.rrep, atol=1e-10).all()
