"""
symel_gen.py — Public API for symmetry-element generation.

This module exposes a single public function, pg_to_symels(), which maps a
Schoenflies point-group symbol to its list of Symel objects.

Internal logic is split across three focused sub-modules:
  group_algebra.py      — arithmetic (_omega, _mult_iCnm, …)
  cyclic_dihedral.py    — _Zn, _Dihn, _direct_product
  cubic_icosahedral.py  — T/O/I family generators and precomputed tuples
"""

import numpy as np

from .point_group import PointGroup
from .symel import Symel
from .sym_ops import reflection_matrix, inversion_matrix, Cn
from .cyclic_dihedral import _Zn, _Dihn, _direct_product
from .cubic_icosahedral import (
    T_SYMELS, TD_SYMELS, TH_SYMELS,
    O_SYMELS, OH_SYMELS,
    I_SYMELS, IH_SYMELS,
)


def pg_to_symels(PG):
    """
    Return the list of symmetry elements for the given point group.

    Parameters
    ----------
    PG : str
        Schoenflies point group symbol (e.g. "C2v", "D6h", "Oh").

    Returns
    -------
    List[Symel]
    """
    pg = PointGroup.from_string(PG)
    argerr = f"Invalid point group or unexpected parse result: {pg.str}"
    z = np.array([0, 0, 1])
    i = Symel("i", None, inversion_matrix(), 0, 0, "i")
    sh = Symel("sigma_h", np.array([0, 0, 1]), reflection_matrix(z), 0, 0, "sigma_h")
    if pg.is_linear:
        if pg.family == "C":
            return [Symel("C", z, None, None, None, None),
                    Symel("sigma_v", None, None, None, None, None)]
        elif pg.family == "D":
            return [Symel("C", z, None, None, None, None),
                    Symel("sigma_v", None, None, None, None, None),
                    Symel("S", z, None, None, None, None),
                    Symel("C_2'", None, None, None, None, None)]
    if pg.n is not None:
        n_is_even = (pg.n % 2 == 0)
        n_is_doubleeven = (pg.n % 4 == 0)
    if pg.family == "C":
        if pg.subfamily == "h":
            cn_symels = _Zn(pg.n, "C")
            if n_is_even:
                return _direct_product(cn_symels, i)
            else:
                return _direct_product(cn_symels, sh)
        elif pg.subfamily == "v":
            return _Dihn(pg.n, "C", "sigma_v")
        elif pg.subfamily == "s":
            return [Symel("E", None, np.eye(3), 0, None, "E"), sh]
        elif pg.subfamily == "i":
            return [Symel("E", None, np.eye(3), 0, None, "E"), i]
        elif pg.subfamily is None:
            return _Zn(pg.n, "C")
        else:
            raise Exception(argerr)
    elif pg.family == "D":
        if pg.subfamily == "h":
            dn_symels = _Dihn(pg.n, "C", "C_2'")
            if n_is_even:
                return _direct_product(dn_symels, i)
            else:
                return _direct_product(dn_symels, sh)
        elif pg.subfamily == "d":
            if n_is_even:
                return _Dihn(pg.n * 2, "S", "C_2'")
            else:
                dn_symels = _Dihn(pg.n, "C", "C_2'")
                return _direct_product(dn_symels, i)
        elif pg.subfamily is None:
            return _Dihn(pg.n, "C", "C_2'")
        else:
            raise Exception(argerr)
    elif pg.family == "S":
        if pg.subfamily is None and n_is_even:
            if n_is_doubleeven:
                return _Zn(pg.n, "S")
            else:
                cn_symels = _Zn(pg.n >> 1, "C")
                return _direct_product(cn_symels, i)
        else:
            raise Exception(argerr)
    else:
        if pg.family == "T":
            if pg.subfamily == "h":
                return TH_SYMELS
            elif pg.subfamily == "d":
                return TD_SYMELS
            else:
                return T_SYMELS
        elif pg.family == "O":
            if pg.subfamily == "h":
                return OH_SYMELS
            else:
                return O_SYMELS
        elif pg.family == "I":
            if pg.subfamily == "h":
                return IH_SYMELS
            else:
                return I_SYMELS
        else:
            raise Exception(argerr)
