"""MolSymPy reference-structure collections.

Usage
-----
    from molsympy.collections import symmetrized, unsymmetrized

    # idealized / symmetrized structures
    for name in symmetrized.names:
        atoms = symmetrized[name]

    # raw / unsymmetrized structures
    atoms = unsymmetrized['C2v_1']
    print(unsymmetrized.point_groups)
"""

import numpy as np
from pathlib import Path
from importlib import resources
from ase import Atoms


def _load_npz(sym: bool) -> "np.lib.npyio.NpzFile":
    filename = "MolSymPy_sym.npz" if sym else "MolSymPy_unsym.npz"

    # 1. Installed package / editable install: next to this __init__.py
    try:
        with resources.as_file(
            resources.files("molsympy.collections").joinpath(filename)
        ) as p:
            return np.load(p, allow_pickle=True)
    except Exception:
        pass

    # 2. Source tree fallback: collections/ directory
    local = Path(__file__).parent / filename
    if local.exists():
        return np.load(local, allow_pickle=True)

    # 3. Development fallback: sibling molsympy_db/ directory
    dev = Path(__file__).parent.parent.parent.parent / "molsympy_db" / filename
    return np.load(dev, allow_pickle=True)


class MolSymPyCollection:
    """Collection of molecular reference structures indexed by point group.

    Keys have the form ``'{PointGroup}_{index}'`` (e.g. ``'C2v_1'``,
    ``'Td_3'``, ``'Ih_1'``).

    Attributes
    ----------
    symmetrized : bool
        True if this collection contains idealized/symmetrized structures.
    names : list[str]
        Sorted list of all available keys.
    point_groups : list[str]
        Sorted list of unique Schoenflies symbols in the collection.

    Examples
    --------
    >>> from molsympy.collections import symmetrized, unsymmetrized
    >>> for name in symmetrized.names:
    ...     atoms = symmetrized[name]
    >>> atoms = unsymmetrized['C2v_1']
    """

    def __init__(self, db, is_symmetrized: bool = False):
        self._db = db
        self.symmetrized = is_symmetrized
        self._formula_to_key: dict[str, str] = {}
        for k in db.files:
            try:
                f = db[k].item().get("formula", "")
                if f:
                    self._formula_to_key[str(f).strip()] = k
            except Exception:
                pass

    # ── collection interface ──────────────────────────────────────────────────

    @property
    def names(self) -> list[str]:
        """Sorted list of all keys in the collection."""
        return sorted(self._db.files)

    @property
    def point_groups(self) -> list[str]:
        """Sorted list of unique Schoenflies symbols."""
        return sorted({k.rsplit("_", 1)[0] for k in self._db.files})

    @property
    def formulas(self) -> list[str]:
        """Sorted list of molecular formulas available in the collection."""
        return sorted(self._formula_to_key)

    def __len__(self) -> int:
        return len(self._db.files)

    def __contains__(self, key: str) -> bool:
        return key in self._db or key in self._formula_to_key

    def __iter__(self):
        return iter(self.names)

    def __repr__(self) -> str:
        kind = "symmetrized" if self.symmetrized else "unsymmetrized"
        return (
            f"MolSymPyCollection({len(self)} structures, "
            f"{len(self.point_groups)} point groups, {kind})"
        )

    # ── item access ───────────────────────────────────────────────────────────

    def __getitem__(self, key: str) -> Atoms:
        """Return the structure for *key* as an ASE Atoms object.

        Parameters
        ----------
        key : str
            Database key of the form ``'{PointGroup}_{index}'``,
            e.g. ``'C2v_1'``.

        Returns
        -------
        ase.Atoms
            Atoms object with ``atoms.info`` populated:
            ``'point_group'``, ``'index'``, ``'energy'``, ``'dataset'``.

        Raises
        ------
        KeyError
            If *key* is not in the collection.
        """
        if key not in self._db:
            # Try resolving as a molecular formula (e.g. 'CNH' → 'C0v_1')
            if key in self._formula_to_key:
                key = self._formula_to_key[key]
            else:
                pg = key.rsplit("_", 1)[0] if "_" in key else key
                available = [k for k in self._db.files if k.rsplit("_", 1)[0] == pg]
                hint = (
                    f"Available for '{pg}': {sorted(available)}"
                    if available
                    else f"Known point groups: {self.point_groups}"
                )
                raise KeyError(f"'{key}' not in collection. {hint}")

        data = self._db[key].item()
        atoms = Atoms(symbols=list(data["elements"]), positions=data["positions"])
        atoms.info["point_group"] = data["point_group"]
        atoms.info["index"]       = data["index"]
        atoms.info["energy"]      = data["energy"]
        atoms.info["dataset"]     = data["dataset"]
        return atoms

    def get(self, key: str, symbol: str | None = None) -> Atoms:
        """Like ``__getitem__`` but optionally override all chemical symbols.

        Parameters
        ----------
        key : str
            Database key (e.g. ``'Td_1'``).
        symbol : str or None
            If given, replace every element with this symbol (e.g. ``'Mo'``).

        Returns
        -------
        ase.Atoms
        """
        atoms = self[key]
        if symbol is not None:
            atoms.symbols = [symbol] * len(atoms)
        return atoms


symmetrized   = MolSymPyCollection(_load_npz(sym=True),  is_symmetrized=True)
unsymmetrized = MolSymPyCollection(_load_npz(sym=False), is_symmetrized=False)
