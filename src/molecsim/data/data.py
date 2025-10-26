from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Final
from collections.abc import Iterable


FLOAT_DTYPE: Final = np.float64
INT_DTYPE: Final = np.int32

type FloatPrecision = np.float64
type StructureArray = npt.NDArray[FloatPrecision]
type StructureView = StructureArray

NO_SYMBOL: str = "NotFound"


class SystemData:
    """Contains Data encoded for simulation system usage."""

    dims = 3

    def __init__(self, n: int) -> None:
        """Generate zeroed System Data for *n* Atoms."""
        self._positions: StructureArray | StructureView = np.zeros(
            (n, 3), dtype=FLOAT_DTYPE
        )
        self._charges: StructureArray | StructureView = np.zeros(n, dtype=FLOAT_DTYPE)
        self._atoms: npt.NDArray[INT_DTYPE] = np.zeros(n, dtype=INT_DTYPE)
        self._atom_str_to_int: dict[str, int] = {}
        self._atom_int_to_str: npt.NDArray[np.object_] = np.full(256, NO_SYMBOL)

        self._first_empty: int = 0
        self._start_index: int = 0

    @classmethod
    def as_owner(
        cls,
        atoms: Iterable[str],
        positions: StructureArray,
        charges: StructureArray,
        mapping: tuple[dict[str, int], npt.NDArray[np.object_]] | None = None,
    ) -> SystemData:
        """Generate Data as a copy where ownership is with this instance."""
        n = len(charges)
        instance = cls(n)

        if not mapping:
            ordered = list(dict.fromkeys(atoms))
            s2i = {s: i for i, s in enumerate(ordered)}
            i2s = np.array(ordered, dtype=object)
        else:
            s2i, i2s = mapping

        instance._atom_str_to_int = s2i
        np.copyto(
            instance._atom_int_to_str[: len(i2s)],
            i2s.astype(instance._atom_int_to_str.dtype),
        )
        instance._atoms = np.fromiter((s2i[a] for a in atoms), dtype=INT_DTYPE, count=n)

        instance._positions = positions.copy()
        instance._charges = charges.copy()
        return instance

    def as_borrow(self, start: int, size: int) -> SystemData:
        """Generate Data as a copy where ownership is with this instance."""
        instance = SystemData(size)
        stop = start + size
        instance._atoms = self._atoms[start:stop]
        instance._positions = self._positions[start:stop]
        instance._charges = self._charges[start:stop]

        instance._atom_int_to_str = self._atom_int_to_str.view()
        instance._atom_str_to_int = self._atom_str_to_int

        instance._start_index = start
        return instance

    def _ensure_symbols(self, symbols: Iterable[str]) -> None:
        s2i = self._atom_str_to_int
        i2s = self._atom_int_to_str

        unique_syms = list(dict.fromkeys(symbols))
        for s in unique_syms:
            if s in s2i:
                continue
            next_idx = len(s2i)
            if next_idx >= i2s.size:
                raise RuntimeError(
                    "Atom type table capacity exceeded; enlarge _atom_int_to_str."
                )
            s2i[s] = next_idx
            i2s[next_idx] = s

    def atom_ids_to_str(self, ids: npt.NDArray[INT_DTYPE]) -> npt.NDArray[np.object_]:
        """Fast conversion from ids to strings."""
        return self._atom_int_to_str[ids]

    def atom_str_to_ids(self, strings: Iterable[str]) -> npt.NDArray[INT_DTYPE]:
        """Fast conversion from strings to ids."""
        self._ensure_symbols(strings)
        s2i = self._atom_str_to_int
        return np.fromiter((s2i[s] for s in strings), dtype=INT_DTYPE)

    def atoms_as_str(self) -> npt.NDArray[np.object_]:
        """Return full system atom types."""
        return self._atom_int_to_str[self._atoms]

    def allocate_molecule(
        self,
        size: int,
    ) -> SystemData:
        """Generate Data as a copy where ownership is with this instance."""
        new_data = self.as_borrow(self.first_empty, size)
        self._first_empty += size
        return new_data

    def copy_from(self, data: SystemData) -> None:
        """Copy from another SystemData, translating atom types and extending this mapping if needed."""
        if (
            self._atoms.shape != data._atoms.shape
            or self._positions.shape != data._positions.shape
            or self._charges.shape != data._charges.shape
        ):
            raise ValueError(
                "copy_from: shape mismatch between destination view and source data"
            )

        # can skip if the instances are the same ie other is borrow of this data
        if not (
            np.shares_memory(self._atom_int_to_str, data._atom_int_to_str)
            and self._atom_str_to_int is data._atom_str_to_int
        ):
            src_syms = data._atom_int_to_str[data._atoms]
            self._ensure_symbols(src_syms)
            dest_ids = np.fromiter(
                (self._atom_str_to_int[s] for s in src_syms),
                dtype=INT_DTYPE,
                count=src_syms.shape[0],
            )
        else:
            dest_ids = data.atoms

        # Copy over
        np.copyto(self._atoms, dest_ids, casting="no")
        np.copyto(self._positions, data._positions, casting="no")
        np.copyto(self._charges, data._charges, casting="no")

    def copy_data(self) -> SystemData:
        """Deep copy arrays; share the same symbol mapping tables."""
        copy = SystemData.as_owner(
            atoms=self.atoms_as_str(),
            positions=self._positions.copy(),
            charges=self._charges.copy(),
            mapping=(self._atom_str_to_int, self._atom_int_to_str),
        )
        copy._start_index = self._start_index
        copy._first_empty = self._first_empty
        return copy

    @property
    def positions(self):
        """The positions property."""
        return self._positions

    @positions.setter
    def positions(self, positions: StructureArray | StructureView) -> None:
        np.copyto(self._positions, positions)

    @property
    def charges(self):
        """The charges property."""
        return self._charges

    @charges.setter
    def charges(self, value):
        np.copyto(self._charges, value)

    @property
    def atoms(self):
        """The atoms property."""
        return self._atoms

    @atoms.setter
    def atoms(self, value):
        self._atoms = value

    @property
    def first_empty(self):
        """The first_empty property."""
        return self._first_empty

    def __len__(self) -> int:
        """Return number of atoms."""
        return len(self._atoms)

    def data_interval(self) -> tuple[int, int]:
        """Return start and end index of data in owner of data."""
        return self._start_index, self._start_index + self.__len__()

    @property
    def symbols(self):
        all_symbols = self._atom_int_to_str.copy()
        return all_symbols[np.where(all_symbols != NO_SYMBOL)]


