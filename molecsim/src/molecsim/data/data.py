from __future__ import annotations
import numpy as np
import numpy.typing as npt


FLOAT_DTYPE = np.float64
INT_DTYPE = np.int32
COMPLEX_TYPE = np.complex128

type ComplexPrecision = COMPLEX_TYPE
type FloatPrecision = FLOAT_DTYPE
type StructureArray = npt.NDArray[FloatPrecision]
type StructureView = StructureArray


class SystemData:
    """State-only container: positions, charges, integer type IDs."""

    dims = 3

    def __init__(self, n: int) -> None:
        """Generate empty Data arrays."""
        self._positions: StructureArray = np.zeros((n, 3), dtype=FLOAT_DTYPE)
        self._velocities: StructureArray = np.zeros((n, 3), dtype=FLOAT_DTYPE)
        self._charges: npt.NDArray[FloatPrecision] = np.zeros(n, dtype=FLOAT_DTYPE)
        self._types: npt.NDArray[INT_DTYPE] = np.zeros(n, dtype=INT_DTYPE)

        self._start_index: int = 0  # index in parent, if this is a view
        self._first_empty: int = 0  # optional allocator cursor

    @classmethod
    def as_owner(
        cls,
        types: npt.NDArray[INT_DTYPE],
        positions: StructureArray,
        charges: StructureArray,
    ) -> SystemData:
        """Create an owning copy from arrays."""
        n = len(charges)
        if positions.shape != (n, 3) or types.shape != (n,):
            raise ValueError("as_owner: shape mismatch among types/positions/charges.")
        inst = cls(n)
        np.copyto(inst._types, types)
        np.copyto(inst._positions, positions)
        np.copyto(inst._charges, charges)
        return inst

    def as_borrow(self, start: int, size: int) -> SystemData:
        """Create a non-owning view [start:start+size] into this data."""
        stop = start + size
        if not (0 <= start <= stop <= len(self)):
            raise IndexError("as_borrow: slice out of range.")
        inst = SystemData(size)
        inst._positions = self._positions[start:stop]
        inst._charges = self._charges[start:stop]
        inst._types = self._types[start:stop]
        inst._start_index = start
        inst._first_empty = 0
        return inst

    def copy_data(self) -> SystemData:
        """Deep-copy arrays; keep start/first_empty metadata."""
        cp = SystemData.as_owner(
            self._types.copy(), self._positions.copy(), self._charges.copy()
        )
        cp._start_index = self._start_index
        cp._first_empty = self._first_empty
        return cp

    def copy_from(self, src: SystemData) -> None:
        """Copy atoms, positions, charges from src. Assumes same registry/IDs."""
        if (
            self._types.shape != src._types.shape
            or self._positions.shape != src._positions.shape
            or self._velocities.shape != src._velocities.shape
            or self._charges.shape != src._charges.shape
        ):
            raise ValueError("copy_from: shape mismatch between destination and source")

        # Fast no-op if all arrays already share memory
        if (
            np.shares_memory(self._types, src._types)
            and np.shares_memory(self._positions, src._positions)
            and np.shares_memory(self._velocities, src._velocities)
            and np.shares_memory(self._charges, src._charges)
        ):
            return

        np.copyto(self._types, src._types, casting="no")
        np.copyto(self._positions, src._positions, casting="no")
        np.copyto(self._velocities, src._velocities, casting="no")
        np.copyto(self._charges, src._charges, casting="no")

    def allocate_molecule(self, size: int) -> SystemData:
        """Allocate a contiguous view."""
        start = self._first_empty
        stop = start + size
        if stop > len(self):
            avail = len(self) - start
            raise ValueError(
                f"allocate_molecule: insufficient capacity (need {size}, have {avail})"
            )
        view = self.as_borrow(start, size)
        self._first_empty = stop
        return view

    @property
    def positions(self) -> StructureArray:
        """The positions property."""
        return self._positions

    @positions.setter
    def positions(self, positions: StructureArray) -> None:
        if positions.shape != self._positions.shape:
            raise ValueError("positions: shape mismatch")
        np.copyto(self._positions, positions)

    @property
    def velocities(self) -> StructureArray:
        """The positions property."""
        return self._velocities

    @velocities.setter
    def velocities(self, velocities: StructureArray) -> None:
        if velocities.shape != self._velocities.shape:
            raise ValueError("positions: shape mismatch")
        np.copyto(self._velocities, velocities)

    @property
    def charges(self) -> StructureArray:
        """The charges property."""
        return self._charges

    @charges.setter
    def charges(self, value: npt.NDArray[FloatPrecision]) -> None:
        if value.shape != self._charges.shape:
            raise ValueError("charges: shape mismatch")
        np.copyto(self._charges, value)

    @property
    def atoms(self) -> npt.NDArray[INT_DTYPE]:
        """The types property."""
        return self._types

    @atoms.setter
    def atoms(self, value: npt.NDArray[INT_DTYPE]) -> None:
        if value.shape != self._types.shape:
            raise ValueError("types: shape mismatch")
        np.copyto(self._types, value)

    @property
    def first_empty(self) -> int:
        """The first empty atom index for placement."""
        return self._first_empty

    def __len__(self) -> int:
        """Length of Data."""
        return int(self._types.shape[0])

    def data_interval(self) -> tuple[int, int]:
        """Start/end indices relative to parent owner (if this is a view)."""
        return self._start_index, self._start_index + len(self)
