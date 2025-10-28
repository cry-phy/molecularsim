from __future__ import annotations
import numpy as np
import numpy.typing as npt

INT_DTYPE = np.int32


class TypeRegistry:
    """Canonical mapping between atom names and integer type IDs."""

    def __init__(self, types: list[str]) -> None:
        """Generate mapping from symbols to indices."""
        ordered: list[str] = []
        seen: set[str] = set()
        for t in types:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        self._types: list[str] = ordered
        self._type_to_idx: dict[str, int] = {t: i for i, t in enumerate(ordered)}

    def ids(self, names: list[str] | npt.NDArray[np.object_]) -> npt.NDArray[np.int32]:
        """Convert list/array of type names to integer IDs."""
        return np.asarray([self._type_to_idx[str(n)] for n in names], dtype=INT_DTYPE)

    def names(self, ids: npt.NDArray[np.int32]) -> npt.NDArray[np.object_]:
        """Convert integer IDs to type names."""
        arr = np.asarray(self._types, dtype=object)
        return arr[ids]

    @property
    def types(self) -> list[str]:
        """Return all Symbols."""
        return self._types

    def index_of(self, name: str) -> int:
        """Get single index of Symbol."""
        return self._type_to_idx[name]
