from __future__ import annotations
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Final
import typing
import numpy as np
import numpy.typing as npt
from typing import Protocol, runtime_checkable, NamedTuple
import types
import inspect

if typing.TYPE_CHECKING:
    from molecsim.system.molecule import Molecule

FLOAT_DTYPE: Final = np.float64

type FloatPrecision = np.float64
type StructureArray = npt.NDArray[FloatPrecision]
type StructureView = StructureArray


class SystemData:
    """Contains Data encoded for simulation system usage."""

    dims = 3

    def __init__(self, n: int) -> None:
        """Generate zeroed System Data for *n* Atoms."""
        self._positions: StructureArray | StructureView = np.zeros(
            shape=(n, 3), dtype=FLOAT_DTYPE
        )
        self._charges: StructureArray | StructureView = np.zeros(n, dtype=FLOAT_DTYPE)
        self._atoms: npt.NDArray[np.object_] = np.array([""] * n)

    @classmethod
    def as_owner(
        cls, atoms: Iterable[str], positions: StructureArray, charges: StructureArray
    ) -> SystemData:
        """Generate Data as a copy where ownership is with this instance."""
        instance = cls(len(charges))
        instance._atoms = np.asarray(atoms, copy=True)
        instance._positions = positions.copy()
        instance._charges = charges.copy()
        return instance

    @classmethod
    def as_borrow(
        cls,
        atoms: npt.NDArray[np.object_],
        positions: StructureArray | StructureView,
        charges: StructureArray | StructureView,
        start: int = 0,
        end: int = -1,
    ) -> SystemData:
        """Generate Data as a copy where ownership is with this instance."""
        instance = cls(len(charges))
        instance._atoms = atoms.view()[start:end]
        instance._positions = positions.view()[start:end]
        instance._charges = charges.view()[start:end]
        return instance

    def copy_from(self, data: SystemData) -> None:
        """Copy data from another system into this location."""
        np.copyto(self._atoms, data._atoms)
        np.copyto(self._positions, data._positions)
        np.copyto(self._charges, data._charges)

    def deepcopy(self) -> SystemData:
        """Generate copy."""
        return SystemData.as_owner(
            self._atoms.copy(),
            self._positions.copy(),
            self._charges.copy(),
        )

    @property
    def positions(self):
        """The positions property."""
        return self._positions

    @positions.setter
    def positions(self, positions: StructureArray | StructureView) -> None:
        np.copyto(self._positions, positions)

    def __len__(self) -> int:
        """Return number of atoms."""
        return len(self._atoms)


class StructureNode:
    """Node in the Structure Tree."""

    def __init__(
        self, molecular_index: int, neighbors: Iterable[StructureNode]
    ) -> None:
        """Generate empty node."""
        self.molecular_index: int = molecular_index
        self.neighbors: list[StructureNode] = list(neighbors)


class SystemProperties:
    """Immutable wrapper for calculating and storing system properties.

    Stores NVT Parameters:
        N: Number molecules (Dict of types)
        V: Volume through box boundaries (Angstrom)
        T: Kelvin
    """

    def __init__(
        self,
        molecules: dict[Molecule, int],
        box: StructureArray,
        temperature: float,
    ) -> None:
        """Initialize data."""
        self._molecules: dict[Molecule, int] = molecules  # Count
        self._box: StructureArray = box  # Angstrom
        self._temperature: float = temperature  # Kelvin

    @property
    def box(self):
        """The box property."""
        return self._box

    @property
    def molecules(self):
        """The molecules property."""
        return self._molecules

    @property
    def temperature(self):
        """The temperature property."""
        return self._temperature


@runtime_checkable
class PotContributor(Protocol):
    """Compute energy (and update accumulators via data) for this term."""

    def __call__(self, data: SystemData, *args: Any, **kwargs: Any) -> float: ...


def validate_contributor(fn, args: Sequence[Any], kwargs: Mapping[str, Any]) -> None:
    """Raise TypeError if (data, *args, **kwargs) wouldn't call cleanly."""
    sig = inspect.signature(fn)
    try:
        sig.bind_partial(object(), *args, **kwargs)
    except TypeError as e:
        raise TypeError(
            f"Incompatible PotContributor '{getattr(fn, '__name__', fn)}'"
        ) from e


class PotCall(NamedTuple):
    """Bundles call to Potential Energy Contributor."""

    fn: PotContributor
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = types.MappingProxyType({})
    name: str | None = None
    enabled: bool = True

    def __iter__(self):
        """Make iterable."""
        yield self.fn
        yield self.args
        yield self.kwargs
        yield self.name
        yield self.enabled


def make_potcall(
    fn: PotContributor,
    *args: Any,
    name: str | None = None,
    enabled: bool = True,
    **kwargs: Any,
) -> PotCall:
    """Generate dispatch ready potential call."""
    frozen_kwargs: Mapping[str, Any] = (
        types.MappingProxyType(dict(kwargs)) if kwargs else types.MappingProxyType({})
    )
    validate_contributor(fn, args, kwargs)
    return PotCall(
        fn=fn, args=tuple(args), kwargs=frozen_kwargs, name=name, enabled=enabled
    )


class SystemForces:
    """Holds force/potential contributors and iterates over them immutably."""

    def __init__(self, items: Sequence[PotCall] | None = None) -> None:
        """Convert lists to single list of immutable dispatchable items."""
        self._items: tuple[PotCall, ...] = tuple(items or ())

    def __iter__(self) -> Iterator[PotCall]:
        """Generate iterator for loops."""
        yield from self._items

    def __len__(self) -> int:
        """Return Number of Contributors."""
        return len(self._items)

    def __getitem__(self, i: int) -> PotCall:
        """Retrieve Contributor by index."""
        return self._items[i]

    def with_added(self, *items: PotCall) -> SystemForces:
        """Pure add: returns a new SystemForces with extra items appended."""
        return SystemForces(self._items + tuple(items))

    def enabled(self) -> SystemForces:
        """Filter enabled terms (chainable)."""
        return SystemForces([pc for pc in self._items if pc.enabled])

    @staticmethod
    def from_parallel(
        contributors: Sequence[PotContributor],
        args: Sequence[Sequence[Any]],
        kwargs: Sequence[Mapping[str, Any]],
    ) -> SystemForces:
        """Migrates from parallel lists."""
        if not (len(contributors) == len(args) == len(kwargs)):
            raise ValueError("contributors, args, kwargs must have equal length")
        items = [
            PotCall(fn=c, args=tuple(a), kwargs=types.MappingProxyType(dict(k)))
            for c, a, k in zip(contributors, args, kwargs, strict=True)
        ]
        return SystemForces(items)
