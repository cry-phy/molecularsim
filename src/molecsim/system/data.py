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
INT_DTYPE: Final = np.int32

type FloatPrecision = np.float64
type StructureArray = npt.NDArray[FloatPrecision]
type StructureView = StructureArray


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
        self._atom_int_to_str: npt.NDArray[np.object_] = np.full(256, "NotFound")

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


class StructureNode:
    """Node in the Structure Tree."""

    def __init__(
        self, molecular_index: int, neighbors: Iterable[StructureNode]
    ) -> None:
        """Generate empty node."""
        self.molecular_index: int = molecular_index
        self.neighbors: list[StructureNode] = list(neighbors)

    def __repr__(self) -> str:
        """Give simple representation."""
        return f"""<StructureNode: {self.molecular_index}, {[neigh.molecular_index for neigh in self.neighbors]}>"""


class SystemProperties:
    """Immutable wrapper for calculating and storing system properties.

    Stores NVT Parameters:
        N: Number molecules (Dict of types)
        V: Volume through box boundaries (Angstrom)
        T: Kelvin
    """

    _k_b = 1  # by definition
    # Coulomb coefficient:
    # 1/(4*pi*epsilon_0) = 8.98e9 N*m²/C²
    _coulomb_coeff = 1.671009e5  # K*A/e²

    # LJ parameters (TIP3P oxygen):
    # epsilon = 0.1521 kcal/mol
    # 0.1521*503.2 = 76.514 K
    # sigma = 3.15 A
    # A = 4*epsilon*sigma¹², B = 4*epsilon*sigma⁶
    _LJ_A = 2.928737e8  # K*A¹²
    _LJ_B = 2.994156e5  # K*A⁶

    def __init__(
        self,
        molecules: dict[Molecule, int],
        box: StructureArray,
        temperature: float,
        periodic: bool = True,
    ) -> None:
        """Initialize data."""
        self._molecules: dict[Molecule, int] = molecules  # Count
        self._box: StructureArray = box  # Angstrom
        self._temperature: float = temperature  # Kelvin
        self.periodic: bool = periodic

    @property
    def box(self):
        """The box property."""
        return self._boxPrope

    @property
    def molecules(self) -> dict[Molecule, int]:
        """The molecules property."""
        return self._molecules

    @property
    def temperature(self):
        """The temperature property."""
        return self._temperature

    @property
    def heat_energy(self):
        """The heat_energy property."""
        return self._temperature * self._k_b

    @property
    def coulomb_coeff(self):
        """The coulomb_coeff property."""
        return self._coulomb_coeff

    @property
    def lj_params(self):
        """The lj_params property."""
        return self._LJ_A, self._LJ_B


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

    def label(self) -> str:
        """Return name for debugging/human readability."""
        return self.name or getattr(self.fn, "__name__", "<unnamed>")


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
    name = fn.__name__ if not name else name
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

    def with_enabled(self, *names: str) -> SystemForces:
        """Enable set of ciontributors."""
        names_set = set(names)

        def _toggle(pc: PotCall) -> PotCall:
            if pc.name in names_set:
                return PotCall(pc.fn, pc.args, pc.kwargs, pc.name, True)
            return pc

        return SystemForces(tuple(_toggle(pc) for pc in self._items))

    def with_disabled(self, *names: str) -> SystemForces:
        """Disable set of contrributors."""
        names_set = set(names)

        def _toggle(pc: PotCall) -> PotCall:
            if pc.name in names_set:
                return PotCall(pc.fn, pc.args, pc.kwargs, pc.name, False)
            return pc

        return SystemForces(tuple(_toggle(pc) for pc in self._items))

    def names(self) -> tuple[str | None, ...]:
        """List all contributors."""
        return tuple(pc.name for pc in self._items)
