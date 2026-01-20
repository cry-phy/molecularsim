from __future__ import annotations
from molecsim.data.data import SystemData
from collections.abc import Iterator, Mapping, Sequence
from typing import Iterable, Protocol, runtime_checkable, NamedTuple, Any
import types
import inspect

from molecsim.data.molecule import Molecule


@runtime_checkable
class PotContributor(Protocol):
    """Compute potential (and update accumulators via data) for this term."""

    def __call__(
        self,
        data: SystemData,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        """PotCall objects are essentially functions."""
        ...


@runtime_checkable
class PotCache(Protocol):
    """Potentials with cache needs must include."""

    def __update__(self, old: Contribution, new: Contribution):
        """Update dunder is used to update cached data."""
        ...


def validate_contributor(fn, args: Sequence[Any], kwargs: Mapping[str, Any]) -> None:
    """Raise TypeError if (data, *args, **kwargs) wouldn't call cleanly."""
    sig = inspect.signature(fn)
    try:
        sig.bind_partial(object(), *args, **kwargs)
    except TypeError as e:
        raise TypeError(
            f"Incompatible PotContributor '{getattr(fn, '__name__', fn)}'"
        ) from e


def _is_updateable(x: Any) -> bool:
    upd = getattr(x, "__update__", None)
    return callable(upd)


def _updater_base_name(obj: Any) -> str:
    return (
        getattr(obj, "name", None)
        or getattr(obj, "__name__", None)
        or obj.__class__.__name__
    )


def _collect_updaters(
    namespace: str,
    items: Sequence[PotCall],
) -> tuple[tuple[str, Any], ...]:
    seen: set[int] = set()
    counts: dict[str, int] = {}
    out: list[tuple[str, Any]] = []

    for pc in items:
        for a in list(pc.args) + list(pc.kwargs.values()):
            if not _is_updateable(a):
                continue
            xid = id(a)
            if xid in seen:
                continue
            seen.add(xid)

            potcall_name = pc.name or getattr(
                pc.fn, "__name__", pc.fn.__class__.__name__
            )
            base = _updater_base_name(a)
            n = counts.get(base, 0) + 1
            counts[base] = n
            key = f"{namespace}:{potcall_name}:{base}:{n}"
            out.append((key, a))

    return tuple(out)


class PotCall(NamedTuple):
    """Bundles call to Potential potential Contributor."""

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
    name = name
    return PotCall(
        fn=fn, args=tuple(args), kwargs=frozen_kwargs, name=name, enabled=enabled
    )


class KeyRegistry:
    """Shared registry for (updater <-> key) mapping and cache key conventions.

    This is intentionally minimal: no cache wrapper, just stable key lookup.
    """

    def __init__(self, namespace: str, updaters: tuple[tuple[str, Any], ...]) -> None:
        self._namespace = namespace
        self._updaters = updaters

        self._key_by_id: dict[int, str] = {id(obj): key for key, obj in updaters}
        self._obj_by_key: dict[str, Any] = dict(updaters)

    @property
    def namespace(self) -> str:
        return self._namespace

    def items(self) -> tuple[tuple[str, Any], ...]:
        return self._updaters

    def keys(self) -> tuple[str, ...]:
        return tuple(self._obj_by_key.keys())

    def updater_for(self, key: str) -> Any:
        return self._obj_by_key[key]

    def key_for(self, obj: Any) -> str:
        return self._key_by_id[id(obj)]

    def cache_key(self, updater_obj: Any, suffix: str) -> str:
        """Return the cache key for an updater, optionally with a suffix."""
        base = self.key_for(updater_obj)
        return base + ":" + suffix


class SystemPotentials:
    """Holds force/potential contributors and iterates over them immutably."""

    def __init__(
        self,
        items: Sequence[PotCall] | None = None,
        *,
        namespace: str | None = None,
    ) -> None:
        """Convert lists to single list of immutable dispatchable items.

        The namespace disambiguates updater keys across multiple SystemPotentials
        instances that may be applied to the same simulation.
        """
        if namespace is None:
            namespace = f"auto-{id(self):x}"
        self._namespace = namespace

        self._items: tuple[PotCall, ...] = tuple(items or ())
        self._updaters: tuple[tuple[str, Any], ...] = _collect_updaters(
            self._namespace, self._items
        )

        self._registry = KeyRegistry(self._namespace, self._updaters)

    @property
    def namespace(self) -> str:
        """Return namespace of this SystemPotentials instance."""
        return self._namespace

    @property
    def registry(self) -> KeyRegistry:
        """Return the shareable key registry for this SystemPotentials instance."""
        return self._registry

    # Keep your existing API but delegate to registry (minimal invasive)
    def get_updaters(self) -> tuple[tuple[str, Any], ...]:
        """Return iterable of all updatable objects as (key, object) pairs."""
        return self._registry.items()

    def updater_for(self, key: str) -> Any:
        """Return updater object associated with a specific key."""
        return self._registry.updater_for(key)

    def updater_keys(self) -> tuple[str, ...]:
        """Return iterable of unique updater keys."""
        return self._registry.keys()

    def key_for(self, obj: Any) -> str:
        """Return key for a specific updater object."""
        return self._registry.key_for(obj)

    def __iter__(self) -> Iterator[PotCall]:
        """Generate iterator for loops."""
        yield from self._items

    def __len__(self) -> int:
        """Return Number of Contributors."""
        return len(self._items)

    def __getitem__(self, i: int) -> PotCall:
        """Retrieve Contributor by index."""
        return self._items[i]

    def with_added(self, *items: PotCall) -> SystemPotentials:
        """Pure add: returns a new SystemPotentials with extra items appended."""
        return SystemPotentials(self._items + tuple(items))

    def enabled(self) -> SystemPotentials:
        """Filter enabled terms (chainable)."""
        return SystemPotentials(
            [pc for pc in self._items if pc.enabled],
            namespace=self._namespace,
        )

    @staticmethod
    def from_parallel(
        contributors: Sequence[PotContributor],
        args: Sequence[Sequence[Any]],
        kwargs: Sequence[Mapping[str, Any]],
    ) -> SystemPotentials:
        """Migrates from parallel lists."""
        if not (len(contributors) == len(args) == len(kwargs)):
            raise ValueError("contributors, args, kwargs must have equal length")
        items = [
            PotCall(fn=c, args=tuple(a), kwargs=types.MappingProxyType(dict(k)))
            for c, a, k in zip(contributors, args, kwargs, strict=True)
        ]
        return SystemPotentials(items)

    def with_enabled(self, *names: str) -> SystemPotentials:
        """Enable set of ciontributors."""
        names_set = set(names)

        def _toggle(pc: PotCall) -> PotCall:
            if pc.name in names_set:
                return PotCall(pc.fn, pc.args, pc.kwargs, pc.name, True)
            return pc

        return SystemPotentials(tuple(_toggle(pc) for pc in self._items))

    def with_disabled(self, *names: str) -> SystemPotentials:
        """Disable set of contrributors."""
        names_set = set(names)

        def _toggle(pc: PotCall) -> PotCall:
            if pc.name in names_set:
                return PotCall(pc.fn, pc.args, pc.kwargs, pc.name, False)
            return pc

        return SystemPotentials(tuple(_toggle(pc) for pc in self._items))

    def names(self) -> tuple[str | None, ...]:
        """List all contributors."""
        return tuple(pc.name for pc in self._items)


def cache_value(
    updater: Any,
    suffix: str,
    value: Any,
    contribution: Contribution,
) -> Any:
    """Store value in contribution cache under updater's key + suffix."""
    reg = contribution.registry
    key = reg.cache_key(updater, suffix)
    contribution.cache[key] = value
    return value


def uncache_value(
    updater: Any,
    suffix: str,
    contribution: Contribution,
) -> Any:
    """Fetch value from contribution cache, optionally returning default."""
    reg = contribution.registry

    key = reg.cache_key(updater, suffix)
    return contribution.cache.get(key)


class Contribution:
    """Potential + per-updater cached payloads for a proposed state."""

    def __init__(
        self,
        potential: Iterable[float],
        molecule: Any,
        cache: Mapping[str, Any],
        registry: KeyRegistry,
    ) -> None:
        """Build contribution."""
        self._potential = list(potential)
        self._cache = dict(cache)
        self._molecule = molecule
        self._registry = registry

    @property
    def registry(self) -> KeyRegistry:
        """Access registry directly."""
        return self._registry

    @property
    def potential(self) -> float:
        """Total potential for this proposal."""
        return sum(self._potential)

    def add_potential(self, potential: float) -> None:
        """Add potential."""
        self._potential.append(potential)

    def get_potentials(self) -> list[float]:
        """Get all potentials split by contributor."""
        return self._potential.copy()

    @property
    def cache(self) -> dict[str, Any]:
        """Cached per-updater payloads keyed by updater keys."""
        return self._cache

    @property
    def molecule(self) -> Any:
        """Optional reference to the molecule/state this contribution corresponds to."""
        return self._molecule


def empty_contribution(
    molecule: Molecule,
    potentials: SystemPotentials,
) -> Contribution:
    """Generate empty contribution."""
    return Contribution((), molecule=molecule, cache={}, registry=potentials.registry)
