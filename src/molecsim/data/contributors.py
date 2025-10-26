from __future__ import annotations
from molecsim.data.data import SystemData
from collections.abc import Iterator, Mapping, Sequence
from typing import Protocol, runtime_checkable, NamedTuple, Any
import types
import inspect


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


class SystemPotentials:
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

    def with_added(self, *items: PotCall) -> SystemPotentials:
        """Pure add: returns a new SystemPotentials with extra items appended."""
        return SystemPotentials(self._items + tuple(items))

    def enabled(self) -> SystemPotentials:
        """Filter enabled terms (chainable)."""
        return SystemPotentials([pc for pc in self._items if pc.enabled])

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
