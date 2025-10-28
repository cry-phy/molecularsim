from collections.abc import Callable, Iterable

from molecsim.data.data import StructureArray

import numpy as np

type MixingRule = Callable[[Iterable[float], Iterable[float]], StructureArray]


def lbmixing(epsilons: StructureArray, sigmas: StructureArray) -> StructureArray:
    """Lorentz–Berthelot mixing with shape (T, T, 2)."""
    eps = np.asarray(epsilons, dtype=float)
    sig = np.asarray(sigmas, dtype=float)

    sig_ij = 0.5 * (sig[:, None] + sig[None, :])
    eps_ij = np.sqrt(eps[:, None] * eps[None, :])

    sig6 = sig_ij**6
    b_table = 4.0 * eps_ij * sig6
    a_table = b_table * sig6

    return np.stack([a_table, b_table], axis=-1)


class LJTable:
    """Lennard-Jones parameters in kcal.

    Aligned to a shared TypeRegistry.
    """

    def __init__(
        self,
        registry,
        epsilons: dict[str, float],
        sigmas: dict[str, float],
        mixing_rule,
    ) -> None:
        """Build table and check for inconsistencies."""
        key_set = set(registry.types)
        if set(epsilons) != set(sigmas):
            missing_in_sig = set(epsilons) - set(sigmas)
            missing_in_eps = set(sigmas) - set(epsilons)
            raise ValueError(
                "LJTable: type keys mismatch. "
                f"missing in sigmas={sorted(missing_in_sig)}; "
                f"missing in epsilons={sorted(missing_in_eps)}"
            )
        missing_any = key_set - set(epsilons)
        if missing_any:
            raise ValueError(
                f"LJTable: parameters missing for registry types: {sorted(missing_any)}"
            )

        self._registry = registry
        self._mixing_rule = mixing_rule
        order = registry.types
        self._eps_arr = np.asarray([epsilons[t] for t in order], dtype=float)
        self._sig_arr = np.asarray([sigmas[t] for t in order], dtype=float)
        self._epsilons = dict(epsilons)
        self._sigmas = dict(sigmas)
        self._table = mixing_rule(self._eps_arr, self._sig_arr)

    @property
    def registry(self):
        """Show registry used to Build this."""
        return self._registry

    @property
    def eps_arr(self):
        """The _eps_arr property."""
        return self._eps_arr.copy()

    @property
    def sig_arr(self):
        """The sig_arr property."""
        return self._sig_arr.copy()

    @property
    def table(self):
        """Pair table returned by mixing_rule."""
        return self._table

    def index_of(self, name: str) -> int:
        """Forward to registry for convenience."""
        return self._registry.index_of(name)

    def types(self) -> list[str]:
        """All Symbols."""
        return list(self._registry.types)
