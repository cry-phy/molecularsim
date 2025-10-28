import numpy as np
import numpy.typing as npt

from molecsim.data.data import FLOAT_DTYPE, INT_DTYPE, StructureArray


class ChargeTable:
    """Per-type charges aligned to a TypeRegistry."""

    def __init__(self, registry, charges_by_name: dict[str, float]) -> None:
        """Generate mapping."""
        reg_types = registry.types
        ks = set(charges_by_name)
        need = set(reg_types)
        if ks != need:
            missing = need - ks
            extra = ks - need
            msg = []
            if missing:
                msg.append(f"missing={sorted(missing)}")
            if extra:
                msg.append(f"unknown={sorted(extra)}")
            raise ValueError("ChargeTable: type/key mismatch: " + ", ".join(msg))

        self._registry = registry
        self._q = np.asarray([charges_by_name[t] for t in reg_types], dtype=FLOAT_DTYPE)
        self._by_name = dict(charges_by_name)

    @property
    def registry(self):
        """Return registry used to build."""
        return self._registry

    @property
    def q_arr(self) -> StructureArray:
        """Charges per type, aligned to registry order (copy)."""
        return self._q.copy()

    def charge_of_type(self, type_name: str) -> float:
        """Charge for a given type name."""
        idx = self._registry.index_of(type_name)
        return float(self._q[idx])

    def __getitem__(self, type_ids: npt.NDArray[INT_DTYPE]) -> StructureArray:
        """Map per-atom type IDs (N,) → per-atom charges (N,)."""
        return self._q[np.asarray(type_ids, dtype=INT_DTYPE)]
