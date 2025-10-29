from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pyarrow import StructArray
from molecsim.data.chargetable import ChargeTable
from molecsim.data.ljtable import LJTable
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
from molecsim.data.typeregistry import Atom, TypeRegistry
from molecsim.utils.parsers import parse_com


def symbol_param_dict(atoms: Iterable[Atom], param: str) -> dict[str, Any]:
    """Generate dict symbol: param."""
    return {atom.symbol: atom.__getattribute__(param) for atom in atoms}


def generate_reg_q_lj(
    atoms: Iterable[Atom],
) -> tuple[TypeRegistry, ChargeTable, LJTable]:
    """Generate Tables from list of atoms."""
    symbols = [atom.symbol for atom in atoms]
    reg = TypeRegistry(symbols)
    qtable = ChargeTable(reg, symbol_param_dict(atoms, "charge"))
    ljtable = LJTable(
        reg,
        symbol_param_dict(atoms, "lj_eps"),
        symbol_param_dict(atoms, "lj_sigma"),
    )
    return reg, qtable, ljtable


def tip3p() -> ModelConstants:
    """Generate System Properties for TIP3P model."""
    defined_types = [
        Atom(
            symbol="H",
            charge=0.417,
            mass=1,
            lj_eps=0.0,
            lj_sigma=0.0,
        ),
        Atom(
            symbol="O",
            charge=-2 * 0.417,
            mass=16,
            lj_eps=0.1521,
            lj_sigma=3.1507,
        ),
    ]
    reg, qtable, ljtab = generate_reg_q_lj(defined_types)
    return ModelConstants(reg, qtable, ljtab)
