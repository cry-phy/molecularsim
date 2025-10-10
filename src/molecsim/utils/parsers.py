from molecsim.system.data import FLOAT_DTYPE, FloatPrecision, SystemData
from molecsim.system.molecule import Molecule
import os
import re
import numpy as np
import numpy.typing as npt
from typing import Protocol

from molecsim.utils.geometry import rot_matrix


class Parser(Protocol):
    """Molecule Parser signature."""

    def __call__(self, path: str) -> Molecule:
        """Make Placeholder callable."""
        ...


def parse_z_matrix(path: str) -> Molecule:
    """Parse a Gaussian input file (Z-Matrix or Cartesian) into structure and Molecule objects."""
    if not os.path.exists(path):
        raise ValueError(f"Provided file ({path}) does not exist.")

    # Detect file extension and determine correct parsing function
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    parsing_function = EXTENSION_MAPPING.get(ext)
    if parsing_function is None:
        supported = ", ".join(EXTENSION_MAPPING.keys())
        raise ValueError(f"Unsupported file extension '{ext}'. Supported: {supported}")

    return parsing_function(path)


def detect_geometry_type(line: str) -> str:
    """Detec gaussian geometry file coordiante type."""
    return "unknown"


def _parse_cartesian_block(
    lines: list[str], start: int
) -> tuple[
    list[str],
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
]:
    """Parse Gaussian Cartesian geometry starting at line index."""
    atoms: list[str] = []
    coords: list[list[float]] = []
    charges: list[float] = []
    multiplicity: list[int] = []
    fragment: list[int] = []

    i = start
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        if not line:
            break

        parts = line.split()
        if len(parts) < 4:
            raise RuntimeError(
                f"Line {line} could not be parsed. Format: Symbol X Y Z [Charge] [Multiplicity] [Fragment]"
            )

        atom = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            raise RuntimeError(
                f"Line {line} could not be parsed. Format: Symbol X Y Z [Charge] [Multiplicity] [Fragment]"
            ) from None
        charges.append(0 if len(parts) <= 5 else float(parts[5]))
        multiplicity.append(1 if len(parts) <= 6 else int(parts[6]))
        fragment.append(1 if len(parts) <= 7 else int(parts[7]))

        atoms.append(atom)
        coords.append([x, y, z])
        i += 1

    if not atoms:
        raise ValueError(f"No Cartesian atoms parsed starting at line {start + 1}.")

    coords_arr = np.asarray(coords, dtype=FLOAT_DTYPE)
    charges_arr = np.asarray(charges, dtype=FLOAT_DTYPE)
    multiplicity_arr = np.asarray(multiplicity, dtype=np.int32)
    fragment_arr = np.asarray(fragment, dtype=np.int32)

    return atoms, coords_arr, charges_arr, multiplicity_arr, fragment_arr


def _normalize(v: npt.NDArray[FloatPrecision]) -> npt.NDArray[np.float64]:
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("Zero-length vector encountered while building local frame.")
    return v / n


def _place_from_refs(
    ref1: npt.NDArray[FloatPrecision],
    ref2: npt.NDArray[FloatPrecision],
    ref3: npt.NDArray[FloatPrecision],
    bond_length: float,
    bond_angle_deg: float,
    dihedral_deg: float,
) -> npt.NDArray[FloatPrecision]:
    """Place a new atom using three already-placed reference atoms.

    - bond_length = |new - ref1|
    - bond_angle  = angle(new, ref1, ref2) in degrees
    - dihedral    = dihedral(new, ref1, ref2, ref3) in degrees
    """
    z_axis = _normalize(ref1 - ref2)
    x_axis = _normalize(np.cross(ref3 - ref2, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack((x_axis, y_axis, z_axis), axis=1)

    theta = np.deg2rad(bond_angle_deg)
    phi = np.deg2rad(dihedral_deg)
    local = np.array(
        [
            bond_length * np.sin(theta) * np.cos(phi),
            bond_length * np.sin(theta) * np.sin(phi),
            -bond_length * np.cos(theta),
        ],
        dtype=FLOAT_DTYPE,
    )
    return ref1 + R @ local


def _parse_zmatrix_block(
    lines: list[str], start: int
) -> tuple[
    list[str],
    npt.NDArray[FloatPrecision],
    npt.NDArray[FloatPrecision],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Parse a Gaussian Z-matrix block starting at line index `start`."""

    def parts_at(i: int) -> list[str]:
        return lines[i].strip().split()

    # first three atoms handled individually
    # Atom 1 at origin
    # Atom 2: "A  ref1  r"
    # Atom 3: "A  ref1  r  ref2  angle"
    p0 = parts_at(start)
    p1 = parts_at(start + 1)
    p2 = parts_at(start + 2)
    atoms: list[str] = [p0[0], p1[0], p2[0]]
    coords: list[npt.NDArray[FloatPrecision]] = []

    coords.append(np.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE))

    ref1_idx = int(p1[1]) - 1
    r12 = float(p1[2])
    pos2 = coords[ref1_idx] + np.array([0.0, 0.0, r12], dtype=FLOAT_DTYPE)
    coords.append(pos2)

    ref1_idx = int(p2[1]) - 1
    r = float(p2[2])
    ref2_idx = int(p2[3]) - 1
    angle_deg = float(p2[4])
    aux_ref3 = coords[ref2_idx] + np.array([1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    pos3 = _place_from_refs(
        ref1=coords[ref1_idx],
        ref2=coords[ref2_idx],
        ref3=aux_ref3,
        bond_length=r,
        bond_angle_deg=angle_deg,
        dihedral_deg=0.0,
    )
    coords.append(pos3)
    charges: list[float] = [0.0, 0.0, 0.0]
    multiplicity: list[int] = [0, 0, 0]
    fragment: list[int] = [0, 0, 0]
    i = start + 3
    n = len(lines)
    while i < n:
        raw = lines[i].strip()
        if not raw:
            break
        parts = raw.split()
        if len(parts) < 5:
            break

        symbol = parts[0]

        try:
            r1 = int(parts[1]) - 1
            r_bond = float(parts[2])
            r2 = int(parts[3]) - 1
            ang_deg = float(parts[4])
            if len(parts) >= 7:
                r3 = int(parts[5]) - 1
                dih_deg = float(parts[6])
            else:
                r3 = r2
                dih_deg = 0.0
        except Exception as exc:
            raise ValueError(f"Malformed Z-matrix line at {i + 1}: '{raw}'") from exc

        if r3 == r2:
            bond_vec = coords[r1] - coords[r2]
            helper = np.array([1.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
            if np.allclose(np.abs(np.dot(_normalize(bond_vec), helper)), 1.0):
                helper = np.array([0.0, 1.0, 0.0], dtype=FLOAT_DTYPE)
            ortho = coords[r2] + np.cross(bond_vec, helper)
            ref3 = ortho
        else:
            ref3 = coords[r3]

        new_pos = _place_from_refs(
            ref1=coords[r1],
            ref2=coords[r2],
            ref3=ref3,
            bond_length=r_bond,
            bond_angle_deg=ang_deg,
            dihedral_deg=dih_deg,
        )

        atoms.append(symbol)
        coords.append(new_pos)
        charges.append(0.0)
        multiplicity.append(0)
        fragment.append(0)
        i += 1

    coords_arr = np.asarray(coords, dtype=FLOAT_DTYPE)
    charges_arr = np.asarray(charges, dtype=FLOAT_DTYPE)
    multiplicity_arr = np.asarray(multiplicity, dtype=np.int32)
    fragment_arr = np.asarray(fragment, dtype=np.int32)

    return atoms, coords_arr, charges_arr, multiplicity_arr, fragment_arr


def parse_com(path: str) -> Molecule:
    """Parse gaussuian input files.

    Allows only for Tree structure Molecules until gaussian Connectivity mode is implemented.
    """
    data = None
    with open(path) as file:
        lines = [line.strip() for line in file if line.strip()]

    geom_start = None
    for i, line in enumerate(lines):
        if re.match(r"^[+-]?\d+\s+[+-]?\d+$", line):
            geom_start = i + 1
            break
    if geom_start is None:
        raise ValueError(f"Could not find charge/multiplicity line in {path}")

    parts = lines[geom_start].split()
    # If 2-3 floats after element: Cartesian
    if len(parts) >= 4 and all(re.match(r"^[+-]?\d*\.\d+$", p) for p in parts[1:4]):
        try:
            atoms, positions, charges, _, _ = _parse_cartesian_block(lines, geom_start)
            data = SystemData.as_owner(atoms, positions, charges)
        except Exception as e:
            raise RuntimeError("Could not parse file.") from e

    # Single atom name: Z-matrix
    if len(parts) == 1:
        try:
            atoms, positions, charges, _, _ = _parse_zmatrix_block(lines, geom_start)
            data = SystemData.as_owner(atoms, positions, charges)
        except Exception as e:
            raise RuntimeError("Could not parse file.") from e

    if not data:
        raise RuntimeError("Could not load data.")
    return Molecule(data)


EXTENSION_MAPPING: dict[str, Parser] = {
    ".com": parse_com,
    ".gjf": parse_com,
    #    ".log": parse_log,
    #    ".fchk": parse_fchk,
}
