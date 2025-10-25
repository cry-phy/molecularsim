from molecsim.system.data import FLOAT_DTYPE, FloatPrecision, StructureNode, SystemData
from molecsim.system.molecule import Molecule

import os
import re
import numpy as np
import numpy.typing as npt
from typing import Protocol

import argparse
from pathlib import Path


class Parser(Protocol):
    """Molecule Parser signature."""

    def __call__(self, path: str) -> Molecule:
        """Make Placeholder callable."""
        ...


_FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _locate_comment_block(lines: list[str], geom_idx: int) -> list[str]:
    """Return the title/comment block lines preceding the charge/multiplicity line at `geom_idx`.

    Gaussian layout (typical):
        [link0]*, route
        <blank>
        TITLE / COMMENT (>=1 lines)
        <blank>
        charge multiplicity
        geometry...
    We walk backward from geom_idx to find the blank that ends the title block.
    """
    # Find the last non-blank line before the charge/multiplicity line
    i = geom_idx - 1
    while i >= 0 and lines[i].strip() == "":
        i -= 1
    # Now scan backward to the previous blank line
    j = i
    while j >= 0 and lines[j].strip() != "":
        j -= 1
    # Title/comment block is lines[j+1 : i+1]
    start = j + 1
    end = i + 1
    if start < 0:
        start = 0
    return lines[start:end]


def _parse_charges_from_comment(
    comment_lines: list[str], natoms: int
) -> np.ndarray | None:
    """Look for '@charges:' (or '@charges =') marker in the title/comment block and parse floats.

    Accept either:
        @charges: q1 q2 q3 ...
    or
        @charges = [ q1, q2, ... ]

    Returns np.ndarray(FLOAT_DTYPE) of length natoms, or None if not present / mismatch.
    """
    if not comment_lines:
        return None

    blob = "\n".join(comment_lines)

    # Explicit marker (case-insensitive)
    m_marker = re.search(r"@charges\s*[:=]", blob, flags=re.IGNORECASE)
    if not m_marker:
        return None

    # Prefer bracketed block after the marker if present (allows multiline)
    m_brack = re.search(
        r"@charges\s*[:=][^\[\]\n]*\[(.*?)\]",
        blob,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_brack:
        body = m_brack.group(1)
        nums = re.findall(_FLOAT_RE, body)
    else:
        # Take numbers from the marker line onward until the next blank line
        tail = blob[m_marker.start() :]
        tail = tail.split("\n\n", 1)[0]
        nums = re.findall(_FLOAT_RE, tail)

    if not nums:
        return None

    arr = np.asarray([float(x) for x in nums], dtype=FLOAT_DTYPE)
    if arr.size != natoms:
        # Length mismatch: ignore silently (non-fatal) to stay robust in batch usage
        return None
    return arr


# ---------- Geometry parsing ----------


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
    """Detect gaussian geometry file coordinate type (stub)."""
    return "unknown"


def _parse_cartesian_block(
    lines: list[str], start: int
) -> tuple[
    list[str],
    npt.NDArray[FloatPrecision],
    npt.NDArray[FloatPrecision],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Parse Gaussian Cartesian geometry starting at line index `start`."""
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
                f"Line {line!r} could not be parsed. Format: Symbol X Y Z [Charge] [Multiplicity] [Fragment]"
            )

        atom = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            raise RuntimeError(
                f"Line {line!r} could not be parsed. Format: Symbol X Y Z [Charge] [Multiplicity] [Fragment]"
            ) from None

        # Optional trailing fields (ignored in favor of comment-block override, but preserved)
        charges.append(0.0 if len(parts) <= 5 else float(parts[5]))
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
    list[StructureNode],
]:
    """Parse a Gaussian Z-matrix block starting at line index `start`."""

    def parts_at(i: int) -> list[str]:
        return lines[i].strip().split()

    # First three atoms handled individually
    p0 = parts_at(start)
    p1 = parts_at(start + 1)
    p2 = parts_at(start + 2)
    atoms: list[str] = [p0[0], p1[0], p2[0]]
    coords: list[npt.NDArray[FloatPrecision]] = []
    nodes: list[StructureNode] = [StructureNode(0, [])]

    # Atom 1 at origin
    coords.append(np.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE))

    # Atom 2: "A  ref1  r"
    ref1_idx = int(p1[1]) - 1
    r12 = float(p1[2])
    pos2 = coords[ref1_idx] + np.array([0.0, 0.0, r12], dtype=FLOAT_DTYPE)
    coords.append(pos2)
    nodes.append(StructureNode(1, [nodes[ref1_idx]]))

    # Atom 3: "A  ref1  r  ref2  angle"
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
    nodes.append(StructureNode(2, [nodes[ref1_idx]]))

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
        nodes.append(StructureNode(i - start, [nodes[r1]]))
        charges.append(0.0)
        multiplicity.append(0)
        fragment.append(0)
        i += 1

    coords_arr = np.asarray(coords, dtype=FLOAT_DTYPE)
    charges_arr = np.asarray(charges, dtype=FLOAT_DTYPE)
    multiplicity_arr = np.asarray(multiplicity, dtype=np.int32)
    fragment_arr = np.asarray(fragment, dtype=np.int32)

    return atoms, coords_arr, charges_arr, multiplicity_arr, fragment_arr, nodes


def parse_com(path: str) -> Molecule:
    """Parse Gaussian input files (.com/.gjf), supporting Cartesian and Z-matrix geometries.

    Also supports overriding per-atom charges from the TITLE/COMMENT block using:
        @charges: q1 q2 ...
    or
        @charges = [ q1, q2, ... ]
    """
    data: SystemData | None = None

    # Keep blank lines; we need them to locate the title/comment block reliably
    with open(path) as file:
        lines = [line.rstrip("\n") for line in file]

    # Locate the charge/multiplicity line (e.g., "0 1")
    geom_line_idx: int | None = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*[+-]?\d+\s+[+-]?\d+\s*$", line):
            geom_line_idx = i
            geom_start = i + 1
            break
    if geom_line_idx is None:
        raise ValueError(f"Could not find charge/multiplicity line in {path}")

    # Extract the comment block to look for @charges
    comment_lines = _locate_comment_block(lines, geom_line_idx)

    # Determine geometry type by peeking at the first geom line
    # (after the charge/multiplicity line)
    if geom_start >= len(lines):
        raise ValueError(f"No geometry found after charge/multiplicity in {path}")

    first_geom = lines[geom_start].strip().split()

    nodes: list[StructureNode] | None = None

    # If 2-3 floats after element: Cartesian
    if len(first_geom) >= 4 and all(
        re.match(rf"^{_FLOAT_RE}$", p) for p in first_geom[1:4]
    ):
        try:
            atoms, positions, charges, _, _ = _parse_cartesian_block(lines, geom_start)
            data = SystemData.as_owner(atoms, positions, charges)
        except Exception as e:
            raise RuntimeError("Could not parse Cartesian geometry.") from e

    # Single atom name: Z-matrix
    elif len(first_geom) == 1:
        try:
            atoms, positions, charges, _, _, nodes = _parse_zmatrix_block(
                lines, geom_start
            )
            data = SystemData.as_owner(atoms, positions, charges)
        except Exception as e:
            raise RuntimeError("Could not parse Z-matrix geometry.") from e
    else:
        raise RuntimeError(
            "Unrecognized geometry format (neither Cartesian nor Z-matrix)."
        )

    if data is None:
        raise RuntimeError("Could not load data.")

    # Comment-block charges override (order preserved)
    natoms = data.positions.shape[0]
    comment_charges = _parse_charges_from_comment(comment_lines, natoms)
    if comment_charges is not None:
        data.charges[:] = comment_charges

    return Molecule(data, nodes)


EXTENSION_MAPPING: dict[str, Parser] = {
    ".com": parse_com,
    ".gjf": parse_com,
    # ".log": parse_log,   # could be added later
    # ".fchk": parse_fchk, # could be added later
}


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for headless MC run."""
    parser = argparse.ArgumentParser(
        prog="molecsim-headless",
        description="Run a headless MC simulation and save final state.",
    )
    parser.add_argument(
        "-f",
        "--files",
        required=True,
        help="Comma-separated list of molecule files (e.g. data1.xyz,data2.xyz)",
    )
    parser.add_argument(
        "-n",
        "--numbers",
        required=True,
        help="Comma-separated list of molecule counts for each file (e.g. 3,5)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        required=True,
        type=float,
        help="System temperature in Kelvin",
    )
    parser.add_argument(
        "-b",
        "--box",
        required=True,
        help="Box size in Angstrom as 'Lx,Ly,Lz' (e.g. 10,10,10)",
    )

    # Minimal extra knobs for headless running:
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=10_000,
        help="Number of MC steps to run (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="RNG seed (default: 10)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("out/state_final.pickle"),
        help="Output pickle path for final state (default: out/state_final.pickle)",
    )

    args = parser.parse_args(argv)

    # Normalize and validate lists
    args.files = [Path(f.strip()) for f in args.files.split(",") if f.strip()]
    args.numbers = [int(n.strip()) for n in args.numbers.split(",") if n.strip()]
    args.box = [float(x.strip()) for x in args.box.split(",") if x.strip()]

    if len(args.box) != 3:
        parser.error("box must have exactly three components: Lx,Ly,Lz")

    if len(args.files) != len(args.numbers):
        parser.error(
            f"number of files ({len(args.files)}) must match number of counts ({len(args.numbers)})"
        )
    missing = [f for f in args.files if not f.is_file()]
    if missing:
        parser.error(f"file(s) not found: {', '.join(str(f) for f in missing)}")

    return args
