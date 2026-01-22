from __future__ import annotations

import numpy as np
import numpy.typing as npt

FLOAT_DTYPE = np.float64
INT_DTYPE = np.int32
COMPLEX_TYPE = np.complex128

type ComplexPrecision = COMPLEX_TYPE
type FloatPrecision = FLOAT_DTYPE
type StructureArray = npt.NDArray[FloatPrecision]
type StructureView = StructureArray

# Optional CUDA support (kept lazy / non-fatal)
try:
    from numba import cuda as _cuda  # type: ignore
except Exception:
    _cuda = None


class CudaUnavailableError(RuntimeError):
    pass


def _require_cuda(cuda=_cuda):
    if cuda is None:
        raise CudaUnavailableError(
            "CUDA not available (numba.cuda could not be imported)."
        )
    try:
        cuda.current_context()
    except Exception as e:
        raise CudaUnavailableError(f"CUDA present but unusable: {e!r}") from e
    return cuda


class SystemDataGpuMirror:
    """Owner-only GPU mirror.

    - Positions are stored as SoA (x,y,z) for coalesced loads.
    - Device dtype defaults to float32 for speed on Consumer grade cards.
    - Charges float32, types int32.
    """

    def __init__(
        self,
        owner: SystemData,
        *,
        cuda,
        pos_dtype=np.float32,
        q_dtype=np.float32,
        with_velocities: bool = False,
    ) -> None:
        self.owner = owner
        self.cuda = cuda

        n = len(owner)
        self.pos_dtype = np.dtype(pos_dtype)
        self.q_dtype = np.dtype(q_dtype)

        # SoA positions
        self.d_x = cuda.device_array(n, dtype=self.pos_dtype)
        self.d_y = cuda.device_array(n, dtype=self.pos_dtype)
        self.d_z = cuda.device_array(n, dtype=self.pos_dtype)

        # Optional velocities (SoA) if you ever need them on device
        self.with_velocities = bool(with_velocities)
        self.d_vx = self.d_vy = self.d_vz = None
        if self.with_velocities:
            self.d_vx = cuda.device_array(n, dtype=self.pos_dtype)
            self.d_vy = cuda.device_array(n, dtype=self.pos_dtype)
            self.d_vz = cuda.device_array(n, dtype=self.pos_dtype)

        self.d_q = cuda.device_array(n, dtype=self.q_dtype)
        self.d_t = cuda.device_array(n, dtype=np.int32)

        # Dirty flags
        self.host_dirty = True
        self.device_dirty = False

    def upload_all(self) -> None:
        """Full host->device sync."""
        pos = self.owner._positions
        q = self.owner._charges
        t = self.owner._types

        self.cuda.to_device(
            np.ascontiguousarray(pos[:, 0], dtype=self.pos_dtype), to=self.d_x
        )
        self.cuda.to_device(
            np.ascontiguousarray(pos[:, 1], dtype=self.pos_dtype), to=self.d_y
        )
        self.cuda.to_device(
            np.ascontiguousarray(pos[:, 2], dtype=self.pos_dtype), to=self.d_z
        )

        if self.with_velocities:
            vel = self.owner._velocities
            self.cuda.to_device(
                np.ascontiguousarray(vel[:, 0], dtype=self.pos_dtype), to=self.d_vx
            )
            self.cuda.to_device(
                np.ascontiguousarray(vel[:, 1], dtype=self.pos_dtype), to=self.d_vy
            )
            self.cuda.to_device(
                np.ascontiguousarray(vel[:, 2], dtype=self.pos_dtype), to=self.d_vz
            )

        self.cuda.to_device(np.ascontiguousarray(q, dtype=self.q_dtype), to=self.d_q)
        self.cuda.to_device(np.ascontiguousarray(t, dtype=np.int32), to=self.d_t)

        self.host_dirty = False
        self.device_dirty = False

    def upload_range(self, start: int, size: int) -> None:
        """Patch host->device for a slice [start:start+size]."""
        if size <= 0:
            return
        stop = start + size
        pos = self.owner._positions[start:stop]
        q = self.owner._charges[start:stop]
        t = self.owner._types[start:stop]

        sl = slice(start, stop)

        self.cuda.to_device(
            np.ascontiguousarray(pos[:, 0], dtype=self.pos_dtype), to=self.d_x[sl]
        )
        self.cuda.to_device(
            np.ascontiguousarray(pos[:, 1], dtype=self.pos_dtype), to=self.d_y[sl]
        )
        self.cuda.to_device(
            np.ascontiguousarray(pos[:, 2], dtype=self.pos_dtype), to=self.d_z[sl]
        )

        if self.with_velocities:
            vel = self.owner._velocities[start:stop]
            self.cuda.to_device(
                np.ascontiguousarray(vel[:, 0], dtype=self.pos_dtype), to=self.d_vx[sl]
            )
            self.cuda.to_device(
                np.ascontiguousarray(vel[:, 1], dtype=self.pos_dtype), to=self.d_vy[sl]
            )
            self.cuda.to_device(
                np.ascontiguousarray(vel[:, 2], dtype=self.pos_dtype), to=self.d_vz[sl]
            )

        self.cuda.to_device(
            np.ascontiguousarray(q, dtype=self.q_dtype), to=self.d_q[sl]
        )
        self.cuda.to_device(np.ascontiguousarray(t, dtype=np.int32), to=self.d_t[sl])

        # In CPU-authoritative mode, after patching, both sides match.
        self.host_dirty = False
        self.device_dirty = False

    def download_all(self) -> None:
        """Full device->host sync (avoid in hot loop)."""
        x = self.d_x.copy_to_host().astype(np.float64, copy=False)
        y = self.d_y.copy_to_host().astype(np.float64, copy=False)
        z = self.d_z.copy_to_host().astype(np.float64, copy=False)

        self.owner._positions[:, 0] = x
        self.owner._positions[:, 1] = y
        self.owner._positions[:, 2] = z

        if self.with_velocities:
            vx = self.d_vx.copy_to_host().astype(np.float64, copy=False)
            vy = self.d_vy.copy_to_host().astype(np.float64, copy=False)
            vz = self.d_vz.copy_to_host().astype(np.float64, copy=False)
            self.owner._velocities[:, 0] = vx
            self.owner._velocities[:, 1] = vy
            self.owner._velocities[:, 2] = vz

        self.owner._charges[:] = self.d_q.copy_to_host().astype(np.float64, copy=False)
        self.owner._types[:] = self.d_t.copy_to_host().astype(np.int32, copy=False)

        self.device_dirty = False
        self.host_dirty = False

    def mark_host_dirty(self) -> None:
        self.host_dirty = True
        self.device_dirty = False

    def mark_device_dirty(self) -> None:
        self.device_dirty = True
        self.host_dirty = False


class SystemData:
    """State-only container: positions, charges, integer type IDs."""

    dims = 3

    def __init__(self, n: int) -> None:
        """Generate empty Data arrays."""
        self._positions: StructureArray = np.zeros((n, 3), dtype=FLOAT_DTYPE)
        self._velocities: StructureArray = np.zeros((n, 3), dtype=FLOAT_DTYPE)
        self._charges: npt.NDArray[FloatPrecision] = np.zeros(n, dtype=FLOAT_DTYPE)
        self._types: npt.NDArray[INT_DTYPE] = np.zeros(n, dtype=INT_DTYPE)

        self._start_index: int = 0  # global start index in owner (if this is a view)
        self._first_empty: int = 0  # allocator cursor (owner only)

        self._owner: SystemData = (
            self  # owner points to itself; views point to the owner
        )
        self._gpu: SystemDataGpuMirror | None = None  # only meaningful on owner

    @classmethod
    def as_owner(
        cls,
        types: npt.NDArray[INT_DTYPE],
        positions: StructureArray,
        charges: StructureArray,
    ) -> SystemData:
        """Create an owning copy from arrays."""
        n = len(charges)
        if positions.shape != (n, 3) or types.shape != (n,):
            raise ValueError("as_owner: shape mismatch among types/positions/charges.")
        inst = cls(n)
        np.copyto(inst._types, types)
        np.copyto(inst._positions, positions)
        np.copyto(inst._charges, charges)
        return inst

    def as_borrow(self, start: int, size: int) -> SystemData:
        """Create a non-owning view [start:start+size] into this data."""
        stop = start + size
        if not (0 <= start <= stop <= len(self)):
            raise IndexError("as_borrow: slice out of range.")

        inst = SystemData(size)

        # Slices relative to *this* instance:
        inst._positions = self._positions[start:stop]
        inst._velocities = self._velocities[
            start:stop
        ]  # important: keep velocities view too
        inst._charges = self._charges[start:stop]
        inst._types = self._types[start:stop]

        # Global indexing in owner coordinates:
        inst._owner = self._owner
        inst._gpu = None
        inst._start_index = self._start_index + start
        inst._first_empty = 0
        return inst

    def copy_data(self) -> SystemData:
        """Deep-copy arrays; keep start/first_empty metadata."""
        cp = SystemData.as_owner(
            self._types.copy(), self._positions.copy(), self._charges.copy()
        )
        cp._velocities = self._velocities.copy()
        cp._start_index = self._start_index
        cp._first_empty = self._first_empty
        return cp

    def enable_gpu(
        self,
        *,
        cuda=None,
        pos_dtype=np.float32,
        q_dtype=np.float32,
        with_velocities: bool = False,
        upload: bool = True,
    ) -> None:
        owner = self._owner
        if owner is not self:
            owner.enable_gpu(
                cuda=cuda,
                pos_dtype=pos_dtype,
                q_dtype=q_dtype,
                with_velocities=with_velocities,
                upload=upload,
            )
            return

        if cuda is None:
            cuda = _require_cuda(_cuda)
        else:
            _require_cuda(cuda)

        if owner._gpu is None:
            owner._gpu = SystemDataGpuMirror(
                owner,
                cuda=cuda,
                pos_dtype=pos_dtype,
                q_dtype=q_dtype,
                with_velocities=with_velocities,
            )
            if upload:
                owner._gpu.upload_all()
            return

        if upload and owner._gpu.host_dirty:
            owner._gpu.upload_all()

    def has_gpu(self) -> bool:
        return self._owner._gpu is not None

    def device_view(self, *, cuda=None):
        """
        Return device buffers for this data (owner-backed), plus (start, size).

        Returns:
            (d_x, d_y, d_z, d_q, d_t, start, size, d_vx, d_vy, d_vz)
        Velocity arrays may be None if with_velocities=False.
        """
        owner = self._owner
        if owner._gpu is None:
            owner.enable_gpu(cuda=cuda, upload=True)
        else:
            # If user passed cuda explicitly, ensure it matches what we have
            if cuda is not None and owner._gpu.cuda is not cuda:
                # Not fatal, but confusing. Keep the original mirror.
                pass

        gpu = owner._gpu
        if gpu.host_dirty:
            gpu.upload_all()

        start = self._start_index
        size = len(self)

        return (
            gpu.d_x,
            gpu.d_y,
            gpu.d_z,
            gpu.d_q,
            gpu.d_t,
            start,
            size,
            gpu.d_vx,
            gpu.d_vy,
            gpu.d_vz,
        )

    def sync_device_to_host(self) -> None:
        """Force device->host sync on the owner mirror (rarely needed)."""
        owner = self._owner
        if owner._gpu is not None:
            owner._gpu.download_all()

    def _mark_host_dirty(self) -> None:
        owner = self._owner
        if owner._gpu is not None:
            owner._gpu.mark_host_dirty()

    def copy_from(self, src: SystemData) -> None:
        """Copy atoms, positions, charges from src. Assumes same registry/IDs."""
        if (
            self._types.shape != src._types.shape
            or self._positions.shape != src._positions.shape
            or self._velocities.shape != src._velocities.shape
            or self._charges.shape != src._charges.shape
        ):
            raise ValueError("copy_from: shape mismatch between destination and source")

        # Fast no-op if all arrays already share memory
        if (
            np.shares_memory(self._types, src._types)
            and np.shares_memory(self._positions, src._positions)
            and np.shares_memory(self._velocities, src._velocities)
            and np.shares_memory(self._charges, src._charges)
        ):
            return

        np.copyto(self._types, src._types, casting="no")
        np.copyto(self._positions, src._positions, casting="no")
        np.copyto(self._velocities, src._velocities, casting="no")
        np.copyto(self._charges, src._charges, casting="no")

        owner = self._owner
        if owner._gpu is not None:
            owner._gpu.upload_range(self._start_index, len(self))

    def allocate_molecule(self, size: int) -> SystemData:
        """Allocate a contiguous view."""
        start = self._first_empty
        stop = start + size
        if stop > len(self):
            avail = len(self) - start
            raise ValueError(
                f"allocate_molecule: insufficient capacity (need {size}, have {avail})"
            )
        view = self.as_borrow(start, size)
        self._first_empty = stop
        return view

    @property
    def positions(self) -> StructureArray:
        """The positions property."""
        owner = self._owner
        if owner._gpu is not None and owner._gpu.device_dirty:
            owner._gpu.download_all()
        return self._positions

    @positions.setter
    def positions(self, positions: StructureArray) -> None:
        if positions.shape != self._positions.shape:
            raise ValueError("positions: shape mismatch")
        np.copyto(self._positions, positions)
        self._mark_host_dirty()

    @property
    def velocities(self) -> StructureArray:
        owner = self._owner
        if owner._gpu is not None and owner._gpu.device_dirty:
            owner._gpu.download_all()
        return self._velocities

    @velocities.setter
    def velocities(self, velocities: StructureArray) -> None:
        if velocities.shape != self._velocities.shape:
            raise ValueError("velocities: shape mismatch")
        np.copyto(self._velocities, velocities)
        self._mark_host_dirty()

    @property
    def charges(self) -> StructureArray:
        owner = self._owner
        if owner._gpu is not None and owner._gpu.device_dirty:
            owner._gpu.download_all()
        return self._charges

    @charges.setter
    def charges(self, value: npt.NDArray[FloatPrecision]) -> None:
        if value.shape != self._charges.shape:
            raise ValueError("charges: shape mismatch")
        np.copyto(self._charges, value)
        self._mark_host_dirty()

    @property
    def atoms(self) -> npt.NDArray[INT_DTYPE]:
        owner = self._owner
        if owner._gpu is not None and owner._gpu.device_dirty:
            owner._gpu.download_all()
        return self._types

    @atoms.setter
    def atoms(self, value: npt.NDArray[INT_DTYPE]) -> None:
        if value.shape != self._types.shape:
            raise ValueError("types: shape mismatch")
        np.copyto(self._types, value)
        self._mark_host_dirty()

    @property
    def first_empty(self) -> int:
        """The first empty atom index for placement."""
        return self._first_empty

    def __len__(self) -> int:
        """Length of Data."""
        return int(self._types.shape[0])

    def data_interval(self) -> tuple[int, int]:
        """Start/end indices relative to parent owner (if this is a view)."""
        return self._start_index, self._start_index + len(self)
