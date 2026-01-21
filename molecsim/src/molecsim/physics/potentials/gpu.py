from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from molecsim.data.contributors import Contribution
from molecsim.data.data import SystemData, FLOAT_DTYPE
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties


# ======================================================================================
# Robust CUDA availability probe
# - Importing numba.cuda can succeed even if NVVM isn't present.
# - We only enable the GPU path if we can load NVVM and compile/launch a tiny kernel.
# ======================================================================================


def _probe_numba_cuda():
    try:
        from numba import cuda  # type: ignore
    except Exception as e:
        return None, f"numba.cuda import failed: {e!r}"

    try:
        if not cuda.is_available():
            return None, "CUDA driver not available (cuda.is_available() is False)"

        # Force NVVM load now (your traceback fails here if libnvvm.so is missing)
        from numba.cuda.cudadrv import nvvm  # type: ignore

        _ = nvvm.NVVM()

        # Smoke-compile + launch to ensure toolchain works
        @cuda.jit
        def _smoke(a):
            i = cuda.grid(1)
            if i < a.size:
                a[i] += 1

        d = cuda.device_array(1, dtype=np.int32)
        _smoke[1, 1](d)
        cuda.synchronize()

        return cuda, None
    except Exception as e:
        return None, f"CUDA present but unusable (NVVM/toolchain issue): {e!r}"


cuda, CUDA_ERROR = _probe_numba_cuda()

# Fixed threads-per-block for shared-memory reductions
TPB = 256


# ======================================================================================
# Utilities: host array handling + optional device buffer reuse
# ======================================================================================


def _as_f64_contig(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return np.ascontiguousarray(a)


def _as_i32_contig(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.int32)
    return np.ascontiguousarray(a)


def _pos_to_flat_f64(pos: np.ndarray) -> np.ndarray:
    """
    Accepts either (N,3) or flat (3N,) and returns flat (3N,) float64 contiguous.
    """
    a = _as_f64_contig(pos)
    if a.ndim == 2:
        if a.shape[1] != 3:
            raise ValueError(f"positions must be (N,3) or flat (3N,), got {a.shape}")
        a = a.reshape(-1)
    elif a.ndim == 1:
        if a.size % 3 != 0:
            raise ValueError(
                f"flat positions must have length multiple of 3, got {a.size}"
            )
    else:
        raise ValueError(f"positions must be (N,3) or flat (3N,), got ndim={a.ndim}")
    return np.ascontiguousarray(a)


class _GpuWorkspace:
    """
    Reuse device allocations to avoid cuda.to_device() allocating every call.
    Copies are still performed, but allocations are cached by size.
    """

    def __init__(self, cuda_mod):
        self.cuda = cuda_mod

        self.d_data_pos = None
        self.d_data_q = None
        self.d_data_types = None

        self.d_mol_pos = None
        self.d_mol_q = None
        self.d_mol_types = None

        self.d_lj_table = None
        self._lj_table_shape = None
        self._lj_table_fingerprint = None  # simple "changed?" marker

        self.d_partials = None
        self._partials_len = 0

        self.d_one = None  # single-element float64 array

    def _ensure_1d(self, d_arr, length: int, dtype) -> "cuda.devicearray.DeviceNDArray":
        if d_arr is None or d_arr.size != length or d_arr.dtype != np.dtype(dtype):
            d_arr = self.cuda.device_array(length, dtype=dtype)
        return d_arr

    def ensure_coulomb(self, M: int, N: int, blocks: int):
        self.d_data_pos = self._ensure_1d(self.d_data_pos, 3 * M, np.float64)
        self.d_data_q = self._ensure_1d(self.d_data_q, M, np.float64)
        self.d_mol_pos = self._ensure_1d(self.d_mol_pos, 3 * N, np.float64)
        self.d_mol_q = self._ensure_1d(self.d_mol_q, N, np.float64)

        if self.d_partials is None or self._partials_len != blocks:
            self.d_partials = self.cuda.device_array(blocks, dtype=np.float64)
            self._partials_len = blocks

        if self.d_one is None:
            self.d_one = self.cuda.device_array(1, dtype=np.float64)

    def ensure_lj(self, M: int, N: int, blocks: int, table: np.ndarray):
        self.d_data_pos = self._ensure_1d(self.d_data_pos, 3 * M, np.float64)
        self.d_mol_pos = self._ensure_1d(self.d_mol_pos, 3 * N, np.float64)

        self.d_data_types = self._ensure_1d(self.d_data_types, M, np.int32)
        self.d_mol_types = self._ensure_1d(self.d_mol_types, N, np.int32)

        # LJ table caching: copy only if changed
        tbl = _as_f64_contig(table)
        fingerprint = (
            tbl.shape,
            tbl.dtype.str,
            float(tbl.sum()),
            float(tbl.ravel()[0]) if tbl.size else 0.0,
        )
        if self.d_lj_table is None or self._lj_table_fingerprint != fingerprint:
            self.d_lj_table = self.cuda.to_device(tbl)
            self._lj_table_shape = tbl.shape
            self._lj_table_fingerprint = fingerprint

        if self.d_partials is None or self._partials_len != blocks:
            self.d_partials = self.cuda.device_array(blocks, dtype=np.float64)
            self._partials_len = blocks

        if self.d_one is None:
            self.d_one = self.cuda.device_array(1, dtype=np.float64)


_WORKSPACE: Optional[_GpuWorkspace] = None


def _ws() -> _GpuWorkspace:
    global _WORKSPACE
    if cuda is None:
        raise RuntimeError(f"CUDA unavailable: {CUDA_ERROR}")
    if _WORKSPACE is None:
        _WORKSPACE = _GpuWorkspace(cuda)
    return _WORKSPACE


# ======================================================================================
# GPU kernels / device helpers
# ======================================================================================

if cuda is not None:
    # A simple reduction helper; returns a Python scalar on the host
    sum_reduce = cuda.reduce(lambda a, b: a + b)

    @cuda.jit(device=True, inline=True)
    def _erfc_approx(x: float) -> float:
        # Abramowitz-Stegun erf approximation, erfc = 1 - erf
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1.0
        if x < 0.0:
            sign = -1.0
            x = -x

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * math.exp(
            -x * x
        )
        erf = sign * y
        return 1.0 - erf

    @cuda.jit(device=True, inline=True)
    def _nint(x: float) -> float:
        # nearest integer as float
        if x >= 0.0:
            return math.floor(x + 0.5)
        else:
            return -math.floor(-x + 0.5)

    @cuda.jit(device=True, inline=True)
    def _mic(dx: float, box: float, inv_box: float) -> float:
        # dx -= box * round(dx/box)
        return dx - box * _nint(dx * inv_box)

    # ----------------------------
    # Coulomb kernels (flat positions)
    # ----------------------------

    @cuda.jit(fastmath=True)
    def _coulomb_extra_flat_kernel(
        data_pos,  # (3*M,) float64
        data_q,  # (M,)  float64
        mol_pos,  # (3*N,) float64
        mol_q,  # (N,)  float64
        box0,
        box1,
        box2,
        inv_box0,
        inv_box1,
        inv_box2,
        periodic,  # int32 0/1
        coulomb_coeff,  # float64
        cutoff,  # float64
        out_partial,  # (blocks,) float64
    ):
        i = cuda.grid(1)
        M = data_q.size
        N = mol_q.size

        tid = cuda.threadIdx.x
        smem = cuda.shared.array(TPB, dtype=np.float64)

        acc = 0.0
        if i < M:
            xi = data_pos[3 * i + 0]
            yi = data_pos[3 * i + 1]
            zi = data_pos[3 * i + 2]
            qi = data_q[i]

            for j in range(N):
                dx = xi - mol_pos[3 * j + 0]
                dy = yi - mol_pos[3 * j + 1]
                dz = zi - mol_pos[3 * j + 2]

                if periodic != 0:
                    dx = _mic(dx, box0, inv_box0)
                    dy = _mic(dy, box1, inv_box1)
                    dz = _mic(dz, box2, inv_box2)

                r2 = dx * dx + dy * dy + dz * dz
                if r2 > 0.0:
                    r = math.sqrt(r2)
                    # keep your CPU behavior: erfc(r - cutoff) / r
                    acc += (qi * mol_q[j]) * (_erfc_approx(r - cutoff) / r)

            acc *= coulomb_coeff

        smem[tid] = acc
        cuda.syncthreads()

        stride = cuda.blockDim.x // 2
        while stride > 0:
            if tid < stride:
                smem[tid] += smem[tid + stride]
            cuda.syncthreads()
            stride //= 2

        if tid == 0:
            out_partial[cuda.blockIdx.x] = smem[0]

    @cuda.jit(fastmath=True)
    def _coulomb_intra_flat_kernel(
        mol_pos,
        mol_q,
        box0,
        box1,
        box2,
        inv_box0,
        inv_box1,
        inv_box2,
        periodic,
        coulomb_coeff,
        cutoff,
        out_one,  # (1,)
    ):
        # Simple: one thread computes intra (molecules are small in MC workflows)
        if cuda.blockIdx.x != 0 or cuda.threadIdx.x != 0:
            return

        N = mol_q.size
        acc = 0.0
        for i in range(N):
            xi = mol_pos[3 * i + 0]
            yi = mol_pos[3 * i + 1]
            zi = mol_pos[3 * i + 2]
            qi = mol_q[i]
            for j in range(i + 1, N):
                dx = xi - mol_pos[3 * j + 0]
                dy = yi - mol_pos[3 * j + 1]
                dz = zi - mol_pos[3 * j + 2]

                if periodic != 0:
                    dx = _mic(dx, box0, inv_box0)
                    dy = _mic(dy, box1, inv_box1)
                    dz = _mic(dz, box2, inv_box2)

                r2 = dx * dx + dy * dy + dz * dz
                if r2 > 0.0:
                    r = math.sqrt(r2)
                    acc += (qi * mol_q[j]) * (_erfc_approx(r - cutoff) / r)

        out_one[0] = coulomb_coeff * acc

    # ----------------------------
    # Lennard-Jones kernels (flat positions)
    # table: (T,T,2) float64, where table[ti,tj,0]=Aij and [ti,tj,1]=Bij
    # ----------------------------

    @cuda.jit(fastmath=True)
    def _lj_extra_flat_kernel(
        data_pos,
        data_types,  # data_pos: (3*M,), data_types: (M,)
        mol_pos,
        mol_types,  # mol_pos:  (3*N,), mol_types:  (N,)
        table,  # (T,T,2)
        box0,
        box1,
        box2,
        inv_box0,
        inv_box1,
        inv_box2,
        periodic,
        out_partial,
    ):
        i = cuda.grid(1)
        M = data_types.size
        N = mol_types.size

        tid = cuda.threadIdx.x
        smem = cuda.shared.array(TPB, dtype=np.float64)

        acc = 0.0
        if i < M:
            xi = data_pos[3 * i + 0]
            yi = data_pos[3 * i + 1]
            zi = data_pos[3 * i + 2]
            ti = data_types[i]

            for j in range(N):
                dx = xi - mol_pos[3 * j + 0]
                dy = yi - mol_pos[3 * j + 1]
                dz = zi - mol_pos[3 * j + 2]

                if periodic != 0:
                    dx = _mic(dx, box0, inv_box0)
                    dy = _mic(dy, box1, inv_box1)
                    dz = _mic(dz, box2, inv_box2)

                r2 = dx * dx + dy * dy + dz * dz
                if r2 > 0.0:
                    tj = mol_types[j]
                    Aij = table[ti, tj, 0]
                    Bij = table[ti, tj, 1]

                    r6 = r2 * r2 * r2
                    r12 = r6 * r6
                    acc += (Aij / r12) - (Bij / r6)

        smem[tid] = acc
        cuda.syncthreads()

        stride = cuda.blockDim.x // 2
        while stride > 0:
            if tid < stride:
                smem[tid] += smem[tid + stride]
            cuda.syncthreads()
            stride //= 2

        if tid == 0:
            out_partial[cuda.blockIdx.x] = smem[0]

    @cuda.jit(fastmath=True)
    def _lj_intra_flat_kernel(
        mol_pos,
        mol_types,
        table,
        box0,
        box1,
        box2,
        inv_box0,
        inv_box1,
        inv_box2,
        periodic,
        out_one,
    ):
        if cuda.blockIdx.x != 0 or cuda.threadIdx.x != 0:
            return

        N = mol_types.size
        acc = 0.0
        for i in range(N):
            xi = mol_pos[3 * i + 0]
            yi = mol_pos[3 * i + 1]
            zi = mol_pos[3 * i + 2]
            ti = mol_types[i]
            for j in range(i + 1, N):
                dx = xi - mol_pos[3 * j + 0]
                dy = yi - mol_pos[3 * j + 1]
                dz = zi - mol_pos[3 * j + 2]

                if periodic != 0:
                    dx = _mic(dx, box0, inv_box0)
                    dy = _mic(dy, box1, inv_box1)
                    dz = _mic(dz, box2, inv_box2)

                r2 = dx * dx + dy * dy + dz * dz
                if r2 > 0.0:
                    tj = mol_types[j]
                    Aij = table[ti, tj, 0]
                    Bij = table[ti, tj, 1]
                    r6 = r2 * r2 * r2
                    r12 = r6 * r6
                    acc += (Aij / r12) - (Bij / r6)

        out_one[0] = acc


# ======================================================================================
# Public API (same signatures as your CPU potentials)
# ======================================================================================


def coulomb_gpu(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    """
    CUDA Coulomb implementation.
    Notes:
    - This computes a rectangular "data x molecule" interaction plus intra(molecule).
    - For correctness you usually want `data` to be the system *excluding* the molecule,
      otherwise you will include molecule-vs-itself terms in the "extra" part.
    - If CUDA/NVVM isn't available, this raises with a helpful error message (so you can skip it).
    """
    if cuda is None:
        raise RuntimeError(
            "CUDA backend is unavailable.\n"
            f"Reason: {CUDA_ERROR}\n\n"
            "Typical fixes:\n"
            "- Install a CUDA toolkit that provides libnvvm.so\n"
            "- Ensure libnvvm.so is on your loader path (e.g. /opt/cuda/nvvm/lib64 on Arch)\n"
            "- Or run this force-set without the GPU backend."
        )

    if len(molecule_data) > len(data):
        raise ValueError(
            f"Data must be larger or equal to molecule_data: data {len(data)}, molecule_data: {len(molecule_data)}"
        )

    # Host arrays -> flat float64
    data_pos = _pos_to_flat_f64(data.positions)
    mol_pos = _pos_to_flat_f64(molecule_data.positions)
    data_q = _as_f64_contig(data.charges).reshape(-1)
    mol_q = _as_f64_contig(molecule_data.charges).reshape(-1)

    M = data_q.size
    N = mol_q.size
    if data_pos.size != 3 * M:
        raise ValueError("data.positions and data.charges size mismatch")
    if mol_pos.size != 3 * N:
        raise ValueError(
            "molecule_data.positions and molecule_data.charges size mismatch"
        )

    threads = TPB
    blocks = (M + threads - 1) // threads

    ws = _ws()
    ws.ensure_coulomb(M=M, N=N, blocks=blocks)

    # Copy into existing device buffers
    ws.d_data_pos.copy_to_device(data_pos)
    ws.d_data_q.copy_to_device(data_q)
    ws.d_mol_pos.copy_to_device(mol_pos)
    ws.d_mol_q.copy_to_device(mol_q)

    box = _as_f64_contig(props.box).reshape(-1)
    if box.size != 3:
        raise ValueError("props.box must be length 3")

    box0, box1, box2 = float(box[0]), float(box[1]), float(box[2])
    inv0, inv1, inv2 = 1.0 / box0, 1.0 / box1, 1.0 / box2
    periodic = np.int32(1 if props.periodic else 0)

    _coulomb_extra_flat_kernel[blocks, threads](
        ws.d_data_pos,
        ws.d_data_q,
        ws.d_mol_pos,
        ws.d_mol_q,
        box0,
        box1,
        box2,
        inv0,
        inv1,
        inv2,
        periodic,
        float(model.coulomb_coeff),
        float(model.cutoff),
        ws.d_partials,
    )

    # Intra (cheap, molecule is small)
    _coulomb_intra_flat_kernel[1, 1](
        ws.d_mol_pos,
        ws.d_mol_q,
        box0,
        box1,
        box2,
        inv0,
        inv1,
        inv2,
        periodic,
        float(model.coulomb_coeff),
        float(model.cutoff),
        ws.d_one,
    )

    # Reduce partials on GPU
    extra = float(sum_reduce(ws.d_partials))
    intra = float(ws.d_one.copy_to_host()[0])

    return float(np.array(extra + intra, dtype=FLOAT_DTYPE))


def lennard_jones_gpu(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    """
    CUDA Lennard-Jones implementation (same rectangular+intra structure as coulomb_gpu).
    Same correctness note: you usually want `data` to exclude the molecule for the "extra" part.
    """
    if cuda is None:
        raise RuntimeError(
            "CUDA backend is unavailable.\n"
            f"Reason: {CUDA_ERROR}\n\n"
            "Typical fixes:\n"
            "- Install a CUDA toolkit that provides libnvvm.so\n"
            "- Ensure libnvvm.so is on your loader path\n"
            "- Or run this force-set without the GPU backend."
        )

    data_pos = _pos_to_flat_f64(data.positions)
    mol_pos = _pos_to_flat_f64(molecule_data.positions)
    data_types = _as_i32_contig(data.atoms).reshape(-1)
    mol_types = _as_i32_contig(molecule_data.atoms).reshape(-1)

    M = data_types.size
    N = mol_types.size
    if data_pos.size != 3 * M:
        raise ValueError("data.positions and data.atoms size mismatch")
    if mol_pos.size != 3 * N:
        raise ValueError(
            "molecule_data.positions and molecule_data.atoms size mismatch"
        )

    table = np.asarray(model.ljtable.table, dtype=np.float64)
    if table.ndim != 3 or table.shape[2] != 2:
        raise ValueError(f"model.ljtable.table must be (T,T,2), got {table.shape}")

    threads = TPB
    blocks = (M + threads - 1) // threads

    ws = _ws()
    ws.ensure_lj(M=M, N=N, blocks=blocks, table=table)

    ws.d_data_pos.copy_to_device(data_pos)
    ws.d_mol_pos.copy_to_device(mol_pos)
    ws.d_data_types.copy_to_device(data_types)
    ws.d_mol_types.copy_to_device(mol_types)

    box = _as_f64_contig(props.box).reshape(-1)
    if box.size != 3:
        raise ValueError("props.box must be length 3")
    box0, box1, box2 = float(box[0]), float(box[1]), float(box[2])
    inv0, inv1, inv2 = 1.0 / box0, 1.0 / box1, 1.0 / box2
    periodic = np.int32(1 if props.periodic else 0)

    _lj_extra_flat_kernel[blocks, threads](
        ws.d_data_pos,
        ws.d_data_types,
        ws.d_mol_pos,
        ws.d_mol_types,
        ws.d_lj_table,
        box0,
        box1,
        box2,
        inv0,
        inv1,
        inv2,
        periodic,
        ws.d_partials,
    )

    _lj_intra_flat_kernel[1, 1](
        ws.d_mol_pos,
        ws.d_mol_types,
        ws.d_lj_table,
        box0,
        box1,
        box2,
        inv0,
        inv1,
        inv2,
        periodic,
        ws.d_one,
    )

    extra = float(sum_reduce(ws.d_partials))
    intra = float(ws.d_one.copy_to_host()[0])

    return float(np.array(extra + intra, dtype=FLOAT_DTYPE))
