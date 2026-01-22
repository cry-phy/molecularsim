from __future__ import annotations

from math import sqrt, exp, floor
from typing import Any

import numpy as np

from molecsim.data.contributors import Contribution
from molecsim.data.data import FLOAT_DTYPE, SystemData
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
from molecsim.physics.geometry import erfc

# ----------------------------
# CUDA availability
# ----------------------------

try:
    from numba import cuda  # type: ignore
except Exception:
    cuda = None


class CudaUnavailableError(RuntimeError):
    pass


def _require_cuda():
    if cuda is None:
        raise CudaUnavailableError("CUDA not available (numba.cuda import failed).")
    try:
        cuda.current_context()
    except Exception as e:
        raise CudaUnavailableError(f"CUDA present but unusable: {e!r}") from e
    return cuda


# ----------------------------
# Small device helpers
# ----------------------------

if cuda is not None:

    @cuda.jit(device=True, inline=True)
    def _rint(x: float) -> float:
        """Round-to-nearest integer as float (ties not important here)."""
        if x >= 0.0:
            return floor(x + 0.5)
        else:
            return -floor(-x + 0.5)

    @cuda.jit(device=True, inline=True)
    def _mic(dx: float, box: float) -> float:
        """Minimum image convention for one component."""
        # dx -= box * round(dx/box)
        return dx - box * _rint(dx / box)

    @cuda.jit(device=True, inline=True)
    def _erfc_approx(x: float) -> float:
        """
        Fast-ish erfc approximation (Abramowitz-Stegun style via erf approx).
        Good enough for real-space damping; validate if you care about precision.
        """
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
        y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * exp(-x * x)
        erf = sign * y
        return 1.0 - erf


# ----------------------------
# Module-level device buffer caches
# ----------------------------

# Keyed by (device_id, K)
_MOL_CACHE: dict[tuple[int, int], tuple[Any, Any, Any, Any, Any]] = {}

# Keyed by (device_id, blocks)
_PARTIALS_CACHE: dict[tuple[int, int], Any] = {}

# Keyed by (device_id, shape, dtype_str)
_LJ_TABLE_CACHE: dict[tuple[int, tuple[int, int, int], str], Any] = {}


def _device_id() -> int:
    cu = _require_cuda()
    dev = cu.get_current_device()

    # Preferred: real numeric id (if provided by this Numba build)
    did = getattr(dev, "id", None)
    if did is not None:
        return int(did)

    # Fallback 1: stable enough inside this process
    return int(id(dev))


def _get_mol_buffers(K: int):
    """Get (or allocate) device buffers for molecule SoA + q + types."""
    cu = _require_cuda()
    did = _device_id()
    key = (did, int(K))
    buf = _MOL_CACHE.get(key)
    if buf is not None:
        return buf

    d_mx = cu.device_array(K, dtype=np.float32)
    d_my = cu.device_array(K, dtype=np.float32)
    d_mz = cu.device_array(K, dtype=np.float32)
    d_mq = cu.device_array(K, dtype=np.float32)
    d_mt = cu.device_array(K, dtype=np.int32)

    _MOL_CACHE[key] = (d_mx, d_my, d_mz, d_mq, d_mt)
    return _MOL_CACHE[key]


def _get_partials(blocks: int):
    """Get (or allocate) per-block partial sums (float64)."""
    cu = _require_cuda()
    did = _device_id()
    key = (did, int(blocks))
    buf = _PARTIALS_CACHE.get(key)
    if buf is not None:
        return buf
    d_partials = cu.device_array((blocks,), dtype=np.float64)
    _PARTIALS_CACHE[key] = d_partials
    return d_partials


def _get_lj_table_device(model: ModelConstants):
    """Upload LJ table once per device."""
    cu = _require_cuda()
    did = _device_id()

    table = np.asarray(model.ljtable.table)  # expected shape (T,T,2)
    if table.ndim != 3 or table.shape[2] != 2:
        raise ValueError(
            f"Expected ljtable.table with shape (T,T,2), got {table.shape}"
        )

    key = (did, tuple(int(x) for x in table.shape), "f32")
    d_tab = _LJ_TABLE_CACHE.get(key)
    if d_tab is not None:
        return d_tab

    h_tab = np.ascontiguousarray(table, dtype=np.float32)
    d_tab = cu.to_device(h_tab)
    _LJ_TABLE_CACHE[key] = d_tab
    return d_tab


# ----------------------------
# Kernels: Extra interactions (mol vs rest of system)
# ----------------------------

if cuda is not None:

    @cuda.jit
    def _coulomb_extra_soa(
        x,
        y,
        z,
        q,  # system SoA (float32)
        n_system: int,
        mx,
        my,
        mz,
        mq,  # molecule SoA (float32)
        k_mol: int,
        m_start: int,
        m_stop: int,
        box0: float,
        box1: float,
        box2: float,
        periodic: int,
        alpha: float,
        cut2: float,
        coulomb_coeff: float,
        out_partials,  # float64[blocks]
    ):
        """
        One thread per system atom j.
        Accumulates sum_m qj*qm*erfc(alpha*r)/r over molecule atoms m.
        Skips j in [m_start,m_stop) to avoid counting intra/self as "extra".
        Writes per-block partial sum to out_partials[blockIdx.x].
        """
        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        bdim = cuda.blockDim.x
        j0 = bid * bdim + tid

        smem = cuda.shared.array(256, dtype=np.float64)  # threads must be <= 256

        acc = 0.0
        if j0 < n_system:
            # exclude molecule slice if valid
            if m_start >= 0 and (m_start <= j0 < m_stop):
                acc = 0.0
            else:
                xj = x[j0]
                yj = y[j0]
                zj = z[j0]
                qj = q[j0]

                # Loop over molecule atoms (small K)
                for m in range(k_mol):
                    dx = xj - mx[m]
                    dy = yj - my[m]
                    dz = zj - mz[m]

                    if periodic != 0:
                        dx = _mic(dx, box0)
                        dy = _mic(dy, box1)
                        dz = _mic(dz, box2)

                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 <= 1e-20:
                        continue
                    if cut2 > 0.0 and r2 > cut2:
                        continue

                    r = sqrt(r2)
                    inv_r = 1.0 / r
                    damp = _erfc_approx(alpha * r)
                    acc += (qj * mq[m]) * damp * inv_r

                acc *= coulomb_coeff

        smem[tid] = float(acc)
        cuda.syncthreads()

        # reduction in shared
        stride = bdim // 2
        while stride > 0:
            if tid < stride:
                smem[tid] += smem[tid + stride]
            cuda.syncthreads()
            stride //= 2

        if tid == 0:
            out_partials[bid] = smem[0]

    @cuda.jit
    def _lj_extra_soa(
        x,
        y,
        z,
        t,  # system SoA + types (float32 + int32)
        n_system: int,
        mx,
        my,
        mz,
        mt,  # molecule SoA + types
        k_mol: int,
        m_start: int,
        m_stop: int,
        box0: float,
        box1: float,
        box2: float,
        periodic: int,
        cut2: float,
        lj_table,  # float32 (T,T,2)
        out_partials,  # float64[blocks]
    ):
        """
        One thread per system atom j.
        Accumulates sum_m (A/r^12 - B/r^6) over molecule atoms m.
        """
        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        bdim = cuda.blockDim.x
        j0 = bid * bdim + tid

        smem = cuda.shared.array(256, dtype=np.float64)

        acc = 0.0
        if j0 < n_system:
            if m_start >= 0 and (m_start <= j0 < m_stop):
                acc = 0.0
            else:
                xj = x[j0]
                yj = y[j0]
                zj = z[j0]
                tj = t[j0]

                for m in range(k_mol):
                    dx = xj - mx[m]
                    dy = yj - my[m]
                    dz = zj - mz[m]

                    if periodic != 0:
                        dx = _mic(dx, box0)
                        dy = _mic(dy, box1)
                        dz = _mic(dz, box2)

                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 <= 1e-20:
                        continue
                    if cut2 > 0.0 and r2 > cut2:
                        continue

                    tm = mt[m]
                    A = lj_table[tj, tm, 0]
                    B = lj_table[tj, tm, 1]

                    r6 = r2 * r2 * r2
                    r12 = r6 * r6
                    acc += (A / r12) - (B / r6)

        smem[tid] = float(acc)
        cuda.syncthreads()

        stride = bdim // 2
        while stride > 0:
            if tid < stride:
                smem[tid] += smem[tid + stride]
            cuda.syncthreads()
            stride //= 2

        if tid == 0:
            out_partials[bid] = smem[0]

    @cuda.jit
    def _coulomb_lj_extra_soa(
        x,
        y,
        z,
        q,
        t,
        n_system: int,
        mx,
        my,
        mz,
        mq,
        mt,
        k_mol: int,
        m_start: int,
        m_stop: int,
        box0: float,
        box1: float,
        box2: float,
        periodic: int,
        alpha: float,
        cut2: float,
        coulomb_coeff: float,
        lj_table,
        out_partials,  # float64
    ):
        """Fused Coulomb + LJ in one pass over system atoms."""
        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        bdim = cuda.blockDim.x
        j0 = bid * bdim + tid

        smem = cuda.shared.array(256, dtype=np.float64)

        acc = 0.0
        if j0 < n_system:
            if m_start >= 0 and (m_start <= j0 < m_stop):
                acc = 0.0
            else:
                xj = x[j0]
                yj = y[j0]
                zj = z[j0]
                qj = q[j0]
                tj = t[j0]

                for m in range(k_mol):
                    dx = xj - mx[m]
                    dy = yj - my[m]
                    dz = zj - mz[m]

                    if periodic != 0:
                        dx = _mic(dx, box0)
                        dy = _mic(dy, box1)
                        dz = _mic(dz, box2)

                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 <= 1e-20:
                        continue
                    if cut2 > 0.0 and r2 > cut2:
                        continue

                    r = sqrt(r2)
                    inv_r = 1.0 / r
                    damp = _erfc_approx(alpha * r)
                    acc += (qj * mq[m]) * damp * inv_r * coulomb_coeff

                    tm = mt[m]
                    A = lj_table[tj, tm, 0]
                    B = lj_table[tj, tm, 1]
                    r6 = r2 * r2 * r2
                    r12 = r6 * r6
                    acc += (A / r12) - (B / r6)

        smem[tid] = float(acc)
        cuda.syncthreads()

        stride = bdim // 2
        while stride > 0:
            if tid < stride:
                smem[tid] += smem[tid + stride]
            cuda.syncthreads()
            stride //= 2

        if tid == 0:
            out_partials[bid] = smem[0]


# ----------------------------
# CPU intra helpers (K is small; keep it simple)
# ----------------------------


def _ewald_alpha(cutoff: float, tol: float) -> float:
    # Rough but effective: choose alpha so exp(-(alpha*cut)^2) ~ tol
    # (erfc has similar tail scale). Avoid tol<=0.
    tol = max(float(tol), 1e-30)
    return float(np.sqrt(-np.log(tol)) / float(cutoff))


def _mic_cpu(d: float, box: float) -> float:
    return d - box * np.rint(d / box)


def _coulomb_intra_cpu(
    mol_pos: np.ndarray,
    mol_q: np.ndarray,
    box: np.ndarray,
    periodic: bool,
    cutoff: float,
    alpha: float,
    coulomb_coeff: float,
) -> float:
    K = mol_pos.shape[0]
    cut2 = cutoff * cutoff
    acc = 0.0
    for i in range(K):
        xi, yi, zi = mol_pos[i]
        qi = mol_q[i]
        for j in range(i + 1, K):
            dx = xi - mol_pos[j, 0]
            dy = yi - mol_pos[j, 1]
            dz = zi - mol_pos[j, 2]
            if periodic:
                dx = _mic_cpu(dx, box[0])
                dy = _mic_cpu(dy, box[1])
                dz = _mic_cpu(dz, box[2])
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 1e-30 or r2 > cut2:
                continue
            r = float(np.sqrt(r2))
            acc += (qi * mol_q[j]) * float(erfc(alpha * r)) / r
    return coulomb_coeff * acc


def _lj_intra_cpu(
    mol_pos: np.ndarray,
    mol_t: np.ndarray,
    lj_table: np.ndarray,
    box: np.ndarray,
    periodic: bool,
    cutoff: float,
) -> float:
    K = mol_pos.shape[0]
    cut2 = cutoff * cutoff
    acc = 0.0
    for i in range(K):
        xi, yi, zi = mol_pos[i]
        ti = int(mol_t[i])
        for j in range(i + 1, K):
            dx = xi - mol_pos[j, 0]
            dy = yi - mol_pos[j, 1]
            dz = zi - mol_pos[j, 2]
            if periodic:
                dx = _mic_cpu(dx, box[0])
                dy = _mic_cpu(dy, box[1])
                dz = _mic_cpu(dz, box[2])
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 1e-30 or r2 > cut2:
                continue
            tj = int(mol_t[j])
            A = float(lj_table[ti, tj, 0])
            B = float(lj_table[ti, tj, 1])
            r6 = r2 * r2 * r2
            r12 = r6 * r6
            acc += (A / r12) - (B / r6)
    return acc


# ----------------------------
# Public potential functions
# ----------------------------


def coulomb_gpu(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    cu = _require_cuda()

    # Ensure resident system buffers exist (your new data.py mirror should provide this).
    if hasattr(data, "enable_gpu"):
        data.enable_gpu(cuda=cu, pos_dtype=np.float32, q_dtype=np.float32, upload=True)

    if not hasattr(data, "device_view"):
        raise RuntimeError(
            "SystemData has no device_view(); install the GPU mirror in data.py first."
        )

    d_x, d_y, d_z, d_q, d_t, sys_start, sys_size, *_ = data.device_view(cuda=cu)
    N = int(sys_size)
    if sys_start != 0:
        # We expect the system passed here to be the owner; if not, still handle it.
        # Kernels treat indices [0..N) relative to the passed arrays.
        pass

    # Molecule slice in owner coordinates (for self-exclusion)
    m_start, m_stop = molecule_data.data_interval()
    m_start = int(m_start)
    m_stop = int(m_stop)
    if not (0 <= m_start <= m_stop <= N):
        m_start = -1
        m_stop = -1

    # Upload molecule (K is small)
    mol_pos64 = np.asarray(molecule_data.positions, dtype=np.float64)
    mol_q64 = np.asarray(molecule_data.charges, dtype=np.float64)
    K = int(mol_q64.shape[0])
    d_mx, d_my, d_mz, d_mq, d_mt = _get_mol_buffers(K)

    cu.to_device(np.ascontiguousarray(mol_pos64[:, 0], dtype=np.float32), to=d_mx)
    cu.to_device(np.ascontiguousarray(mol_pos64[:, 1], dtype=np.float32), to=d_my)
    cu.to_device(np.ascontiguousarray(mol_pos64[:, 2], dtype=np.float32), to=d_mz)
    cu.to_device(np.ascontiguousarray(mol_q64, dtype=np.float32), to=d_mq)
    cu.to_device(np.ascontiguousarray(molecule_data.atoms, dtype=np.int32), to=d_mt)

    # Scalars
    box = np.asarray(props.box, dtype=np.float32).reshape(3)
    periodic = np.int32(1 if props.periodic else 0)
    cutoff = float(model.cutoff)
    cut2 = np.float32(cutoff * cutoff)
    alpha = np.float32(_ewald_alpha(cutoff, float(model.tolerance)))
    coulomb_coeff = np.float32(float(model.coulomb_coeff))

    # Launch config
    threads = 256
    blocks = (N + threads - 1) // threads
    d_partials = _get_partials(blocks)

    _coulomb_extra_soa[blocks, threads](
        d_x,
        d_y,
        d_z,
        d_q,
        N,
        d_mx,
        d_my,
        d_mz,
        d_mq,
        K,
        m_start,
        m_stop,
        box[0],
        box[1],
        box[2],
        periodic,
        alpha,
        cut2,
        coulomb_coeff,
        d_partials,
    )

    extra = float(d_partials.copy_to_host()[:blocks].sum(dtype=np.float64))

    # Intra on CPU (small K); uses float64 erfc from numpy
    intra = _coulomb_intra_cpu(
        mol_pos64,
        mol_q64,
        np.asarray(props.box, dtype=np.float64).reshape(3),
        bool(props.periodic),
        cutoff,
        float(alpha),
        float(model.coulomb_coeff),
    )

    return float(np.array(extra + intra, dtype=FLOAT_DTYPE))


def lennard_jones_gpu(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    cu = _require_cuda()

    if hasattr(data, "enable_gpu"):
        data.enable_gpu(cuda=cu, pos_dtype=np.float32, q_dtype=np.float32, upload=True)

    if not hasattr(data, "device_view"):
        raise RuntimeError(
            "SystemData has no device_view(); install the GPU mirror in data.py first."
        )

    d_x, d_y, d_z, d_q, d_t, sys_start, sys_size, *_ = data.device_view(cuda=cu)
    N = int(sys_size)

    m_start, m_stop = molecule_data.data_interval()
    m_start = int(m_start)
    m_stop = int(m_stop)
    if not (0 <= m_start <= m_stop <= N):
        m_start = -1
        m_stop = -1

    # molecule upload (pos + types)
    mol_pos64 = np.asarray(molecule_data.positions, dtype=np.float64)
    K = int(mol_pos64.shape[0])
    d_mx, d_my, d_mz, d_mq, d_mt = _get_mol_buffers(K)  # reuse q buffer, harmless

    cu.to_device(np.ascontiguousarray(mol_pos64[:, 0], dtype=np.float32), to=d_mx)
    cu.to_device(np.ascontiguousarray(mol_pos64[:, 1], dtype=np.float32), to=d_my)
    cu.to_device(np.ascontiguousarray(mol_pos64[:, 2], dtype=np.float32), to=d_mz)
    cu.to_device(np.ascontiguousarray(molecule_data.atoms, dtype=np.int32), to=d_mt)

    # LJ table on device
    d_table = _get_lj_table_device(model)

    box = np.asarray(props.box, dtype=np.float32).reshape(3)
    periodic = np.int32(1 if props.periodic else 0)
    cutoff = float(model.cutoff)
    cut2 = np.float32(cutoff * cutoff)

    threads = 256
    blocks = (N + threads - 1) // threads
    d_partials = _get_partials(blocks)

    _lj_extra_soa[blocks, threads](
        d_x,
        d_y,
        d_z,
        d_t,
        N,
        d_mx,
        d_my,
        d_mz,
        d_mt,
        K,
        m_start,
        m_stop,
        box[0],
        box[1],
        box[2],
        periodic,
        cut2,
        d_table,
        d_partials,
    )

    extra = float(d_partials.copy_to_host()[:blocks].sum(dtype=np.float64))

    # Intra on CPU
    intra = _lj_intra_cpu(
        mol_pos64,
        np.asarray(molecule_data.atoms, dtype=np.int32),
        np.asarray(model.ljtable.table, dtype=np.float64),
        np.asarray(props.box, dtype=np.float64).reshape(3),
        bool(props.periodic),
        cutoff,
    )

    return float(np.array(extra + intra, dtype=FLOAT_DTYPE))


def coulomb_lj_gpu(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    """
    Fused Coulomb + LJ. Use this if you want fewer kernel launches per evaluation.
    """
    cu = _require_cuda()

    if hasattr(data, "enable_gpu"):
        data.enable_gpu(cuda=cu, pos_dtype=np.float32, q_dtype=np.float32, upload=True)

    if not hasattr(data, "device_view"):
        raise RuntimeError(
            "SystemData has no device_view(); install the GPU mirror in data.py first."
        )

    d_x, d_y, d_z, d_q, d_t, sys_start, sys_size, *_ = data.device_view(cuda=cu)
    N = int(sys_size)

    m_start, m_stop = molecule_data.data_interval()
    m_start = int(m_start)
    m_stop = int(m_stop)
    if not (0 <= m_start <= m_stop <= N):
        m_start = -1
        m_stop = -1

    mol_pos64 = np.asarray(molecule_data.positions, dtype=np.float64)
    mol_q64 = np.asarray(molecule_data.charges, dtype=np.float64)
    K = int(mol_q64.shape[0])
    d_mx, d_my, d_mz, d_mq, d_mt = _get_mol_buffers(K)

    cu.to_device(np.ascontiguousarray(mol_pos64[:, 0], dtype=np.float32), to=d_mx)
    cu.to_device(np.ascontiguousarray(mol_pos64[:, 1], dtype=np.float32), to=d_my)
    cu.to_device(np.ascontiguousarray(mol_pos64[:, 2], dtype=np.float32), to=d_mz)
    cu.to_device(np.ascontiguousarray(mol_q64, dtype=np.float32), to=d_mq)
    cu.to_device(np.ascontiguousarray(molecule_data.atoms, dtype=np.int32), to=d_mt)

    d_table = _get_lj_table_device(model)

    box = np.asarray(props.box, dtype=np.float32).reshape(3)
    periodic = np.int32(1 if props.periodic else 0)
    cutoff = float(model.cutoff)
    cut2 = np.float32(cutoff * cutoff)
    alpha = np.float32(_ewald_alpha(cutoff, float(model.tolerance)))
    coulomb_coeff = np.float32(float(model.coulomb_coeff))

    threads = 256
    blocks = (N + threads - 1) // threads
    d_partials = _get_partials(blocks)

    _coulomb_lj_extra_soa[blocks, threads](
        d_x,
        d_y,
        d_z,
        d_q,
        d_t,
        N,
        d_mx,
        d_my,
        d_mz,
        d_mq,
        d_mt,
        K,
        m_start,
        m_stop,
        box[0],
        box[1],
        box[2],
        periodic,
        alpha,
        cut2,
        coulomb_coeff,
        d_table,
        d_partials,
    )

    extra = float(d_partials.copy_to_host()[:blocks].sum(dtype=np.float64))

    intra = _coulomb_intra_cpu(
        mol_pos64,
        mol_q64,
        np.asarray(props.box, dtype=np.float64).reshape(3),
        bool(props.periodic),
        cutoff,
        float(alpha),
        float(model.coulomb_coeff),
    ) + _lj_intra_cpu(
        mol_pos64,
        np.asarray(molecule_data.atoms, dtype=np.int32),
        np.asarray(model.ljtable.table, dtype=np.float64),
        np.asarray(props.box, dtype=np.float64).reshape(3),
        bool(props.periodic),
        cutoff,
    )

    return float(np.array(extra + intra, dtype=FLOAT_DTYPE))
