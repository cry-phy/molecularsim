from __future__ import annotations

import numpy as np

from molecsim.data.contributors import Contribution
from molecsim.data.data import SystemData, FLOAT_DTYPE
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties

try:
    from numba import cuda
except Exception:
    cuda = None


# ----------------------------
# GPU math helpers
# ----------------------------

if cuda is not None:
    from math import sqrt, exp  # device-supported subset

    @cuda.jit(device=True, inline=True)
    def _erfc_approx(x: float) -> float:
        """
        Approx erfc(x) suitable for GPU.
        This is a placeholder: replace with your preferred approximation.

        Uses a classic Abramowitz-Stegun style approximation for erf,
        then erfc = 1 - erf.
        Accuracy is "decent" for many MD/MC cutoffs but you should validate.
        """
        # erf approximation constants
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
# Coulomb GPU kernels
# ----------------------------

if cuda is not None:

    @cuda.jit
    def _coulomb_extra_kernel(
        data_pos,  # (M,3) float64
        data_q,  # (M,)  float64
        mol_pos,  # (N,3) float64
        mol_q,  # (N,)  float64
        box,  # (3,)  float64
        periodic,  # int32 (0/1)
        coulomb_coeff,  # float64
        cutoff,  # float64
        out_partial,  # (grid_size,) float64
    ):
        """
        Each thread handles one i in data and sums over all mol atoms j.
        Writes per-block partial sum to out_partial[blockIdx.x] via shared reduction.
        """
        i = cuda.grid(1)
        M = data_pos.shape[0]
        N = mol_pos.shape[0]

        smem = cuda.shared.array(256, dtype=np.float64)  # assumes blockDim<=256
        tid = cuda.threadIdx.x

        acc = 0.0
        if i < M:
            xi = data_pos[i, 0]
            yi = data_pos[i, 1]
            zi = data_pos[i, 2]
            qi = data_q[i]

            for j in range(N):
                dx = xi - mol_pos[j, 0]
                dy = yi - mol_pos[j, 1]
                dz = zi - mol_pos[j, 2]

                if periodic != 0:
                    # MIC: dx -= box * round(dx/box)
                    dx -= box[0] * np.rint(dx / box[0])
                    dy -= box[1] * np.rint(dy / box[1])
                    dz -= box[2] * np.rint(dz / box[2])

                r2 = dx * dx + dy * dy + dz * dz
                r = sqrt(r2)

                # matches your CPU: erfc(r - cutoff)
                cut = _erfc_approx(r - cutoff)

                acc += (qi * mol_q[j]) * (cut / r)

            acc *= coulomb_coeff
            # NOTE: your CPU used "/2" with a mask approach; this extra term is rectangular,
            # so typically no "/2" is applied here.

        smem[tid] = acc
        cuda.syncthreads()

        # block reduction
        stride = cuda.blockDim.x // 2
        while stride > 0:
            if tid < stride:
                smem[tid] += smem[tid + stride]
            cuda.syncthreads()
            stride //= 2

        if tid == 0:
            out_partial[cuda.blockIdx.x] = smem[0]

    @cuda.jit
    def _coulomb_intra_kernel(
        mol_pos,
        mol_q,
        box,
        periodic,
        coulomb_coeff,
        cutoff,
        out_one,
    ):
        """Intra for small N: run on one block, one thread does it all (simple draft), or parallelize over pairs if you ever need bigger molecules."""
        if cuda.blockIdx.x != 0 or cuda.threadIdx.x != 0:
            return

        N = mol_pos.shape[0]
        acc = 0.0
        for i in range(N):
            xi = mol_pos[i, 0]
            yi = mol_pos[i, 1]
            zi = mol_pos[i, 2]
            qi = mol_q[i]
            for j in range(i + 1, N):
                dx = xi - mol_pos[j, 0]
                dy = yi - mol_pos[j, 1]
                dz = zi - mol_pos[j, 2]
                if periodic != 0:
                    dx -= box[0] * np.rint(dx / box[0])
                    dy -= box[1] * np.rint(dy / box[1])
                    dz -= box[2] * np.rint(dz / box[2])
                r2 = dx * dx + dy * dy + dz * dz
                r = sqrt(r2)
                cut = _erfc_approx(r - cutoff)
                acc += (qi * mol_q[j]) * (cut / r)

        out_one[0] = coulomb_coeff * acc


def coulomb_gpu(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    """GPU draft: same signature, CUDA backend if available, else raises."""
    if cuda is None:
        raise RuntimeError("CUDA not available (numba.cuda.is_available() == False)")

    if len(molecule_data) > len(data):
        raise ValueError(
            f"Data must be larger or equal to molecule_data: data {len(data)}, molecule_data: {len(molecule_data)}"
        )

    # host arrays
    data_pos = np.asarray(data.positions, dtype=np.float64)
    data_q = np.asarray(data.charges, dtype=np.float64)
    mol_pos = np.asarray(molecule_data.positions, dtype=np.float64)
    mol_q = np.asarray(molecule_data.charges, dtype=np.float64)
    box = np.asarray(props.box, dtype=np.float64)
    periodic = np.int32(1 if props.periodic else 0)

    # device
    d_data_pos = cuda.to_device(data_pos)
    d_data_q = cuda.to_device(data_q)
    d_mol_pos = cuda.to_device(mol_pos)
    d_mol_q = cuda.to_device(mol_q)
    d_box = cuda.to_device(box)

    # launch config
    threads = 256
    blocks = (data_pos.shape[0] + threads - 1) // threads

    d_partials = cuda.device_array((blocks,), dtype=np.float64)
    d_intra = cuda.device_array((1,), dtype=np.float64)

    _coulomb_extra_kernel[blocks, threads](
        d_data_pos,
        d_data_q,
        d_mol_pos,
        d_mol_q,
        d_box,
        periodic,
        float(model.coulomb_coeff),
        float(model.cutoff),
        d_partials,
    )
    _coulomb_intra_kernel[1, 1](
        d_mol_pos,
        d_mol_q,
        d_box,
        periodic,
        float(model.coulomb_coeff),
        float(model.cutoff),
        d_intra,
    )

    extra = float(d_partials.copy_to_host().sum())
    intra = float(d_intra.copy_to_host()[0])

    return float(np.array(extra + intra, dtype=FLOAT_DTYPE))


if cuda is not None:

    @cuda.jit
    def _lj_extra_kernel(
        data_pos,
        data_types,
        mol_pos,
        mol_types,
        table,  # (T,T,2)
        box,
        periodic,
        out_partial,
    ):
        i = cuda.grid(1)
        M = data_pos.shape[0]
        N = mol_pos.shape[0]

        smem = cuda.shared.array(256, dtype=np.float64)
        tid = cuda.threadIdx.x

        acc = 0.0
        if i < M:
            xi = data_pos[i, 0]
            yi = data_pos[i, 1]
            zi = data_pos[i, 2]
            ti = data_types[i]

            for j in range(N):
                dx = xi - mol_pos[j, 0]
                dy = yi - mol_pos[j, 1]
                dz = zi - mol_pos[j, 2]
                if periodic != 0:
                    dx -= box[0] * np.rint(dx / box[0])
                    dy -= box[1] * np.rint(dy / box[1])
                    dz -= box[2] * np.rint(dz / box[2])
                r2 = dx * dx + dy * dy + dz * dz

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

    @cuda.jit
    def _lj_intra_kernel(mol_pos, mol_types, table, box, periodic, out_one):
        if cuda.blockIdx.x != 0 or cuda.threadIdx.x != 0:
            return

        N = mol_pos.shape[0]
        acc = 0.0
        for i in range(N):
            xi = mol_pos[i, 0]
            yi = mol_pos[i, 1]
            zi = mol_pos[i, 2]
            ti = mol_types[i]
            for j in range(i + 1, N):
                dx = xi - mol_pos[j, 0]
                dy = yi - mol_pos[j, 1]
                dz = zi - mol_pos[j, 2]
                if periodic != 0:
                    dx -= box[0] * np.rint(dx / box[0])
                    dy -= box[1] * np.rint(dy / box[1])
                    dz -= box[2] * np.rint(dz / box[2])
                r2 = dx * dx + dy * dy + dz * dz

                tj = mol_types[j]
                Aij = table[ti, tj, 0]
                Bij = table[ti, tj, 1]
                r6 = r2 * r2 * r2
                r12 = r6 * r6
                acc += (Aij / r12) - (Bij / r6)

        out_one[0] = acc


def lennard_jones_gpu(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    if cuda is None:
        raise RuntimeError("CUDA not available (numba.cuda.is_available() == False)")

    data_pos = np.asarray(data.positions, dtype=np.float64)
    mol_pos = np.asarray(molecule_data.positions, dtype=np.float64)
    data_types = np.asarray(data.atoms, dtype=np.int32)
    mol_types = np.asarray(molecule_data.atoms, dtype=np.int32)

    table = np.asarray(model.ljtable.table, dtype=np.float64)
    box = np.asarray(props.box, dtype=np.float64)
    periodic = np.int32(1 if props.periodic else 0)

    d_data_pos = cuda.to_device(data_pos)
    d_mol_pos = cuda.to_device(mol_pos)
    d_data_types = cuda.to_device(data_types)
    d_mol_types = cuda.to_device(mol_types)
    d_table = cuda.to_device(table)
    d_box = cuda.to_device(box)

    threads = 256
    blocks = (data_pos.shape[0] + threads - 1) // threads

    d_partials = cuda.device_array((blocks,), dtype=np.float64)
    d_intra = cuda.device_array((1,), dtype=np.float64)

    _lj_extra_kernel[blocks, threads](
        d_data_pos,
        d_data_types,
        d_mol_pos,
        d_mol_types,
        d_table,
        d_box,
        periodic,
        d_partials,
    )
    _lj_intra_kernel[1, 1](d_mol_pos, d_mol_types, d_table, d_box, periodic, d_intra)

    extra = float(d_partials.copy_to_host().sum())
    intra = float(d_intra.copy_to_host()[0])

    return float(np.array(extra + intra, dtype=FLOAT_DTYPE))
