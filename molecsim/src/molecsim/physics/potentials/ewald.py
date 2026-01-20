from molecsim.data.contributors import Contribution, cache_value, uncache_value
from molecsim.data.data import COMPLEX_TYPE, StructureArray, SystemData
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
import numpy as np
import numpy.typing as npt

from molecsim.data.system import SimSystem
from molecsim.physics.potentials.coulomb_pot import coulomb


def _compute_reciprocals(
    box: StructureArray,
    alpha: float,
    kmax: float,
) -> tuple[StructureArray, StructureArray]:
    kx_max = int(np.ceil(kmax * box[0] / (2.0 * np.pi)))
    ky_max = int(np.ceil(kmax * box[1] / (2.0 * np.pi)))
    kz_max = int(np.ceil(kmax * box[2] / (2.0 * np.pi)))

    nx = np.arange(-kx_max, kx_max + 1, dtype=int)
    ny = np.arange(-ky_max, ky_max + 1, dtype=int)
    nz = np.arange(-kz_max, kz_max + 1, dtype=int)

    gx, gy, gz = np.meshgrid(nx, ny, nz, indexing="ij")
    nvecs = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))

    keep = ~((nvecs[:, 0] == 0) & (nvecs[:, 1] == 0) & (nvecs[:, 2] == 0))
    nvecs = nvecs[keep]

    ks = np.empty((nvecs.shape[0], 3), dtype=float)
    ks[:, 0] = (2.0 * np.pi) * nvecs[:, 0] / box[0]
    ks[:, 1] = (2.0 * np.pi) * nvecs[:, 1] / box[1]
    ks[:, 2] = (2.0 * np.pi) * nvecs[:, 2] / box[2]

    k2 = np.sum(ks * ks, axis=1)
    keep = k2 <= (kmax * kmax)
    ks = ks[keep]
    k2 = k2[keep]

    volume = float(np.prod(box))
    ws = (2.0 * np.pi / volume) * np.exp(-k2 / (4.0 * alpha * alpha)) / k2
    return ks, ws


class EwaldProperties:
    """Contains Properties used for ewald summation."""

    def __init__(
        self,
        rcut: float,
        alpha: float,
        kmax: float,
        ks: StructureArray,
        ws: StructureArray,
        s: npt.NDArray[np.complex128],
        max_s_reuse: int,
    ) -> None:
        """Initialize all fields directly.

        Not intended for direct usage.
        """
        self._ks = ks
        self._ws = ws
        self._rcut = rcut
        self._alpha = alpha
        self._kmax = kmax
        self._s = s
        self._s_reuse_counter = 0
        self._max_s_reuse = int(max_s_reuse)

    @property
    def ks(self):
        """Vectors k."""
        return self._ks

    @property
    def ws(self):
        """Weights for S(k) split."""
        return self._ws

    @property
    def rcut(self):
        """Real space cutoff for coulomb."""
        return self._rcut

    @property
    def alpha(self):
        """Ewald cutoff."""
        return self._alpha

    @property
    def kmax(self):
        """Maximum k vector length."""
        return self._kmax

    @property
    def s(self):
        """S(k) vector."""
        return self._s

    def refresh_s(self, data: SystemData, model: ModelConstants) -> None:
        """Refresh S(k) with current system positions."""
        self._s = compute_s(data, self)
        self._s_reuse_counter = 0

    def commit_delta_s(
        self,
        delta_s: npt.NDArray[np.complex128],
        data: SystemData,
        model: ModelConstants,
    ) -> None:
        """Update S(k) with the difference to old state."""
        self._s += delta_s
        self._s_reuse_counter += 1
        if self._s_reuse_counter >= self._max_s_reuse:
            self.refresh_s(data, model)

    def __update__(
        self,
        sim: SimSystem,
        old_contribution: Contribution,
        new_contribution: Contribution,
        key: str,
    ) -> None:
        """Update method used to update cached quantities."""
        old_cache = uncache_value(self, "s", old_contribution)
        new_cache = uncache_value(self, "s", new_contribution)
        self._s += new_cache - old_cache

    def cache_s(self, s: npt.NDArray[COMPLEX_TYPE], contribution: Contribution) -> None:
        """Cache S(k) to known key."""
        cache_value(self, "s", s, contribution)

    def uncache_s(self, contribution: Contribution) -> npt.NDArray[COMPLEX_TYPE]:
        """Get S(k) from cache."""
        return uncache_value(self, "s", contribution)


def generate_ewald(
    data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    max_s_reuse: int = 100,
) -> EwaldProperties:
    """Generate Ewald Properties to be used for PotCall."""
    rcut = float(model.cutoff)
    p = -np.log(float(model.tolerance))
    alpha = np.sqrt(p) / rcut
    kmax = 2.0 * alpha * np.sqrt(p)

    ks, ws = _compute_reciprocals(props.box, alpha, kmax)
    s0 = np.zeros(ks.shape[0], dtype=np.complex128)

    ewald_props = EwaldProperties(
        rcut=rcut,
        alpha=alpha,
        kmax=kmax,
        ks=ks,
        ws=ws,
        s=s0,
        max_s_reuse=max_s_reuse,
    )
    ewald_props.refresh_s(data, model)
    return ewald_props


def compute_s(
    data: SystemData,
    ewald_props: EwaldProperties,
) -> npt.NDArray[COMPLEX_TYPE]:
    """Compute vector S(k)."""
    exponents = np.sum(ewald_props.ks[:, None] * data.positions[None, :], axis=2)
    s_per_atom = data.charges * np.exp(1j * exponents)
    s = np.sum(s_per_atom, axis=1)
    return s


def reciprocal_ewald(
    molecule_data: SystemData,
    model: ModelConstants,
    ewald_props: EwaldProperties,
    contribution: Contribution,
) -> float:
    """Compute Coulomb contribution of molecule with ewald potential."""
    s_molecule = compute_s(molecule_data, ewald_props)
    ewald_props.cache_s(s_molecule, contribution)
    s_rest = ewald_props.s - s_molecule
    mode_parts = s_molecule**2 + 2 * np.real(s_molecule * np.conjugate(s_rest))
    contrib = np.sum(ewald_props.ws * mode_parts).astype(float).value
    return model.coulomb_coeff * contrib


def ewald(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    ewald_props: EwaldProperties,
    *,
    contribution: Contribution,
) -> float:
    """Compute Coulomb contribution with ewald potential."""
    real_part = coulomb(
        data,
        molecule_data,
        props,
        model,
    )
    reciprocal_part = reciprocal_ewald(molecule_data, model, ewald_props, contribution)
    return real_part + reciprocal_part
