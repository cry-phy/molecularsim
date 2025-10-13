from molecsim.system.data import SystemForces
from molecsim.system.system import SimSystem


def full_potential(
    system: SimSystem,
    forces: SystemForces,
) -> float:
    """Calculate full system potential from all contributing calculations."""
    data = system.data
    potential = 0
    for force_fn, args, kwargs, _, _ in forces:
        potential += force_fn(data, *args, **kwargs)
    return potential
