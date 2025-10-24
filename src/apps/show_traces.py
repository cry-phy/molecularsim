import numpy as np
from molecsim.utils import RNG

from molecsim.system.data import SystemData
from molecsim.system.molecule import Molecule
from molecsim.utils.place import randomized_system
from molecsim.utils.plot import add_periodic_debug_traces, plot_system
import plotly.graph_objects as go


class DotDict(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]


data = SystemData.as_owner(
    ["H"], np.array([0.0, 0.0, 0.0]).reshape((1, 3)), np.array([1.0])
)
molec = Molecule(data)
molec_dict = {molec: 5}

custom_props = DotDict(
    temperature=10,
    box=np.array([10, 10, 10]),
    molecules=molec_dict,
    lj_params=(1, 1),
    coulomb_coeff=1,
    heat_energy=1,
)

rng = RNG(1)

sim = randomized_system(custom_props, rng)

fig, update = plot_system(sim)

fig = add_periodic_debug_traces(fig, sim)

fig.show()
