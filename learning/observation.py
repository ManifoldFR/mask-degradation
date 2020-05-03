import torch
from torch import Tensor
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam

from scipy import constants
from physics import penetration

import configs


_DEBIT = 85*constants.liter/constants.minute


class ObservationModel(PyroModule):
    """Observation model."""
    
    def __init__(self, name=''):
        """
        Parameters
        ----------
        resp_model : callable that generates a tuple (surface_area, layers)
        """
        super().__init__(name=name)
        self.obs_scale = PyroSample(dist.InverseGamma(torch.tensor(2.), 0.5))

    def forward(self, diameters: Tensor, resp_model=configs.respiratorA, debit=_DEBIT):
        """
        Parameters
        ----------
        diameters (Tensor): particle diameters
        resp_model (function):
            Tuple of surface area and list of `~MaskLayers`.
        """
        surface_area_, layers_ = resp_model()

        face_vel = debit / surface_area_

        with pyro.plate("diameters"):
            phi = pyro.deterministic("phi", penetration.compute_penetration_profile(
                diameters, layers_, face_vel,
                configs.temperature, configs.viscosity, return_log=True
            ))
            
            obs = pyro.sample('obs_log', dist.Normal(loc=phi, scale=self.obs_scale))
        return obs
