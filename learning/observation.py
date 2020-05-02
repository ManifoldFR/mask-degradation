import torch
from torch import Tensor
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule

from scipy import constants
from physics import penetration

import configs


_DEBIT = 85*constants.liter/constants.minute


class Observation(PyroModule):
    """Observation model."""
    
    def __init__(self, name='', resp_model=configs.respiratorA, debit=_DEBIT):
        """
        Parameters
        ----------
        resp_model : callable that generates a tuple (surface_area, layers)
        """
        super().__init__(name=name)
        
        self.surface_area_, self.layers_ = resp_model()
        self.debit = debit
        self.face_vel = debit / self.surface_area_
        
        self.obs_scale = torch.tensor(0.1)


    def forward(self, diameters: Tensor):
        """
        Parameters
        ----------
        diameters (Tensor): particle diameters
        resp_model (function):
            Tuple of surface area and list of `~MaskLayers`.
        """
        
        
        with pyro.plate("diameters"):
            phi = penetration.compute_penetration_profile(
                diameters, self.layers_, self.face_vel,
                configs.temperature, configs.viscosity, return_log=True
            )
            
            obs = pyro.sample('obs_log', dist.Normal(loc=phi, scale=self.obs_scale))
        return obs
