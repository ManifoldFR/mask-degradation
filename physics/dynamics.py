"""Add degradation dynamics to the mask layer structure."""
from physics.penetration import compute_penetration_profile, LayerParams
from scipy import constants
from typing import List
import copy
import dataclasses


class LinearDegradation:
    
    def __init__(self, rate, init_layer_params: List[LayerParams], names=None):
        """
        rate :
            If given a float, then intepreted as the fiber charge degradation slope. If a dictionary,
            its keys specify the parameters of the layers that will be degraded.
        """
        super().__init__()
        self.init_params = init_layer_params
        self.state = copy.deepcopy(init_layer_params)
        # names to degrade
        # slope at which to degrade params
        # as % of initial parameter
        if isinstance(rate, dict):
            self.slope = rate
        else:    
            self.slope = {
                "charge_density": rate
            }


    def step(self, t: float):
        """Step through the model to time t.
        
        t: float
            Current time.
        """
        state = []
        for params in self.init_params:
            # convert layer parameters to dict
            coeffs_ = dataclasses.asdict(params)
            coeffs_.update({
                k: (1. - self.slope[k] * t) * getattr(params, k) for k in self.slope.keys()
            })
            state.append(LayerParams(**coeffs_))
        self.state = state
        return state


class DispersionModelDegradation:
    """Model mask degradation using physical parameters."""
    pass
