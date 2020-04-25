"""Add degradation dynamics to the mask layer structure."""
from physics.penetration import compute_penetration_profile, LayerParams
from scipy import constants
from typing import List
import copy
import dataclasses


class LinearDegradation:
    
    def __init__(self, rate, init_layer_params: List[LayerParams], names=None):
        super().__init__()
        self.init_params = init_layer_params
        self.state = copy.deepcopy(init_layer_params)
        # names to degrade
        self.names = names or ["d_f", "charge_density"]
        # slope at which to degrade params
        # as % of initial parameter
        if isinstance(rate, dict):
            self.slope = rate
        else:    
            self.slope = {
                "d_f": rate,
                "charge_density": -rate
            }


    def step(self, t: float):
        """Initialize parameter slopes as % of initial value.
        
        t: float
            Actual time.
        """
        state = []
        for params in self.state:
            # convert layer parameters to dict
            coeffs_ = dataclasses.asdict(params)
            coeffs_.update({
                k: (1. + self.slope[k] * t) * getattr(params, k) for k in self.names
            })
            state.append(LayerParams(**coeffs_))
        return state

