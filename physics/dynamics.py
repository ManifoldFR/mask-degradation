"""Add degradation dynamics to the mask layer structure."""
from physics.penetration import compute_penetration_profile, LayerParams
from scipy import constants
from typing import List
import copy
import dataclasses


class LinearDegradation:
    
    def __init__(self, rate, init_layer_params: List[LayerParams]):
        super().__init__()
        self.init_params = init_layer_params
        self.state = copy.deepcopy(init_layer_params)
        # slope at which to degrade params
        # as % of initial parameter
        self.slope = rate


    def step(self, t: float):
        """Initialize parameter slopes as % of initial value.
        
        t: float
            Actual time.
        """
        state = []
        for layer in self.state:
            # convert layer parameters to dict
            params = dataclasses.asdict(layer)
            coeffs_ = [(1 - self.slope * t) * v for k, v in params.items()]
            state.append(LayerParams(*coeffs_))

        return state

