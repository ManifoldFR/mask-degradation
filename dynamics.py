"""Add degradation dynamics to the mask layer structure."""
from physics.penetration import compute_penetration_profile, LayerParams
from scipy import constants
from typing import List
import copy
import dataclasses

DEGRADE_RATE = 30 / constants.hour  # mask characteristics degrade at 2%/hour


class LinearDegradation:
    
    def __init__(self, init_layer_params: List[LayerParams]):
        super().__init__()
        self.init_params = init_layer_params
        self.state = copy.deepcopy(init_layer_params)
        # slope at which to degrade params
        # as % of initial parameter
        self.slope = DEGRADE_RATE


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



if __name__ == "__main__":
    import numpy as np
    from configs import respirator_A
    
    # Fluid parameters
    temp = 273.15
    viscosity = 1.81e-5  # air viscosity in kg / (m.s)

    # Load mask configuration
    surface_area, layer_params = respirator_A()

    debit = 85 * constants.liter / constants.minute
    face_vel = debit / surface_area

    print(layer_params)

    # Particle size range

    particle_diam_log = np.linspace(np.log(10), np.log(1000), 41)
    particle_diam = np.exp(particle_diam_log) * constants.nano
    
    surface_area, layer_params = respirator_A()
    degrad = LinearDegradation(layer_params)

    # 2 hours
    n_steps = 31
    times = np.linspace(0, 4, n_steps)
    
    results = []
    
    for t in times:
        state = degrad.step(t)
        res_t = compute_penetration_profile(
            particle_diam, state, face_vel, temp, viscosity)
        results.append(res_t)


    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots()
    
    line, = plt.plot(particle_diam, results[0])
    def init():
        line.set_ydata(results[0])
        return line,
    
    def anim_func(k):
        line.set_ydata(results[k])
        return line,
    
    anim_ = animation.FuncAnimation(fig, anim_func, init_func=init,
                                    frames=n_steps,
                                    interval=2, blit=True)

    plt.show()
