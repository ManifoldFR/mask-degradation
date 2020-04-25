"""Propagate uncertainties after putting a prior on the charge."""
from physics import LayerParams, compute_penetration_profile
from configs import respirator_A
from scipy import constants
import torch
import numpy as np
import math

import pyro
import pyro.distributions as dist


# q ~ LogNormal(13, 1.0) in nanometers

# Fluid parameters
temp = 273.15
viscosity = 1.81e-5  # air viscosity in kg / (m.s)

# Load mask configuration
particle_diam_log = torch.linspace(math.log(10), math.log(1000), 41)
particle_diam = torch.exp(particle_diam_log) * constants.nano

def respirator_stochastic():
    loc = 13 * constants.nano
    std = 1.0 * constants.nano
    charge = pyro.sample('q', dist.Normal(loc, std))
    

    surface_area, layer_params = respirator_A()
    debit = 85 * constants.liter / constants.minute
    face_vel = debit / surface_area

    for param in layer_params:
        param.charge_density = charge



    results = compute_penetration_profile(
        particle_diam, layer_params, face_vel, temp, viscosity
    )


    return results


with pyro.plate("respirator", 1000, dim=-2):
    results: torch.Tensor = respirator_stochastic()

print(results.shape)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

# plt.plot(particle_diam, results)
low_bound = results.mean(0) - 2 * results.std(0)
high_bound = results.mean(0) + 2 * results.std(0)

plt.plot(particle_diam, results.mean(dim=0), label="Mean")
plt.fill_between(particle_diam, low_bound, high_bound, alpha=.4,
                 label=r"95% confidence interval")
plt.xlabel("Particle size $d_p$")
plt.ylabel("Penetration")
plt.xscale('log')
plt.title("Charge density prior $q \sim \\mathcal{N}(13, 1)$ nm")
plt.legend()
plt.tight_layout()
plt.show()



