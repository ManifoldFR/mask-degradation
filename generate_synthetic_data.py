"""Generate synthetic data."""
import os
import torch
import numpy as np
import math
import pyro
import pyro.distributions as dist

from configs import respirator_A, temperature, viscosity
from physics import compute_penetration_profile
from scipy import constants

import matplotlib.pyplot as plt


batch_size = 5


surface_area, layer_params = respirator_A()

debit = 85 * constants.liter / constants.minute
face_vel = debit / surface_area

particle_diam = torch.linspace(math.log(10), math.log(1000), 21)
particle_diam.exp_() 
particle_diam *= constants.nano

def model():
    results = torch.log(compute_penetration_profile(
        particle_diam, layer_params, face_vel, temperature, viscosity
    ))

    scale = torch.ones(results.shape) * 1e-1
    mask = (particle_diam >= 100e-9)
    scale[mask] *= 4 * particle_diam[mask] * 1e7
    hetero_skedastic_noise = dist.Normal(0, scale).to_event(1)
    noise = hetero_skedastic_noise.sample((batch_size,))
    
    results_noisy = results + noise
    results_noisy.exp_()
    results.exp_()
    return results, results_noisy

mask = (particle_diam <= 200e-9)

penet_real, penet_samples = model()
penet_samples = penet_samples[:, mask]

plt.figure()
plt.scatter(particle_diam[mask].repeat(batch_size, 1), penet_samples, s=10, alpha=.6)
plt.plot(particle_diam[mask], penet_samples.mean(0), ls='--', c='k')
plt.plot(particle_diam, penet_real, c='r')
plt.title("Samples")
plt.xscale('log')
plt.xlabel('Particle size $d_p$')
plt.tight_layout()
plt.show()

os.makedirs("data", exist_ok=True)
np.savetxt("data/synthetic.txt", penet_samples.numpy())
