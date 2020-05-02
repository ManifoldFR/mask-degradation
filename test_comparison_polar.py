"""Reproduce the curves from Bałazy, A. et al. comparing the theoretical models
for particle penetration through the mechanical and electret filters."""
import numpy as np
from scipy import constants
from physics import penetration
from physics.penetration import MaskLayer

from configs import respiratorA, temperature, viscosity


# Load mask configuration
surface_area, layer_params = respiratorA()

debit = 85 * constants.liter / constants.minute
face_vel = debit / surface_area

print(layer_params)

# Particle size range

particle_diam_log = np.linspace(np.log(10), np.log(1000), 41)
particle_diam = np.exp(particle_diam_log) * constants.nano

results = penetration.compute_penetration_profile(
    particle_diam, layer_params, face_vel, temperature, viscosity,
    return_log=True)


## Test without polarization capture term

for param in layer_params:
    param.charge_density = None
    param.permittivity = None

results_nopolar = penetration.compute_penetration_profile(
    particle_diam, layer_params, face_vel, temperature, viscosity
)

mpps_nopolar = particle_diam[np.argmax(results_nopolar)]

import matplotlib.pyplot as plt

plt.style.use('ggplot')

plt.plot(particle_diam, results, label="With charge term")
plt.plot(particle_diam, results_nopolar, ls='--', label="No charge term")
plt.vlines(mpps_nopolar, *plt.ylim(), ls='--')
plt.xscale('log')
plt.xlabel("Particle size $d_p$")
plt.ylabel("Penetration")
plt.title("Parameters: $U_0={:.3g}$".format(face_vel))
plt.legend()
plt.show()
plt.tight_layout()
