import numpy as np
from scipy import constants
from physics import penetration
from physics.penetration import LayerParams



# Fluid parameters
temp = 273.15
viscosity = 1.81e-5  # air viscosity in kg / (m.s)
# Define mask characteristics

# face velocity (respirator A at 85 liters/min)
face_vel = 12.9 * constants.centi
# face_vel = 4.5 * constants.centi

# permittivity of polypropylene
permittivity = 2.3

charge_density = 13 * constants.nano

layer_params = [
    LayerParams(39.49*constants.micro, 0.31*constants.milli, 0.165, charge_density, permittivity),
    LayerParams(7.84 *constants.micro, 1.77*constants.milli, 0.069, charge_density, permittivity),
    LayerParams(40.88*constants.micro, 1.05*constants.milli, 0.200, charge_density, permittivity)
]

print(layer_params)

# Particle size range

particle_diam_log = np.linspace(np.log(10), np.log(1000), 41)
particle_diam = np.exp(particle_diam_log) * constants.nano

results = penetration.compute_penetration_profile(
    particle_diam, layer_params, face_vel, temp, viscosity
)


## Test without polarization capture term

for param in layer_params:
    param.charge_density = None
    param.permittivity = None

results_nopolar = penetration.compute_penetration_profile(
    particle_diam, layer_params, face_vel, temp, viscosity
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
plt.legend()
plt.show()

