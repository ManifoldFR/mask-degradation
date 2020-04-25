import numpy as np
from scipy import constants
from configs import respirator_A
from physics import compute_penetration_profile
from physics.dynamics import LinearDegradation

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Fluid parameters
temp = 273.15
viscosity = 1.81e-5  # air viscosity in kg / (m.s)

# Load mask configuration
surface_area, layer_params = respirator_A()

debit = 85 * constants.liter / constants.minute
face_vel = debit / surface_area

# Particle size range

particle_diam_log = np.linspace(np.log(10), np.log(1000), 41)
particle_diam = np.exp(particle_diam_log) * constants.nano

surface_area, layer_params = respirator_A()

# Create degradation model
DEGRADE_RATE = 30 / constants.hour  # mask characteristics degrade at 2%/hour

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