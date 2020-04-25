import numpy as np
from scipy import constants
from configs import respirator_A
from physics import compute_penetration_profile
from physics.dynamics import LinearDegradation

import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.style.use('ggplot')

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
DEGRADE_RATE = 0.04 / constants.hour  # mask characteristics degrade at 2%/hour

degrad = LinearDegradation(DEGRADE_RATE, layer_params)

# 2 hours
n_steps = 91
times = np.linspace(0, 2, n_steps) * constants.hour

results = []

for t in times:
    state = degrad.step(t)
    print("Time t:", state)
    res_t = compute_penetration_profile(
        particle_diam, state, face_vel, temp, viscosity)
    results.append(res_t)


fig: plt.Figure = plt.figure(dpi=70)
ax = fig.add_subplot(111)

text: plt.Text = plt.text(.1, .1, "$t={:.3g}$".format(0),
                          transform=ax.transAxes)
line, = plt.plot(particle_diam, results[0])

def init():
    line.set_ydata(results[0])
    return line, text,

def anim_func(k):
    text.set_text("$t={:.3f}$h".format(times[k] / constants.hour))
    line.set_ydata(results[k])
    return line, text,

fps = 15
anim_ = animation.FuncAnimation(fig, anim_func,# init_func=init,
                                frames=n_steps,
                                interval=1e3/fps, repeat=False)

ylims_ = plt.ylim()
plt.ylim((ylims_[0], np.max(results) * 1.05))
plt.xlabel("Particle size $d_p$")
plt.ylabel("Penetration")
plt.xscale('log')
plt.tight_layout()
plt.show()

anim_.save("linear_degradation.gif", writer='imagemagick',
           fps=fps)
