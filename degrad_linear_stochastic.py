import numpy as np
from scipy import constants
from configs import respirator_A
from physics import compute_penetration_profile
from physics.dynamics import LinearDegradation

import math
import torch
import pyro
import pyro.distributions as dist

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from configs import temperature, viscosity

plt.style.use('ggplot')


# Load mask configuration
surface_area, layer_params = respirator_A()

debit = 85 * constants.liter / constants.minute
face_vel = debit / surface_area

# Particle size range

particle_diam_log = torch.linspace(np.log(10), np.log(1000), 41)
particle_diam = torch.exp(particle_diam_log) * constants.nano

# 2 hours
n_steps = 91
times = torch.linspace(0, 2, n_steps) * constants.hour

def model():
    loc = 13 * constants.nano
    std = 1 * constants.nano
    
    # Lognormal prior
    charge = pyro.sample("charge", dist.Normal(loc, std))
    
    surface_area, layer_params = respirator_A()
    
    for param in layer_params:
        param.charge_density = charge

    # Create degradation model
    DEGRADE_RATE = 0.04 / constants.hour  # mask characteristics degrade at 2%/hour

    degrad = LinearDegradation(DEGRADE_RATE, layer_params)

    # Step through the model.
    results = []

    for t in times:
        state = degrad.step(t)
        res_t = compute_penetration_profile(
            particle_diam, state, face_vel, temperature, viscosity)
        results.append(res_t)

    results = torch.stack(results)
    results.transpose_(0, 1)
    return results

# Run model in a plate with batching
with pyro.plate("charge_batch", 1000, dim=-2) as ctx:
    results = model()

results_mean = results.mean(0)
results_low = results_mean - 2 * results.std(0)
results_high = results_mean + 2 * results.std(0)

THRESHOLD = 6e-2


fig: plt.Figure = plt.figure(dpi=70)
ax = fig.add_subplot(111)



text: plt.Text = plt.text(.1, .1, "$t={:.3g}$".format(0),
                          transform=ax.transAxes)
line, = plt.plot(particle_diam, results_mean[0])


def init():
    line.set_ydata(results[0])
    conf_interv = plt.fill_between(particle_diam, results_low[0], results_high[0],
                                   alpha=.4, facecolor='C0')
    return line, text, conf_interv

def anim_func(k):
    text.set_text("$t={:.3f}$h".format(times[k] / constants.hour))
    line.set_ydata(results_mean[k])
    conf_interv = plt.fill_between(particle_diam, results_low[k], results_high[k],
                                   alpha=.4, facecolor='C0')
    return line, text, conf_interv

fps = 15
anim_ = animation.FuncAnimation(fig, anim_func, init_func=init,
                                frames=n_steps, blit=True,
                                interval=1e3/fps, repeat=False)

plt.hlines(THRESHOLD, particle_diam.min(), particle_diam.max())

ylims_ = plt.ylim()
plt.title("Evolution with $\\mathcal{N}(13, 1)$ in nm")
plt.ylim((ylims_[0], torch.max(results_high) * 1.05))
plt.xlabel("Particle size $d_p$")
plt.ylabel("Penetration")
plt.xscale('log')
plt.tight_layout()
plt.show()

# anim_.save("assets/stochastic_linear_degrad.gif", writer='imagemagick', fps=fps)
