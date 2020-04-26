import numpy as np
from scipy import constants
from physics import compute_penetration_profile
from physics import dynamics

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from configs import respirator_A, temperature, viscosity

plt.style.use('ggplot')

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

mode = "sigmoid"
if mode == "linear":
    degrad = dynamics.LinearDegradation(DEGRADE_RATE, layer_params)
elif mode == "sigmoid":
    DEGRADE_RATE = 3 / constants.hour  # mask characteristics degrade at 2%/hour
    degrad = dynamics.SigmoidDegradation(DEGRADE_RATE, layer_params, tau_0=1.4*constants.hour)


if __name__ == "__main__":
    
    # 2 hours
    n_steps = 91
    times = np.linspace(0, 2, n_steps) * constants.hour

    results = []
    charge_t = []

    for t in times:
        state = degrad.step(t)
        res_t = compute_penetration_profile(
            particle_diam, state, face_vel, temperature, viscosity)
        results.append(res_t)
        charge_t.append([l.charge_density for l in state])


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
    plt.title("Penetration evolution under sigmoid charge evolution")
    plt.xlabel("Particle size $d_p$")
    plt.ylabel("Penetration")
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

    charge_t = np.asarray(charge_t)

    plt.figure()
    plt.plot(times, charge_t)
    plt.title("Charge density evolution")
    plt.xlabel("Time")
    plt.ylabel("Charge density (C/m)")
    plt.show()

    # anim_.save("assets/{}_degradation.gif".format(mode), writer='imagemagick', fps=fps)
