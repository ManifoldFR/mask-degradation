"""Propagate uncertainties after putting a prior on the charge."""
from physics import LayerParams, compute_penetration_profile
from configs import respirator_A
from scipy import constants
import torch
import numpy as np
import math

import pyro
import pyro.distributions as dist
from pyro.ops.stats import quantile

from configs import temperature, viscosity


particle_diam_log = torch.linspace(math.log(10), math.log(1000), 41)
particle_diam = torch.exp(particle_diam_log) * constants.nano

def respirator_stochastic():
    loc = (13 * constants.nano)
    std = (1 * constants.nano)
    charge = pyro.sample('q', dist.Normal(loc, std))
    # charge = pyro.sample('q', dist.Uniform(13 * constants.nano, 14 * constants.nano))
    

    surface_area, layer_params = respirator_A(charge)
    debit = 85 * constants.liter / constants.minute
    face_vel = debit / surface_area

    results = compute_penetration_profile(
        particle_diam, layer_params, face_vel, temperature, viscosity
    )

    return results


if __name__ == "__main__":
    
    with pyro.plate("respirator", 1000, dim=-2):
        results: torch.Tensor = respirator_stochastic()

    import matplotlib.pyplot as plt

    plt.style.use('ggplot')
    plt.rcParams['text.usetex'] = True

    # plt.plot(particle_diam, results)
    low_bound, high_bound = quantile(results, (0.05, 0.95))

    fig: plt.Figure = plt.figure()
    plt.plot(particle_diam, results.mean(dim=0), label="Mean")
    plt.fill_between(particle_diam, low_bound, high_bound, alpha=.4,
                    label=r"95\% confidence interval")
    plt.xlabel("Particle size $d_p$")
    plt.ylabel("Penetration")
    plt.xscale('log')
    plt.title("Penetration profile. Prior charge density $q \sim \\mathcal{N}(13, 1)$ nm")
    # plt.title("Prior charge density $q \sim \\mathcal{U}(13, 14)$ nm")
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig.savefig('assets/penetration_gaussian_prior.png')
