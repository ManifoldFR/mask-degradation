import numpy as np
import torch
import math
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
from torch.distributions import constraints

from physics import compute_penetration_profile
from scipy import constants
from configs import respiratorA, temperature, viscosity

seed = 43
np.random.seed(seed)
pyro.set_rng_seed(seed)


# particle_diam = torch.linspace(math.log(10), math.log(1000), 21)
# particle_diam.exp_()
# particle_diam *= constants.nano

# prior distribution on q
theta = 1 * constants.nano
beta = 1./theta
alpha = 3
# We use a Gamma prior -- take care in Pyro Gamma is parameterized using (alpha, beta) !
charge_prior = dist.Gamma(alpha, beta)

def get_profile(diameters):
    
    charge = pyro.sample('q', charge_prior)

    surface_area, layer_params = respiratorA(charge)
    
    debit = 85 * constants.liter / constants.minute
    face_vel = debit / surface_area

    results = compute_penetration_profile(
        diameters, layer_params, face_vel, temperature, viscosity)
    
    obs_scale = pyro.param('obs_scale', .05 * torch.ones(1),
                           constraint=constraints.positive)
    
    obs = pyro.sample('obs', dist.Normal(results, obs_scale))
    # obs = pyro.deterministic('obs', results)
    return obs


if __name__ == "__main__":
    import pandas as pd
    import json

    filename = "data/synthetic_respA.csv"
    perf_data = pd.read_csv(filename, index_col=0)
    data_diameters = torch.from_numpy(perf_data.index.values)
    perf_data = torch.from_numpy(perf_data.values).T
    
    with open("data/synthetic_summary.json") as f:
        metadata = json.load(f)[filename]
    
    # Define a runnable conditioned model
    with pyro.plate("data", perf_data.shape[0], dim=-2):
        cond_model = pyro.condition(get_profile, data={'obs': perf_data})
    
    # Run two Markov chains in parallel
    num_samples = 400
    num_chains = 2
    kernel = pyro.infer.NUTS(model=cond_model)
    mcmc = pyro.infer.MCMC(kernel, num_samples, warmup_steps=500, num_chains=num_chains)
    mcmc.run(data_diameters)

    # Collect our MCMC run data
    samples_post = mcmc.get_samples()
    q_post_samples = samples_post['q']
    gap = q_post_samples.max() - q_post_samples.min()
    qrange = torch.linspace(0, q_post_samples.max()+.2*gap)

    import matplotlib.pyplot as plt
    from scipy import stats
    
    plt.style.use('ggplot')
    
    q_prior_samples = charge_prior.sample((num_samples,))
    n_bins = int(num_samples ** (1/2))
    
    plt.figure()

    probs = torch.exp(charge_prior.log_prob(qrange))
    true_charge = metadata.get("true_charge", None)
    
    p, = plt.plot(qrange, probs,
             label="Prior distribution")
    prior_color = p.get_color()
    plt.hist(q_prior_samples, bins=n_bins, rwidth=.8, density=True,
             label="Prior samples", color=prior_color, alpha=.8)
    
    # plot the posterior distribution histogram of the charge density
    plt.hist(q_post_samples, bins=n_bins, rwidth=.8, density=True,
             label="Posterior")
    if true_charge is not None:
        plt.vlines(true_charge, *plt.ylim(), ls='--', label="True (point) value")
    plt.legend()
    plt.title("Prior and posterior distributions of the charge density")
    plt.xlabel("Charge density (C/m)")
    plt.tight_layout()
    plt.show()
