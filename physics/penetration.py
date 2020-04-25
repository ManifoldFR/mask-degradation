"""
Compute the penetration response model from Bałazy, A. et al.
(see README for full reference).
"""
import numpy as np
from scipy import constants  # physics constants! :)
from dataclasses import dataclass
from typing import List
import math


def kuwabara_number(alpha):
    """Compute the Kuwabara hydrodynamic factor."""
    return -0.5 * np.log(alpha) - 0.75 + alpha - .25 * alpha ** 2


def peclet_number(d_f, face_velocity, diff_coef):
    """
    diff_coef : particle diffusion coefficient
    d_f : fiber diameter of the layer.
    """
    return face_velocity * d_f / diff_coef

def particle_diffusion_coef(d_p, temp, viscosity, slip_factor):
    """
    d_p : particle size in meters
    temp : fluid absolute temperature
    viscosity : fluid viscosity
    slip_factor : Cunningham slip correction factor
    """
    num_ = constants.Boltzmann * temp * slip_factor
    den_ = 3 * constants.pi * viscosity * d_p
    return num_ / den_


def cunningham_slip_correction(d_p, knudsen):
    """
    d_p : particle diameter in meters
    knudsen : knudsen number
    """
    # knudsen number
    c_c = 1 + knudsen * (1.142 + 0.558 * np.exp(-0.999/knudsen))
    return c_c


def knudsen_number(d_p):
    gas_free_path = 65 * constants.nano  # under normal conditions
    # particle radius = 1/2 diameter (?)
    radius = d_p
    knudsen = gas_free_path / radius
    return knudsen

# Efficiencies

def diffusion_efficiency(d_p, d_f, face_velocity, diff_coef, alpha):
    """
    Compute the particle collection efficiency for the diffusion
    mechanism, according to Payet et al.
    Formula corrected using the original paper instead of the expression
    in Bałazy, A. et al.
    """
    ku = kuwabara_number(alpha)
    pe = peclet_number(d_f, face_velocity, diff_coef)
    pe_23 = pe ** (-2/3)
    al_ku_ratio = np.power((1 - alpha) / ku, 1/3)
    numerator = 1.6 * al_ku_ratio * pe_23
    denominator = 1. + numerator
    coef = numerator / denominator
    return coef        
    

def interception_efficiency(d_p, d_f, alpha):
    """
    Single-fiber efficiency for the interception mechanism.
    
    d_p : particle diameter
    d_f : fiber diameter
    k_n : Knudsen number
    n_r : interception parameter
    """
    ku = kuwabara_number(alpha)
    # interception parameter
    n_r = d_p / d_f
    k_n = knudsen_number(d_p)
    efficiency = (0.6 * (1 - alpha) / ku * 
                  n_r ** 2 / (1 + n_r) *
                  (1 + 1.9996 * k_n))
    return efficiency


def polar_efficiency(d_p, d_f, charge, permittivity, slip_factor, face_velocity, viscosity):
    """Capture efficiency factor from polarization.
    See Formula (11) in the paper.""" 
    num = slip_factor * charge ** 2 * d_p ** 2
    den = 3 * constants.pi ** 2 * constants.epsilon_0 * viscosity * d_f ** 3 * face_velocity
    nq_0 = num / den * (permittivity - 1) / (permittivity + 2)
    eff = 0.06 * np.power(nq_0, 2/5)
    return eff


def polar_efficiency2(d_p, d_f, alpha, charge, permittivity, dielectric, slip_factor, face_velocity, viscosity):
    """Capture efficiency factor from polarization.
    See Formula (13) in the paper."""
    B = 0.21
    ku = kuwabara_number(alpha)
    c1 = ((1 - alpha) / ku) ** 2/5
    from configs import DIELECTRIC_POLYPROPYLENE
    num = slip_factor * charge ** 2 * d_p ** 2
    den = (1 + dielectric) ** 2 * 3 * constants.pi * \
        constants.epsilon_0 * viscosity * d_f ** 3 * face_velocity
    nq_0 = (num / den * ((permittivity - 1) / (permittivity + 2))
            )
    eff = c1 * constants.pi * nq_0 / (1 + 2*constants.pi*nq_0 ** (2/3))
    return eff

# Penetration

def layer_penetration(d_p, d_f, thickness, alpha, face_velocity, temp, viscosity,
                      charge=None, permittivity=None, dielectric=None):
    """
    Compute log-layer penetration.
    
    d_f : fiber density
    thickness : layer thickness parameter (L in paper)
    face_velocity : U_0 in paper
    alpha : packing density of the material
    temp : fluid absolute temperature (in Kelvin)
    viscosity : fluid viscosity
    """
    k_n = knudsen_number(d_p)
    
    slip_factor = cunningham_slip_correction(d_p, k_n)
    diff_coef = particle_diffusion_coef(d_p, temp, viscosity, slip_factor)

    diff_eff = diffusion_efficiency(d_p, d_f, face_velocity,
                                    diff_coef, alpha)
    inter_eff = interception_efficiency(d_p, d_f, alpha)
    

    # Single-fiber collection efficiency for the layer.
    if charge is not None:
        permittivity = permittivity or 1.
        # q_eff = polar_efficiency(
        #     d_p, d_f, charge, permittivity, slip_factor, face_velocity, viscosity)
        q_eff = polar_efficiency2(
            d_p, d_f, alpha, charge, permittivity, dielectric, slip_factor, face_velocity, viscosity)
        fiber_coll_eff = 1 - (1 - diff_eff) * (1 - inter_eff) * (1 - q_eff)
    else:
        fiber_coll_eff = 1 - (1 - diff_eff) * (1 - inter_eff)

    log_penet = (-4 * alpha * fiber_coll_eff * thickness /
                 (constants.pi * d_f * (1 - alpha)))
    return log_penet

@dataclass
class LayerParams:
    d_f: float
    thickness: float
    alpha: float
    charge_density: float = None
    permittivity: float = None
    dielectric: float = None


def compute_penetration_profile(d_p, layer_params: List[LayerParams], face_velocity, temp, viscosity):
    res_ = 0.
    for param in layer_params:
        layer_penet_ = layer_penetration(d_p, param.d_f, param.thickness, param.alpha,
                                         face_velocity, temp, viscosity,
                                         charge=param.charge_density, permittivity=param.permittivity, dielectric=param.dielectric)
        res_ += layer_penet_
    if isinstance(res_, np.ndarray):
        return np.exp(res_)
    else:
        # assume torch.Tensor
        return res_.exp()
