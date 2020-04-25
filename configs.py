from physics import LayerParams
from scipy import constants


# dielectric constants
DIELECTRIC_POLYPROPYLENE = 2.1
DIELECTRIC_POLYETHYLENE = 2.25
# permittivity of polypropylene
PERMITTIVITY_POLYPROPYLENE = 2.3
# permittivity of polyethylene
PERMITTIVITY_POLYETHYLENE = 2.25


def respirator_A():
    surface_area = 0.011  # in m2

    # permittivity of polypropylene
    charge_density = 13 * constants.nano

    layer_params = [
        LayerParams(39.49*constants.micro, 0.31*constants.milli,
                    0.165, charge_density, PERMITTIVITY_POLYPROPYLENE, DIELECTRIC_POLYPROPYLENE),
        LayerParams(7.84 * constants.micro, 1.77*constants.milli,
                    0.069, charge_density, PERMITTIVITY_POLYPROPYLENE, DIELECTRIC_POLYPROPYLENE),
        LayerParams(40.88*constants.micro, 1.05*constants.milli,
                    0.200, charge_density, PERMITTIVITY_POLYPROPYLENE, DIELECTRIC_POLYPROPYLENE)
    ]

    return surface_area, layer_params


def respirator_B():
    surface_area = 0.0134  # in m2
    charge_density = 13 * constants.nano

    layer_params = [
        LayerParams(7.19*constants.micro, 0.35*constants.milli,
                    0.091, charge_density, PERMITTIVITY_POLYPROPYLENE, DIELECTRIC_POLYPROPYLENE),
        LayerParams(7.19 * constants.micro, 0.35*constants.milli,
                    0.091, charge_density, PERMITTIVITY_POLYPROPYLENE, DIELECTRIC_POLYPROPYLENE),
        LayerParams(34.25*constants.micro, 0.36*constants.milli,
                    0.108, charge_density, PERMITTIVITY_POLYETHYLENE, DIELECTRIC_POLYETHYLENE)
    ]

    return surface_area, layer_params
