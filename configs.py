from physics import LayerParams
from scipy import constants


def respirator_A():
    surface_area = 0.011  # in m2

    # permittivity of polypropylene
    permittivity = 2.3
    charge_density = 13 * constants.nano

    layer_params = [
        LayerParams(39.49*constants.micro, 0.31*constants.milli,
                    0.165, charge_density, permittivity),
        LayerParams(7.84 * constants.micro, 1.77*constants.milli,
                    0.069, charge_density, permittivity),
        LayerParams(40.88*constants.micro, 1.05*constants.milli,
                    0.200, charge_density, permittivity)
    ]

    return surface_area, layer_params

