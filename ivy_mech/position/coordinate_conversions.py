"""
Collection of Position Co-ordinate Conversion Functions
"""

# global
from ivy.framework_handler import get_framework as _get_framework

MIN_DENOMINATOR = 1e-12


def polar_to_cartesian_coords(polar_coords, f=None):
    """
    Convert spherical polar co-ordinates :math:`\mathbf{x}_p = [r, α, β]` to cartesian co-ordinates
    :math:`\mathbf{x}_c = [x, y, z]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    :param polar_coords: Spherical polar co-ordinates *[batch_shape,3]*
    :type polar_coords: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Cartesian co-ordinates *[batch_shape,3]*
    """

    f = _get_framework(polar_coords, f=f)

    # BS x 1
    phi = polar_coords[..., 0:1]
    theta = polar_coords[..., 1:2]
    r = polar_coords[..., 2:3]

    x = r * f.sin(theta) * f.cos(phi)
    y = r * f.sin(theta) * f.sin(phi)
    z = r * f.cos(theta)

    # BS x 3
    return f.concatenate((x, y, z), -1)


def cartesian_to_polar_coords(cartesian_coords, f=None):
    """
    Convert cartesian co-ordinates :math:`\mathbf{x}_c = [x, y, z]` to spherical polar co-ordinates
    :math:`\mathbf{x}_p = [r, α, β]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    :param cartesian_coords: Cartesian co-ordinates *[batch_shape,3]*
    :type cartesian_coords: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Spherical polar co-ordinates *[batch_shape,3]*
    """

    f = _get_framework(cartesian_coords, f=f)

    # BS x 1
    x = cartesian_coords[..., 0:1]
    y = cartesian_coords[..., 1:2]
    z = cartesian_coords[..., 2:3]

    r = (x**2 + y**2 + z**2)**0.5
    phi = f.atan2(y, x)
    theta = f.acos(z / (r + MIN_DENOMINATOR))

    # BS x 3
    return f.concatenate((phi, theta, r), -1)
