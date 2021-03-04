"""
Collection of Position Co-ordinate Conversion Functions
"""

# global
import ivy as _ivy

MIN_DENOMINATOR = 1e-12


def polar_to_cartesian_coords(polar_coords):
    """
    Convert spherical polar co-ordinates :math:`\mathbf{x}_p = [r, α, β]` to cartesian co-ordinates
    :math:`\mathbf{x}_c = [x, y, z]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    :param polar_coords: Spherical polar co-ordinates *[batch_shape,3]*
    :type polar_coords: array
    :return: Cartesian co-ordinates *[batch_shape,3]*
    """

    # BS x 1
    phi = polar_coords[..., 0:1]
    theta = polar_coords[..., 1:2]
    r = polar_coords[..., 2:3]

    x = r * _ivy.sin(theta) * _ivy.cos(phi)
    y = r * _ivy.sin(theta) * _ivy.sin(phi)
    z = r * _ivy.cos(theta)

    # BS x 3
    return _ivy.concatenate((x, y, z), -1)


def cartesian_to_polar_coords(cartesian_coords):
    """
    Convert cartesian co-ordinates :math:`\mathbf{x}_c = [x, y, z]` to spherical polar co-ordinates
    :math:`\mathbf{x}_p = [r, α, β]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    :param cartesian_coords: Cartesian co-ordinates *[batch_shape,3]*
    :type cartesian_coords: array
    :return: Spherical polar co-ordinates *[batch_shape,3]*
    """

    # BS x 1
    x = cartesian_coords[..., 0:1]
    y = cartesian_coords[..., 1:2]
    z = cartesian_coords[..., 2:3]

    r = (x**2 + y**2 + z**2)**0.5
    phi = _ivy.atan2(y, x)
    theta = _ivy.acos(z / (r + MIN_DENOMINATOR))

    # BS x 3
    return _ivy.concatenate((phi, theta, r), -1)
