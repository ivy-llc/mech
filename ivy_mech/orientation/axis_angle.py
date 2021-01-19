"""
Collection of Rotation Conversion Functions to Axis-Angle Format
"""

# global
from ivy.framework_handler import get_framework as _get_framework

# local
from ivy_mech.orientation import quaternion as _ivy_q

MIN_DENOMINATOR = 1e-12


def quaternion_to_axis_angle(quaternion, dev=None, f=None):
    """
    Convert quaternion :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]` to rotation axis unit vector
    :math:`\mathbf{e} = [e_x, e_y, e_z]` and rotation angle :math:`θ`.\n
    `[reference] <https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Quaternions>`_

    :param quaternion: Input quaternion *[batch_shape,4]*
    :type quaternion: array
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Rotation axis unit vector and angle *[batch_shape,4]*
    """

    f = _get_framework(quaternion, f=f)

    if dev is None:
        dev = f.get_device(quaternion)

    # BS x 1
    e1 = quaternion[..., 0:1]
    e2 = quaternion[..., 1:2]
    e3 = quaternion[..., 2:3]
    n = quaternion[..., 3:4]

    # BS x 1
    theta = 2 * f.acos(f.clip(n, 0, 1))
    vector_x = f.where(theta != 0, e1 / (f.sin(theta / 2) + MIN_DENOMINATOR),
                          f.zeros_like(theta, dev=dev))
    vector_y = f.where(theta != 0, e2 / (f.sin(theta / 2) + MIN_DENOMINATOR),
                          f.zeros_like(theta, dev=dev))
    vector_z = f.where(theta != 0, e3 / (f.sin(theta / 2) + MIN_DENOMINATOR),
                          f.zeros_like(theta, dev=dev))

    # BS x 4
    return f.concatenate((vector_x, vector_y, vector_z, theta), -1)


def quaternion_to_polar_axis_angle(quaternion, dev=None, f=None):
    """Convert quaternion :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]` to a polar axis angle representation, which
    constitutes the elevation and azimuth angles of the axis, as well as the rotation angle
    :math:`\mathbf{θ}_{paa} = [ϕ_e, ϕ_a, θ]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation>`_

    :param quaternion: Input quaternion *[batch_shape,4]*
    :type quaternion: array
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Polar axis angle representation *[batch_shape,3]*
    """

    f = _get_framework(quaternion, f=f)

    if dev is None:
        dev = f.get_device(quaternion)

    # BS x 4
    vector_and_angle = quaternion_to_axis_angle(quaternion, dev)

    # BS x 1
    theta = f.acos(vector_and_angle[..., 2:3])
    phi = f.atan2(vector_and_angle[..., 1:2], vector_and_angle[..., 0:1])

    # BS x 3
    return f.concatenate((theta, phi, vector_and_angle[..., -1:]), -1)


# noinspection PyUnusedLocal
def quaternion_to_rotation_vector(quaternion, dev=None, f=None):
    """Convert quaternion :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]` to rotation vector
    :math:`\mathbf{θ}_{rv} = [θe_x, θe_y, θe_z]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_axis_and_angle_(rotation_vector)>`_

    :param quaternion: Input quaternion *[batch_shape,4]*
    :type quaternion: array
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Rotation vector *[batch_shape,3]*
    """

    if dev is None:
        dev = f.get_device(quaternion)

    # BS x 4
    vector_and_angle = quaternion_to_axis_angle(quaternion, dev)

    # BS x 3
    return vector_and_angle[..., :-1] * vector_and_angle[..., -1:]
