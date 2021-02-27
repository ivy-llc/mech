"""
Collection of Rotation Conversion Functions to Axis-Angle Format
"""

# global
import ivy
from ivy.framework_handler import get_framework as _get_framework

# local
from ivy_mech.orientation import quaternion as _ivy_q

MIN_DENOMINATOR = 1e-12


def rot_mat_to_axis_angle(rot_mat,  dev=None, f=None):
    """
    Convert rotation matrix :math:`\mathbf{R}\in\mathbb{R}^{3×3}` to rotation axis unit vector
    :math:`\mathbf{e} = [e_x, e_y, e_z]` and rotation angle :math:`θ`.

    :param rot_mat: Rotation matrix *[batch_shape,3,3]*
    :type rot_mat: array
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Rotation axis unit vector and angle *[batch_shape,4]*
    """
    f = _get_framework(rot_mat, f=f)
    quat = _ivy_q.rot_mat_to_quaternion(rot_mat, f=f)
    return quaternion_to_axis_angle(quat, dev, f=f)


def euler_to_axis_angle(euler_angles, convention='zyx', batch_shape=None, dev=None, f=None):
    """
    Convert :math:`zyx` Euler angles :math:`\mathbf{θ}_{abc} = [ϕ_a, ϕ_b, ϕ_c]` to rotation axis unit vector
    :math:`\mathbf{e} = [e_x, e_y, e_z]` and rotation angle :math:`θ`.

    :param euler_angles: Input euler angles *[batch_shape,3]*
    :type euler_angles: array
    :param convention: The axes for euler rotation, in order of L.H.S. matrix multiplication.
    :type convention: str, optional
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Rotation axis unit vector and angle *[batch_shape,4]*
    """
    f = _get_framework(euler_angles, f=f)
    quat = _ivy_q.euler_to_quaternion(euler_angles, convention, batch_shape, f=f)
    return quaternion_to_axis_angle(quat, dev, f=f)


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
        dev = ivy.get_device(quaternion, f=f)

    # BS x 1
    e1 = quaternion[..., 0:1]
    e2 = quaternion[..., 1:2]
    e3 = quaternion[..., 2:3]
    n = quaternion[..., 3:4]

    # BS x 1
    theta = 2 * ivy.acos(ivy.clip(n, 0, 1, f=f), f=f)
    vector_x = ivy.where(theta != 0, e1 / (ivy.sin(theta / 2, f=f) + MIN_DENOMINATOR),
                          ivy.zeros_like(theta, dev=dev, f=f), f=f)
    vector_y = ivy.where(theta != 0, e2 / (ivy.sin(theta / 2, f=f) + MIN_DENOMINATOR),
                          ivy.zeros_like(theta, dev=dev, f=f), f=f)
    vector_z = ivy.where(theta != 0, e3 / (ivy.sin(theta / 2, f=f) + MIN_DENOMINATOR),
                          ivy.zeros_like(theta, dev=dev, f=f), f=f)

    # BS x 4
    return ivy.concatenate((vector_x, vector_y, vector_z, theta), -1, f=f)


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
        dev = ivy.get_device(quaternion, f=f)

    # BS x 4
    vector_and_angle = quaternion_to_axis_angle(quaternion, dev)

    # BS x 1
    theta = ivy.acos(vector_and_angle[..., 2:3], f=f)
    phi = ivy.atan2(vector_and_angle[..., 1:2], vector_and_angle[..., 0:1], f=f)

    # BS x 3
    return ivy.concatenate((theta, phi, vector_and_angle[..., -1:]), -1, f=f)


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
        dev = ivy.get_device(quaternion, f=f)

    # BS x 4
    vector_and_angle = quaternion_to_axis_angle(quaternion, dev)

    # BS x 3
    return vector_and_angle[..., :-1] * vector_and_angle[..., -1:]


def get_random_axis_angle(f, batch_shape=None):
    """
    Generate random axis unit vector :math:`\mathbf{e} = [e_x, e_y, e_z]`
    and rotation angle :math:`θ`
    :param f: Machine learning framework.
    :type f: ml_framework
    :param batch_shape: Shape of batch. Shape of [1] is assumed if None.
    :type batch_shape: sequence of ints, optional
    :return: Random rotation axis unit vector and angle *[batch_shape,4]*
    """

    if f is None:
        raise Exception('framework f must be specified for calling ivy.get_random_euler()')

    return quaternion_to_axis_angle(
        _ivy_q.get_random_quaternion(f, batch_shape=batch_shape))
