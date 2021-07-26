"""
Collection of Rotation Conversion Functions to Rotation Matrix Format
"""

# global
import ivy as _ivy

# local
from ivy_mech.orientation import quaternion as _ivy_quat


# Private Helpers #
# ----------------#

def _x_axis_rotation_matrix(identity_matrix, zeros, sin_theta, cos_theta):
    rot_x_u = identity_matrix[..., 0:1, :]
    rot_x_m = _ivy.concatenate((zeros, cos_theta, -sin_theta), -1)
    rot_x_l = _ivy.concatenate((zeros, sin_theta, cos_theta), -1)
    return _ivy.concatenate((rot_x_u, rot_x_m, rot_x_l), -2)


def _y_axis_rotation_matrix(identity_matrix, zeros, sin_theta, cos_theta):
    rot_y_u = _ivy.concatenate((cos_theta, zeros, sin_theta), -1)
    rot_y_m = identity_matrix[..., 1:2, :]
    rot_y_l = _ivy.concatenate((-sin_theta, zeros, cos_theta), -1)
    return _ivy.concatenate((rot_y_u, rot_y_m, rot_y_l), -2)


def _z_axis_rotation_matrix(identity_matrix, zeros, sin_theta, cos_theta):
    rot_z_u = _ivy.concatenate((cos_theta, -sin_theta, zeros), -1)
    rot_z_m = _ivy.concatenate((sin_theta, cos_theta, zeros), -1)
    rot_z_l = identity_matrix[..., 2:3, :]
    return _ivy.concatenate((rot_z_u, rot_z_m, rot_z_l), -2)


ROTATION_FUNC_DICT = {'x': _x_axis_rotation_matrix, 'y': _y_axis_rotation_matrix, 'z': _z_axis_rotation_matrix}


# General #
# --------#

def rot_vec_to_rot_mat(rot_vec):
    """
    Convert rotation vector :math:`\mathbf{θ}_{rv} = [θe_x, θe_y, θe_z]` to rotation matrix
    :math:`\mathbf{R}\in\mathbb{R}^{3×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle>`_

    :param rot_vec: Rotation vector *[batch_shape,3]*
    :type rot_vec: array
    :return: Rotation matrix *[batch_shape,3,3]*
    """

    # BS x 1
    t = _ivy.reductions.reduce_sum(rot_vec**2, -1, keepdims=True)**0.5

    # BS x 3
    u = rot_vec / t

    # BS x 1 x 1
    cost = _ivy.expand_dims(_ivy.math.cos(t), -1)
    sint = _ivy.expand_dims(_ivy.math.sin(t), -1)
    ux = _ivy.expand_dims(u[..., 0:1], -1)
    uy = _ivy.expand_dims(u[..., 1:2], -1)
    uz = _ivy.expand_dims(u[..., 2:3], -1)

    om_cost = 1 - cost
    ux_sint = ux * sint
    uy_sint = uy * sint
    uz_sint = uz * sint

    ux_uy_om_cost = ux * uy * om_cost
    ux_uz_om_cost = ux * uz * om_cost
    uy_uz_om_cost = uy * uz * om_cost

    top_left = cost + ux**2 * om_cost
    top_middle = ux_uy_om_cost - uz_sint
    top_right = ux_uz_om_cost + uy_sint
    middle_left = ux_uy_om_cost + uz_sint
    middle_middle = cost + uy**2 * om_cost
    middle_right = uy_uz_om_cost - ux_sint
    bottom_left = ux_uz_om_cost - uy_sint
    bottom_middle = uy_uz_om_cost + ux_sint
    bottom_right = cost + uz**2 * om_cost

    # BS x 1 x 3
    top_row = _ivy.concatenate((top_left, top_middle, top_right), -1)
    middle_row = _ivy.concatenate((middle_left, middle_middle, middle_right), -1)
    bottom_row = _ivy.concatenate((bottom_left, bottom_middle, bottom_right), -1)

    # BS x 3 x 3
    return _ivy.concatenate((top_row, middle_row, bottom_row), -2)


def quaternion_to_rot_mat(quaternion):
    """
    Convert quaternion :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]` to rotation matrix
    :math:`\mathbf{R}\in\mathbb{R}^{3×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Conversion_to_and_from_the_matrix_representation>`_

    :param quaternion: Quaternion *[batch_shape,4]*
    :type quaternion: array
    :return: Rotation matrix *[batch_shape,3,3]*
    """

    # BS x 1 x 1
    a = _ivy.expand_dims(quaternion[..., 3:4], -1)
    b = _ivy.expand_dims(quaternion[..., 0:1], -1)
    c = _ivy.expand_dims(quaternion[..., 1:2], -1)
    d = _ivy.expand_dims(quaternion[..., 2:3], -1)

    # BS x 1 x 1
    top_left = a**2 + b**2 - c**2 - d**2
    top_middle = 2*b*c - 2*a*d
    top_right = 2*b*d + 2*a*c
    middle_left = 2*b*c + 2*a*d
    middle_middle = a**2 - b**2 + c**2 - d**2
    middle_right = 2*c*d - 2*a*b
    bottom_left = 2*b*d - 2*a*c
    bottom_middle = 2*c*d + 2*a*b
    bottom_right = a**2 - b**2 - c**2 + d**2

    # BS x 1 x 3
    top_row = _ivy.concatenate((top_left, top_middle, top_right), -1)
    middle_row = _ivy.concatenate((middle_left, middle_middle, middle_right), -1)
    bottom_row = _ivy.concatenate((bottom_left, bottom_middle, bottom_right), -1)

    # BS x 3 x 3
    return _ivy.concatenate((top_row, middle_row, bottom_row), -2)


# Euler Conversions #
# ------------------#

def euler_to_rot_mat(euler_angles, convention='zyx', batch_shape=None, dev_str=None):
    """
    Convert :math:`zyx` Euler angles :math:`\mathbf{θ}_{abc} = [ϕ_a, ϕ_b, ϕ_c]` to rotation matrix
    :math:`\mathbf{R}\in\mathbb{R}^{3×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix>`_

    :param euler_angles: Euler angles *[batch_shape,3]*
    :type euler_angles: array
    :param convention: The axes for euler rotation, in order of L.H.S. matrix multiplication.
    :type convention: str, optional
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :return: Rotation matrix *[batch_shape,3,3]*
    """

    if batch_shape is None:
        batch_shape = euler_angles.shape[:-1]
        
    if dev_str is None:
        dev_str = _ivy.dev_str(euler_angles)

    # BS x 1 x 1
    zeros = _ivy.zeros(list(batch_shape) + [1, 1], dev_str=dev_str)

    alpha = _ivy.expand_dims(euler_angles[..., 0:1], -1)
    beta = _ivy.expand_dims(euler_angles[..., 1:2], -1)
    gamma = _ivy.expand_dims(euler_angles[..., 2:3], -1)

    cos_alpha = _ivy.cos(alpha)
    sin_alpha = _ivy.sin(alpha)

    cos_beta = _ivy.cos(beta)
    sin_beta = _ivy.sin(beta)

    cos_gamma = _ivy.cos(gamma)
    sin_gamma = _ivy.sin(gamma)

    # BS x 3 x 3
    identity_matrix = _ivy.identity(3, batch_shape=batch_shape)

    # BS x 3 x 3
    rot_alpha = ROTATION_FUNC_DICT[convention[0]](identity_matrix, zeros, sin_alpha, cos_alpha)
    rot_beta = ROTATION_FUNC_DICT[convention[1]](identity_matrix, zeros, sin_beta, cos_beta)
    rot_gamma = ROTATION_FUNC_DICT[convention[2]](identity_matrix, zeros, sin_gamma, cos_gamma)
    return _ivy.matmul(rot_gamma, _ivy.matmul(rot_beta, rot_alpha))


# noinspection PyUnresolvedReferences
def target_facing_rotation_matrix(body_pos, target_pos, batch_shape=None, dev_str=None):
    """
    Determine rotation matrix :math:`\mathbf{R}\in\mathbb{R}^{3×3}` of body which corresponds with
    it's positive z axis facing towards a target world co-ordinate :math:`\mathbf{x}_t = [t_x, t_y, t_z]`, given the body
    world co-ordinate :math:`\mathbf{x}_b = [b_x, b_y, b_z]`, while assuming world-space positve-z to correspond to up direction.
    This is particularly useful for orienting cameras.
    `[reference] <https://stackoverflow.com/questions/21830340/understanding-glmlookat>`_

    :param body_pos: Cartesian position of body *[batch_shape,3]*
    :type body_pos: array
    :param target_pos: Cartesian position of target *[batch_shape,3]*
    :type target_pos: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :return: Rotation vector, which faces body towards target *[batch_shape,4]*
    """

    # ToDo: make this more general, allowing arbitrary view and up direction axes to be specified

    if batch_shape is None:
        batch_shape = body_pos.shape[:-1]
    batch_shape = list(batch_shape)
    num_batch_dims = len(batch_shape)

    if dev_str is None:
        dev_str = _ivy.dev_str(body_pos)

    # BS x 3
    up = _ivy.tile(_ivy.reshape(_ivy.array([0., 0., 1.], dev_str=dev_str),
                                [1] * num_batch_dims + [3]), batch_shape + [1])

    z = target_pos - body_pos
    z = z / _ivy.reduce_sum(z ** 2, -1, keepdims=True) ** 0.5

    x = _ivy.cross(up, z)

    y = _ivy.cross(z, x)

    x = x / _ivy.reduce_sum(x ** 2, -1, keepdims=True) ** 0.5
    y = y / _ivy.reduce_sum(y ** 2, -1, keepdims=True) ** 0.5

    # BS x 1 x 3
    x = _ivy.expand_dims(x, -2)
    y = _ivy.expand_dims(y, -2)
    z = _ivy.expand_dims(z, -2)

    # BS x 3 x 3
    return _ivy.pinv(_ivy.concatenate((x, y, z), -2))


def axis_angle_to_rot_mat(axis_angle):
    """
    Convert rotation axis unit vector :math:`\mathbf{e} = [e_x, e_y, e_z]` and
    rotation angle :math:`θ` to rotation matrix :math:`\mathbf{R}\in\mathbb{R}^{3×3}`.

    :param axis_angle: Axis-angle *[batch_shape,4]*
    :type axis_angle: array
    :return: Rotation matrix *[batch_shape,3,3]*
    """

    return quaternion_to_rot_mat(_ivy_quat.axis_angle_to_quaternion(axis_angle))


def get_random_rot_mat(batch_shape=None):
    """
    Generate random rotation matrix :math:`\mathbf{R}\in\mathbb{R}^{3×3}`.
    :param batch_shape: Shape of batch. Shape of [1] is assumed if None.
    :type batch_shape: sequence of ints, optional
    :return: Random rotation matrix *[batch_shape,3,3]*
    """

    return quaternion_to_rot_mat(
        _ivy_quat.get_random_quaternion(batch_shape=batch_shape))
