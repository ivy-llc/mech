"""Collection of Rotation Conversion Functions to Rotation Matrix Format"""
# global
import ivy

# local
from ivy_mech.orientation import quaternion as ivy_quat


# Private Helpers #
# ----------------#


def _x_axis_rotation_matrix(identity_matrix, zeros, sin_theta, cos_theta):
    """
    Parameters
    ----------
    identity_matrix

    zeros

    sin_theta

    cos_theta


    """
    rot_x_u = identity_matrix[..., 0:1, :]
    rot_x_m = ivy.concat([zeros, cos_theta, -sin_theta], axis=-1)
    rot_x_l = ivy.concat([zeros, sin_theta, cos_theta], axis=-1)
    return ivy.concat([rot_x_u, rot_x_m, rot_x_l], axis=-2)


def _y_axis_rotation_matrix(identity_matrix, zeros, sin_theta, cos_theta):
    """
    Parameters
    ----------
    identity_matrix

    zeros

    sin_theta

    cos_theta


    """
    rot_y_u = ivy.concat([cos_theta, zeros, sin_theta], axis=-1)
    rot_y_m = identity_matrix[..., 1:2, :]
    rot_y_l = ivy.concat([-sin_theta, zeros, cos_theta], axis=-1)
    return ivy.concat([rot_y_u, rot_y_m, rot_y_l], axis=-2)


def _z_axis_rotation_matrix(identity_matrix, zeros, sin_theta, cos_theta):
    """
    Parameters
    ----------
    identity_matrix

    zeros

    sin_theta

    cos_theta


    """
    rot_z_u = ivy.concat([cos_theta, -sin_theta, zeros], axis=-1)
    rot_z_m = ivy.concat([sin_theta, cos_theta, zeros], axis=-1)
    rot_z_l = identity_matrix[..., 2:3, :]
    return ivy.concat([rot_z_u, rot_z_m, rot_z_l], axis=-2)


ROTATION_FUNC_DICT = {
    "x": _x_axis_rotation_matrix,
    "y": _y_axis_rotation_matrix,
    "z": _z_axis_rotation_matrix,
}


# General #
# --------#


def rot_vec_to_rot_mat(rot_vec):
    r"""Convert rotation vector :math:`\mathbf{θ}_{rv} = [θe_x, θe_y, θe_z]` to
    rotation matrix :math:`\mathbf{R}\in\mathbb{R}^{3×3}`.\n `[reference]
    <https://en.wikipedia.org/wiki/Rotation_matrix
    #Rotation_matrix_from_axis_and_angle>`_

    Parameters
    ----------
    rot_vec
        Rotation vector *[batch_shape,3]*

    Returns
    -------
    ret
        Rotation matrix *[batch_shape,3,3]*

    """
    # BS x 1
    t = ivy.sum(rot_vec**2, axis=-1, keepdims=True) ** 0.5

    # BS x 3
    u = rot_vec / t

    # BS x 1 x 1
    cost = ivy.expand_dims(ivy.cos(t), axis=-1)
    sint = ivy.expand_dims(ivy.sin(t), axis=-1)
    ux = ivy.expand_dims(u[..., 0:1], axis=-1)
    uy = ivy.expand_dims(u[..., 1:2], axis=-1)
    uz = ivy.expand_dims(u[..., 2:3], axis=-1)

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
    top_row = ivy.concat([top_left, top_middle, top_right], axis=-1)
    middle_row = ivy.concat([middle_left, middle_middle, middle_right], axis=-1)
    bottom_row = ivy.concat([bottom_left, bottom_middle, bottom_right], axis=-1)

    # BS x 3 x 3
    return ivy.concat([top_row, middle_row, bottom_row], axis=-2)


def quaternion_to_rot_mat(quaternion):
    r"""Convert quaternion :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]` to rotation
    matrix :math:`\mathbf{R}\in\mathbb{R}^{3×3}`.\n `[reference]
    <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    #Conversion_to_and_from_the_matrix_representation>`_

    Parameters
    ----------
    quaternion
        Quaternion *[batch_shape,4]*

    Returns
    -------
    ret
        Rotation matrix *[batch_shape,3,3]*

    """
    # BS x 1 x 1
    a = ivy.expand_dims(quaternion[..., 3:4], axis=-1)
    b = ivy.expand_dims(quaternion[..., 0:1], axis=-1)
    c = ivy.expand_dims(quaternion[..., 1:2], axis=-1)
    d = ivy.expand_dims(quaternion[..., 2:3], axis=-1)

    # BS x 1 x 1
    top_left = a**2 + b**2 - c**2 - d**2
    top_middle = 2 * b * c - 2 * a * d
    top_right = 2 * b * d + 2 * a * c
    middle_left = 2 * b * c + 2 * a * d
    middle_middle = a**2 - b**2 + c**2 - d**2
    middle_right = 2 * c * d - 2 * a * b
    bottom_left = 2 * b * d - 2 * a * c
    bottom_middle = 2 * c * d + 2 * a * b
    bottom_right = a**2 - b**2 - c**2 + d**2

    # BS x 1 x 3
    top_row = ivy.concat([top_left, top_middle, top_right], axis=-1)
    middle_row = ivy.concat([middle_left, middle_middle, middle_right], axis=-1)
    bottom_row = ivy.concat([bottom_left, bottom_middle, bottom_right], axis=-1)

    # BS x 3 x 3
    return ivy.concat([top_row, middle_row, bottom_row], axis=-2)


# Euler Conversions #
# ------------------#


def euler_to_rot_mat(euler_angles, convention="zyx", batch_shape=None, device=None):
    r"""Convert :math:`zyx` Euler angles :math:`\mathbf{θ}_{abc} = [ϕ_a, ϕ_b, ϕ_c]` to
    rotation matrix
    :math:`\mathbf{R}\in\mathbb{R}^{3×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix>`_

    Parameters
    ----------
    euler_angles
        Euler angles *[batch_shape,3]*
    convention
        The axes for euler rotation, in order of L.H.S. matrix multiplication.
        (Default value = 'zyx')
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Rotation matrix *[batch_shape,3,3]*

    """
    if batch_shape is None:
        batch_shape = euler_angles.shape[:-1]

    if device is None:
        device = ivy.dev(euler_angles)

    # BS x 1 x 1
    zeros = ivy.zeros(list(batch_shape) + [1, 1], device=device)

    alpha = ivy.expand_dims(euler_angles[..., 0:1], axis=-1)
    beta = ivy.expand_dims(euler_angles[..., 1:2], axis=-1)
    gamma = ivy.expand_dims(euler_angles[..., 2:3], axis=-1)

    cos_alpha = ivy.cos(alpha)
    sin_alpha = ivy.sin(alpha)

    cos_beta = ivy.cos(beta)
    sin_beta = ivy.sin(beta)

    cos_gamma = ivy.cos(gamma)
    sin_gamma = ivy.sin(gamma)

    # BS x 3 x 3
    identity_matrix = ivy.eye(3, 3, batch_shape=batch_shape, device=device)

    # BS x 3 x 3
    rot_alpha = ROTATION_FUNC_DICT[convention[0]](
        identity_matrix, zeros, sin_alpha, cos_alpha
    )
    rot_beta = ROTATION_FUNC_DICT[convention[1]](
        identity_matrix, zeros, sin_beta, cos_beta
    )
    rot_gamma = ROTATION_FUNC_DICT[convention[2]](
        identity_matrix, zeros, sin_gamma, cos_gamma
    )
    return ivy.matmul(rot_gamma, ivy.matmul(rot_beta, rot_alpha))


# noinspection PyUnresolvedReferences
def target_facing_rotation_matrix(body_pos, target_pos, batch_shape=None, device=None):
    r"""Determine rotation matrix :math:`\mathbf{R}\in\mathbb{R}^{3×3}` of body which
    corresponds with it's positive z axis facing towards a target world co-ordinate
    :math:`\mathbf{x}_t = [t_x, t_y, t_z]`, given the body world co-ordinate
    :math:`\mathbf{x}_b = [b_x, b_y, b_z]`, while assuming world-space positve-z to
    correspond to up direction. This is particularly useful for orienting cameras. `[
    reference] <https://stackoverflow.com/questions/21830340/understanding-glmlookat>`_

    Parameters
    ----------
    body_pos
        Cartesian position of body *[batch_shape,3]*
    target_pos
        Cartesian position of target *[batch_shape,3]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Rotation vector, which faces body towards target *[batch_shape,4]*

    """
    # ToDo: make this more general, allowing arbitrary view and up direction axes to
    #  be specified

    if batch_shape is None:
        batch_shape = body_pos.shape[:-1]
    batch_shape = list(batch_shape)
    num_batch_dims = len(batch_shape)

    if device is None:
        device = ivy.dev(body_pos)

    # BS x 3
    up = ivy.tile(
        ivy.reshape(
            ivy.array([0.0, 0.0, 1.0], device=device), [1] * num_batch_dims + [3]
        ),
        batch_shape + [1],
    )

    z = target_pos - body_pos
    z = z / ivy.sum(z**2, axis=-1, keepdims=True) ** 0.5

    x = ivy.cross(up, z)

    y = ivy.cross(z, x)

    x = x / ivy.sum(x**2, axis=-1, keepdims=True) ** 0.5
    y = y / ivy.sum(y**2, axis=-1, keepdims=True) ** 0.5

    # BS x 1 x 3
    x = ivy.expand_dims(x, axis=-2)
    y = ivy.expand_dims(y, axis=-2)
    z = ivy.expand_dims(z, axis=-2)

    # BS x 3 x 3
    return ivy.pinv(ivy.concat([x, y, z], axis=-2))


def axis_angle_to_rot_mat(axis_angle):
    r"""Convert rotation axis unit vector :math:`\mathbf{e} = [e_x, e_y, e_z]` and
    rotation angle :math:`θ` to rotation matrix :math:`\mathbf{R}\in\mathbb{R}^{3×3}`.

    Parameters
    ----------
    axis_angle
        Axis-angle *[batch_shape,4]*

    Returns
    -------
    ret
        Rotation matrix *[batch_shape,3,3]*

    """
    return quaternion_to_rot_mat(ivy_quat.axis_angle_to_quaternion(axis_angle))


def get_random_rot_mat(batch_shape=None):
    r"""Generate random rotation matrix :math:`\mathbf{R}\in\mathbb{R}^{3×3}`.

    Parameters
    ----------
    batch_shape
        Shape of batch. Shape of [1] is assumed if None. (Default value = None)

    Returns
    -------
    ret
        Random rotation matrix *[batch_shape,3,3]*

    """
    return quaternion_to_rot_mat(
        ivy_quat.get_random_quaternion(batch_shape=batch_shape)
    )
