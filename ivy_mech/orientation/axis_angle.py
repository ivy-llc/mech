"""Collection of Rotation Conversion Functions to Axis-Angle Format"""

# global
import ivy as _ivy

# local
from ivy_mech.orientation import quaternion as _ivy_q

MIN_DENOMINATOR = 1e-12


def rot_mat_to_axis_angle(rot_mat,  device=None):
    """Convert rotation matrix :math:`\mathbf{R}\in\mathbb{R}^{3×3}` to rotation axis unit vector
    :math:`\mathbf{e} = [e_x, e_y, e_z]` and rotation angle :math:`θ`.

    Parameters
    ----------
    rot_mat
        Rotation matrix *[batch_shape,3,3]*
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Rotation axis unit vector and angle *[batch_shape,4]*

    """
    quat = _ivy_q.rot_mat_to_quaternion(rot_mat)
    return quaternion_to_axis_angle(quat, device)


def euler_to_axis_angle(euler_angles, convention='zyx', batch_shape=None, device=None):
    """Convert :math:`zyx` Euler angles :math:`\mathbf{θ}_{abc} = [ϕ_a, ϕ_b, ϕ_c]` to rotation axis unit vector
    :math:`\mathbf{e} = [e_x, e_y, e_z]` and rotation angle :math:`θ`.

    Parameters
    ----------
    euler_angles
        Input euler angles *[batch_shape,3]*
    convention
        The axes for euler rotation, in order of L.H.S. matrix multiplication. (Default value = 'zyx')
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Rotation axis unit vector and angle *[batch_shape,4]*

    """
    quat = _ivy_q.euler_to_quaternion(euler_angles, convention, batch_shape)
    return quaternion_to_axis_angle(quat, device)


def quaternion_to_axis_angle(quaternion, device=None):
    """Convert quaternion :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]` to rotation axis unit vector
    :math:`\mathbf{e} = [e_x, e_y, e_z]` and rotation angle :math:`θ`.\n
    `[reference] <https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Quaternions>`_

    Parameters
    ----------
    quaternion
        Input quaternion *[batch_shape,4]*
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Rotation axis unit vector and angle *[batch_shape,4]*

    """

    # BS x 1
    e1 = quaternion[..., 0:1]
    e2 = quaternion[..., 1:2]
    e3 = quaternion[..., 2:3]
    n = quaternion[..., 3:4]

    # BS x 1
    theta = 2 * _ivy.acos(_ivy.clip(n, 0, 1))
    vector_x = _ivy.where(theta != 0, e1 / (_ivy.sin(theta / 2) + MIN_DENOMINATOR),
                          _ivy.zeros_like(theta, device=device))
    vector_y = _ivy.where(theta != 0, e2 / (_ivy.sin(theta / 2) + MIN_DENOMINATOR),
                          _ivy.zeros_like(theta, device=device))
    vector_z = _ivy.where(theta != 0, e3 / (_ivy.sin(theta / 2) + MIN_DENOMINATOR),
                          _ivy.zeros_like(theta, device=device))

    # BS x 4
    return _ivy.concat([vector_x, vector_y, vector_z, theta], axis=-1)


def quaternion_to_polar_axis_angle(quaternion, device=None):
    """Convert quaternion :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]` to a polar axis angle representation, which
    constitutes the elevation and azimuth angles of the axis, as well as the rotation angle
    :math:`\mathbf{θ}_{paa} = [ϕ_e, ϕ_a, θ]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation>`_

    Parameters
    ----------
    quaternion
        Input quaternion *[batch_shape,4]*
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Polar axis angle representation *[batch_shape,3]*

    """

    if device is None:
        device = _ivy.dev(quaternion)

    # BS x 4
    vector_and_angle = quaternion_to_axis_angle(quaternion, device)

    # BS x 1
    theta = _ivy.acos(vector_and_angle[..., 2:3])
    phi = _ivy.atan2(vector_and_angle[..., 1:2], vector_and_angle[..., 0:1])

    # BS x 3
    return _ivy.concat([theta, phi, vector_and_angle[..., -1:]], axis=-1)


# noinspection PyUnusedLocal
def quaternion_to_rotation_vector(quaternion, device=None):
    """Convert quaternion :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]` to rotation vector
    :math:`\mathbf{θ}_{rv} = [θe_x, θe_y, θe_z]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_axis_and_angle_(rotation_vector)>`_

    Parameters
    ----------
    quaternion
        Input quaternion *[batch_shape,4]*
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Rotation vector *[batch_shape,3]*

    """

    if device is None:
        device = _ivy.dev(quaternion)

    # BS x 4
    vector_and_angle = quaternion_to_axis_angle(quaternion, device)

    # BS x 3
    return vector_and_angle[..., :-1] * vector_and_angle[..., -1:]


def get_random_axis_angle(batch_shape=None):
    """Generate random axis unit vector :math:`\mathbf{e} = [e_x, e_y, e_z]`
    and rotation angle :math:`θ`

    Parameters
    ----------
    batch_shape
        Shape of batch. Shape of [1] is assumed if None. (Default value = None)

    Returns
    -------
    ret
        Random rotation axis unit vector and angle *[batch_shape,4]*

    """

    return quaternion_to_axis_angle(_ivy_q.get_random_quaternion(batch_shape=batch_shape))
