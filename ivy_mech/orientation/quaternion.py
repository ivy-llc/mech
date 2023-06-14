"""Collection of Rotation Conversion Functions to Quaternion Format"""
# global
import ivy
import math

# local
from ivy_mech.orientation import rotation_matrix as ivy_rot_mat
from ivy_mech.orientation import axis_angle as ivy_aa

MIN_DENOMINATOR = 1e-12


# Representation Conversions #
# ---------------------------#


def axis_angle_to_quaternion(axis_angle):
    r"""Convert rotation axis unit vector :math:`\mathbf{e} = [e_x, e_y, e_z]`
    and rotation angle :math:`θ` to quaternion
    :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Quaternions>`_ # noqa

    Parameters
    ----------
    axis_angle
        Axis to rotate about and angle to rotate *[batch_shape,4]*

    Returns
    -------
    ret
        Quaternion *[batch_shape,4]*

    """
    # BS x 1
    angle = axis_angle[..., -1:]
    n = ivy.cos(angle / 2)
    e1 = ivy.sin(angle / 2) * axis_angle[..., 0:1]
    e2 = ivy.sin(angle / 2) * axis_angle[..., 1:2]
    e3 = ivy.sin(angle / 2) * axis_angle[..., 2:3]

    # BS x 4
    quaternion = ivy.concat([e1, e2, e3, n], axis=-1)
    return quaternion


def polar_axis_angle_to_quaternion(polar_axis_angle):
    r"""Convert polar axis-angle representation, which constitutes the elevation and
    azimuth angles of the axis as well
    as the rotation angle :math:`\mathbf{θ}_{paa} = [ϕ_e, ϕ_a, θ]`, to quaternion form
    :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation>`_

    Parameters
    ----------
    polar_axis_angle
        Polar axis angle representation *[batch_shape,3]*

    Returns
    -------
    ret
        Quaternion *[batch_shape,4]*

    """
    # BS x 1
    theta = polar_axis_angle[..., 0:1]
    phi = polar_axis_angle[..., 1:2]
    angle = polar_axis_angle[..., 2:3]
    x = ivy.sin(theta) * ivy.cos(phi)
    y = ivy.sin(theta) * ivy.sin(phi)
    z = ivy.cos(theta)

    # BS x 3
    vector = ivy.concat([x, y, z], axis=-1)

    # BS x 4
    return axis_angle_to_quaternion(ivy.concat([vector, angle], axis=-1))


def rot_mat_to_quaternion(rot_mat):
    r"""Convert rotation matrix :math:`\mathbf{R}\in\mathbb{R}^{3×3}` to quaternion
    :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]`.\n
    `[reference] <http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/>`_ # noqa

    Parameters
    ----------
    rot_mat
        Rotation matrix *[batch_shape,3,3]*

    Returns
    -------
    ret
        Quaternion *[batch_shape,4]*

    """
    # BS x 1 x 1
    tr = rot_mat[..., 0:1, 0:1] + rot_mat[..., 1:2, 1:2] + rot_mat[..., 2:3, 2:3]

    # if tf > 0
    # BS x 1 x 1
    s_1 = ((tr + 1) ** 0.5) * 2
    qw_1 = 0.25 * s_1
    qx_1 = (rot_mat[..., 2:3, 1:2] - rot_mat[..., 1:2, 2:3]) / (s_1 + MIN_DENOMINATOR)
    qy_1 = (rot_mat[..., 0:1, 2:3] - rot_mat[..., 2:3, 0:1]) / (s_1 + MIN_DENOMINATOR)
    qz_1 = (rot_mat[..., 1:2, 0:1] - rot_mat[..., 0:1, 1:2]) / (s_1 + MIN_DENOMINATOR)

    # BS x 4 x 1
    quat_1 = ivy.concat([qx_1, qy_1, qz_1, qw_1], axis=-2)

    # elif (m[:,0,0] > m[:,1,1]) and (m[:,0,0] > m[:,2,2])
    # BS x 1 x 1
    s_2 = (
        (1 + rot_mat[..., 0:1, 0:1] - rot_mat[..., 1:2, 1:2] - rot_mat[..., 2:3, 2:3])
        ** 0.5
    ) * 2
    qw_2 = (rot_mat[..., 2:3, 1:2] - rot_mat[..., 1:2, 2:3]) / (s_2 + MIN_DENOMINATOR)
    qx_2 = 0.25 * s_2
    qy_2 = (rot_mat[..., 0:1, 1:2] + rot_mat[..., 1:2, 0:1]) / (s_2 + MIN_DENOMINATOR)
    qz_2 = (rot_mat[..., 0:1, 2:3] + rot_mat[..., 2:3, 0:1]) / (s_2 + MIN_DENOMINATOR)

    # BS x 4 x 1
    quat_2 = ivy.concat([qx_2, qy_2, qz_2, qw_2], axis=-2)

    # elif m[:,1,1] > m[:,2,2]
    # BS x 1 x 1
    s_3 = (
        (1 + rot_mat[..., 1:2, 1:2] - rot_mat[..., 0:1, 0:1] - rot_mat[..., 2:3, 2:3])
        ** 0.5
    ) * 2
    qw_3 = (rot_mat[..., 0:1, 2:3] - rot_mat[..., 2:3, 0:1]) / (s_3 + MIN_DENOMINATOR)
    qx_3 = (rot_mat[..., 0:1, 1:2] + rot_mat[..., 1:2, 0:1]) / (s_3 + MIN_DENOMINATOR)
    qy_3 = 0.25 * s_3
    qz_3 = (rot_mat[..., 1:2, 2:3] + rot_mat[..., 2:3, 1:2]) / (s_3 + MIN_DENOMINATOR)

    # BS x 4 x 1
    quat_3 = ivy.concat([qx_3, qy_3, qz_3, qw_3], axis=-2)

    # else
    # BS x 1 x 1
    s_4 = (
        (1 + rot_mat[..., 2:3, 2:3] - rot_mat[..., 0:1, 0:1] - rot_mat[..., 1:2, 1:2])
        ** 0.5
    ) * 2
    qw_4 = (rot_mat[..., 1:2, 0:1] - rot_mat[..., 0:1, 1:2]) / (s_4 + MIN_DENOMINATOR)
    qx_4 = (rot_mat[..., 0:1, 2:3] + rot_mat[..., 2:3, 0:1]) / (s_4 + MIN_DENOMINATOR)
    qy_4 = (rot_mat[..., 1:2, 2:3] + rot_mat[..., 2:3, 1:2]) / (s_4 + MIN_DENOMINATOR)
    qz_4 = 0.25 * s_4

    # BS x 4 x 1
    quat_4 = ivy.concat([qx_4, qy_4, qz_4, qw_4], axis=-2)
    quat_3_or_other = ivy.where(
        rot_mat[..., 1:2, 1:2] > rot_mat[..., 2:3, 2:3], quat_3, quat_4
    )
    quat_2_or_other = ivy.where(
        ivy.logical_and(
            (rot_mat[..., 0:1, 0:1] > rot_mat[..., 1:2, 1:2]),
            (rot_mat[..., 0:1, 0:1] > rot_mat[..., 2:3, 2:3]),
        ),
        quat_2,
        quat_3_or_other,
    )

    # BS x 4
    return ivy.where(tr > 0, quat_1, quat_2_or_other)[..., 0]


def rotation_vector_to_quaternion(rot_vector):
    r"""Convert rotation vector :math:`\mathbf{θ}_{rv} = [θe_x, θe_y, θe_z]` to
    quaternion
    :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_axis_and_angle_(rotation_vector)>`_ # noqa

    Parameters
    ----------
    rot_vector
        Rotation vector *[batch_shape,3]*

    Returns
    -------
    ret
        Quaternion *[batch_shape,4]*

    """
    # BS x 1
    theta = (ivy.sum(rot_vector**2, axis=-1, keepdims=True)) ** 0.5

    # BS x 3
    vector = rot_vector / (theta + MIN_DENOMINATOR)

    # BS x 4
    return axis_angle_to_quaternion(ivy.concat([vector, theta], axis=-1))


def euler_to_quaternion(euler_angles, convention="zyx", batch_shape=None):
    r"""Convert :math:`zyx` Euler angles :math:`\mathbf{θ}_{abc} = [ϕ_a, ϕ_b, ϕ_c]` to
    quaternion :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]`.\n `[reference]
    <https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles>`_

    Parameters
    ----------
    euler_angles
        Euler angles, in zyx rotation order form *[batch_shape,3]*
    convention
        The axes for euler rotation, in order of L.H.S. matrix multiplication.
        (Default value = 'zyx')
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)

    Returns
    -------
    ret
        Quaternion *[batch_shape,4]*

    """
    if batch_shape is None:
        batch_shape = euler_angles.shape[:-1]

    # BS x 4
    return rot_mat_to_quaternion(
        ivy_rot_mat.euler_to_rot_mat(euler_angles, convention, batch_shape)
    )


# Quaternion Operations #
# ----------------------#


def inverse_quaternion(quaternion):
    r"""Compute inverse quaternion :math:`\mathbf{q}^{-1}.\n `[reference]
    <https://github.com/KieranWynn/pyquaternion/blob
    /446c31cba66b708e8480871e70b06415c3cb3b0f/pyquaternion/quaternion.py#L473>`_

    Parameters
    ----------
    quaternion
        Quaternion *[batch_shape,4]*

    Returns
    -------
    ret
        Inverse quaternion *[batch_shape,4]*

    """
    # BS x 1
    sum_of_squares = ivy.sum(quaternion**2, axis=-1)
    vector_conjugate = ivy.concat(
        [-quaternion[..., 0:3], quaternion[..., -1:]], axis=-1
    )
    return vector_conjugate / (sum_of_squares + MIN_DENOMINATOR)


def get_random_quaternion(max_rot_ang=math.pi, batch_shape=None):
    r"""Generate random quaternion :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]`,
    adhering to maximum absolute rotation angle.\n
    `[reference] <https://en.wikipedia.org/wiki/Quaternion>`_

    Parameters
    ----------
    max_rot_ang
        Absolute value of maximum rotation angle for quaternion.
        Default value of :math:`π`.
    batch_shape
        Shape of batch. Shape of [1] is assumed if None. (Default value = None)

    Returns
    -------
    ret
        Random quaternion *[batch_shape,4]*

    """
    if batch_shape is None:
        batch_shape = []

    # BS x 3
    quaternion_vector = ivy.random_uniform(low=0, high=1, shape=list(batch_shape) + [3])
    vec_len = ivy.vector_norm(quaternion_vector, axis=-1)
    quaternion_vector /= vec_len + MIN_DENOMINATOR

    # BS x 1
    theta = ivy.random_uniform(
        low=-max_rot_ang, high=max_rot_ang, shape=list(batch_shape) + [1]
    )

    # BS x 4
    return axis_angle_to_quaternion(ivy.concat([quaternion_vector, theta], axis=-1))


def scale_quaternion_rotation_angle(quaternion, scale):
    r"""Scale the rotation angle :math:`θ` of a given quaternion :math:`\mathbf{q} = [
    q_i, q_j, q_k, q_r]`.\n `[reference]
    <https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
    #Quaternions>`_

    Parameters
    ----------
    quaternion
        Quaternion *[batch_shape,4]*
    scale
        Value to scale by the rotation angle by. Can be negative to change
        rotation direction.

    Returns
    -------
    ret
        Quaternion with rotation angle scaled *[batch_shape,4]*

    """
    # BS x 4
    vector_and_angle = ivy_aa.quaternion_to_axis_angle(quaternion)

    # BS x 1
    scaled_angle = vector_and_angle[..., -1:] * scale

    # BS x 4
    return axis_angle_to_quaternion(
        ivy.concat([vector_and_angle[..., :-1], scaled_angle], axis=-1)
    )


def hamilton_product(quaternion1, quaternion2):
    r"""Compute hamilton product
    :math:`\mathbf{h}_p = \mathbf{q}_1 × \mathbf{q}_2` between
    :math:`\mathbf{q}_1 = [q_{1i}, q_{1j}, q_{1k}, q_{1r}]` and
    :math:`\mathbf{q}_2 = [q_{2i}, q_{2j}, q_{2k}, q_{2r}]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Quaternion#Hamilton_product>`_

    Parameters
    ----------
    quaternion1
        Quaternion 1 *[batch_shape,4]*
    quaternion2
        Quaternion 2 *[batch_shape,4]*

    Returns
    -------
    ret
        New quaternion after product *[batch_shape,4]*

    """
    # BS x 1
    a1 = quaternion1[..., 3:4]
    a2 = quaternion2[..., 3:4]
    b1 = quaternion1[..., 0:1]
    b2 = quaternion2[..., 0:1]
    c1 = quaternion1[..., 1:2]
    c2 = quaternion2[..., 1:2]
    d1 = quaternion1[..., 2:3]
    d2 = quaternion2[..., 2:3]

    term_r = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    term_i = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    term_j = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    term_k = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

    # BS x 4
    return ivy.concat([term_i, term_j, term_k, term_r], axis=-1)
