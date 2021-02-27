"""
Collection of Rotation Conversion Functions to Quaternion Format
"""

# global
import ivy
import math as _math
from ivy.framework_handler import get_framework as _get_framework

# local
from ivy_mech.orientation import rotation_matrix as _ivy_rot_mat
from ivy_mech.orientation import axis_angle as _ivy_aa

MIN_DENOMINATOR = 1e-12


# Representation Conversions #
# ---------------------------#


def axis_angle_to_quaternion(axis_angle, f=None):
    """
    Convert rotation axis unit vector :math:`\mathbf{e} = [e_x, e_y, e_z]` and rotation angle :math:`θ` to quaternion
    :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Quaternions>`_

    :param axis: Axis to rotate about *[batch_shape,3]*
    :type axis: array
    :param angle: Angle to rotate about the rotation axis *[batch_shape,1]*
    :type angle: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Quaternion *[batch_shape,4]*
    """

    f = _get_framework(axis_angle, f=f)

    # BS x 1
    angle = axis_angle[..., -1:]
    n = ivy.cos(angle / 2, f=f)
    e1 = ivy.sin(angle / 2, f=f) * axis_angle[..., 0:1]
    e2 = ivy.sin(angle / 2, f=f) * axis_angle[..., 1:2]
    e3 = ivy.sin(angle / 2, f=f) * axis_angle[..., 2:3]

    # BS x 4
    quaternion = ivy.concatenate((e1, e2, e3, n), -1, f=f)
    return quaternion


def polar_axis_angle_to_quaternion(polar_axis_angle, f=None):
    """
    Convert polar axis-angle representation, which constitutes the elevation and azimuth angles of the axis as well
    as the rotation angle :math:`\mathbf{θ}_{paa} = [ϕ_e, ϕ_a, θ]`, to quaternion form
    :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation>`_

    :param polar_axis_angle: Polar axis angle representation *[batch_shape,3]*
    :type polar_axis_angle: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Quaternion *[batch_shape,4]*
    """

    f = _get_framework(polar_axis_angle, f=f)

    # BS x 1
    theta = polar_axis_angle[..., 0:1]
    phi = polar_axis_angle[..., 1:2]
    angle = polar_axis_angle[..., 2:3]
    x = ivy.sin(theta, f=f) * ivy.cos(phi, f=f)
    y = ivy.sin(theta, f=f) * ivy.sin(phi, f=f)
    z = ivy.cos(theta, f=f)

    # BS x 3
    vector = ivy.concatenate((x, y, z), -1, f=f)

    # BS x 4
    return axis_angle_to_quaternion(ivy.concatenate([vector, angle], -1, f=f))


def rot_mat_to_quaternion(rot_mat, f=None):
    """
    Convert rotation matrix :math:`\mathbf{R}\in\mathbb{R}^{3×3}` to quaternion
    :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]`.\n
    `[reference] <http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/>`_

    :param rot_mat: Rotation matrix *[batch_shape,3,3]*
    :type rot_mat: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Quaternion *[batch_shape,4]*
    """

    f = _get_framework(rot_mat, f=f)

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
    quat_1 = ivy.concatenate((qx_1, qy_1, qz_1, qw_1), -2, f=f)

    # elif (m[:,0,0] > m[:,1,1]) and (m[:,0,0] > m[:,2,2])
    # BS x 1 x 1
    s_2 = ((1 + rot_mat[..., 0:1, 0:1] - rot_mat[..., 1:2, 1:2] - rot_mat[..., 2:3, 2:3]) ** 0.5) * 2
    qw_2 = (rot_mat[..., 2:3, 1:2] - rot_mat[..., 1:2, 2:3]) / (s_2 + MIN_DENOMINATOR)
    qx_2 = 0.25 * s_2
    qy_2 = (rot_mat[..., 0:1, 1:2] + rot_mat[..., 1:2, 0:1]) / (s_2 + MIN_DENOMINATOR)
    qz_2 = (rot_mat[..., 0:1, 2:3] + rot_mat[..., 2:3, 0:1]) / (s_2 + MIN_DENOMINATOR)

    # BS x 4 x 1
    quat_2 = ivy.concatenate((qx_2, qy_2, qz_2, qw_2), -2, f=f)

    # elif m[:,1,1] > m[:,2,2]
    # BS x 1 x 1
    s_3 = ((1 + rot_mat[..., 1:2, 1:2] - rot_mat[..., 0:1, 0:1] - rot_mat[..., 2:3, 2:3]) ** 0.5) * 2
    qw_3 = (rot_mat[..., 0:1, 2:3] - rot_mat[..., 2:3, 0:1]) / (s_3 + MIN_DENOMINATOR)
    qx_3 = (rot_mat[..., 0:1, 1:2] + rot_mat[..., 1:2, 0:1]) / (s_3 + MIN_DENOMINATOR)
    qy_3 = 0.25 * s_3
    qz_3 = (rot_mat[..., 1:2, 2:3] + rot_mat[..., 2:3, 1:2]) / (s_3 + MIN_DENOMINATOR)

    # BS x 4 x 1
    quat_3 = ivy.concatenate((qx_3, qy_3, qz_3, qw_3), -2, f=f)

    # else
    # BS x 1 x 1
    s_4 = ((1 + rot_mat[..., 2:3, 2:3] - rot_mat[..., 0:1, 0:1] - rot_mat[..., 1:2, 1:2]) ** 0.5) * 2
    qw_4 = (rot_mat[..., 1:2, 0:1] - rot_mat[..., 0:1, 1:2]) / (s_4 + MIN_DENOMINATOR)
    qx_4 = (rot_mat[..., 0:1, 2:3] + rot_mat[..., 2:3, 0:1]) / (s_4 + MIN_DENOMINATOR)
    qy_4 = (rot_mat[..., 1:2, 2:3] + rot_mat[..., 2:3, 1:2]) / (s_4 + MIN_DENOMINATOR)
    qz_4 = 0.25 * s_4

    # BS x 4 x 1
    quat_4 = ivy.concatenate((qx_4, qy_4, qz_4, qw_4), -2, f=f)
    quat_3_or_other = ivy.where(rot_mat[..., 1:2, 1:2] > rot_mat[..., 2:3, 2:3], quat_3, quat_4, f=f)
    quat_2_or_other = \
        ivy.where(ivy.logical_and((rot_mat[..., 0:1, 0:1] > rot_mat[..., 1:2, 1:2]),
                                  (rot_mat[..., 0:1, 0:1] > rot_mat[..., 2:3, 2:3]), f=f), quat_2, quat_3_or_other, f=f)

    # BS x 4
    return ivy.where(tr > 0, quat_1, quat_2_or_other, f=f)[..., 0]


def rotation_vector_to_quaternion(rot_vector, f=None):
    """
    Convert rotation vector :math:`\mathbf{θ}_{rv} = [θe_x, θe_y, θe_z]` to quaternion
    :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_axis_and_angle_(rotation_vector)>`_

    :param rot_vector: Rotation vector *[batch_shape,3]*
    :type rot_vector: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Quaternion *[batch_shape,4]*
    """

    f = _get_framework(rot_vector, f=f)

    # BS x 1
    theta = (ivy.reduce_sum(rot_vector ** 2, axis=-1, keepdims=True, f=f)) ** 0.5

    # BS x 3
    vector = rot_vector / (theta + MIN_DENOMINATOR)

    # BS x 4
    return axis_angle_to_quaternion(ivy.concatenate([vector, theta], -1, f=f), f=f)


def euler_to_quaternion(euler_angles, convention='zyx', batch_shape=None, f=None):
    """
    Convert :math:`zyx` Euler angles :math:`\mathbf{θ}_{abc} = [ϕ_a, ϕ_b, ϕ_c]` to quaternion
    :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles>`_

    :param euler_angles: Euler angles, in zyx rotation order form *[batch_shape,3]*
    :type euler_angles: array
    :param convention: The axes for euler rotation, in order of L.H.S. matrix multiplication.
    :type convention: str, optional
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Quaternion *[batch_shape,4]*
    """

    f = _get_framework(euler_angles, f=f)

    if batch_shape is None:
        batch_shape = euler_angles.shape[:-1]

    # BS x 4
    return rot_mat_to_quaternion(_ivy_rot_mat.euler_to_rot_mat(euler_angles, convention, batch_shape, f=f))


# Quaternion Operations #
# ----------------------#

def inverse_quaternion(quaternion, f=None):
    """
    Compute inverse quaternion :math:`\mathbf{q}^{-1}.\n
    `[reference] <https://github.com/KieranWynn/pyquaternion/blob/446c31cba66b708e8480871e70b06415c3cb3b0f/pyquaternion/quaternion.py#L473>`_

    :param quaternion: Quaternion *[batch_shape,4]*
    :type quaternion: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Inverse quaternion *[batch_shape,4]*
    """

    f = _get_framework(quaternion, f=f)

    # BS x 1
    sum_of_squares = ivy.reduce_sum(quaternion ** 2, -1, f=f)
    vector_conjugate = ivy.concatenate((-quaternion[..., 0:3], quaternion[..., -1:]), -1, f=f)
    return vector_conjugate/(sum_of_squares + MIN_DENOMINATOR)


def get_random_quaternion(f, max_rot_ang=_math.pi, batch_shape=None):
    """
    Generate random quaternion :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]`, adhering to maximum absolute rotation angle.\n
    `[reference] <https://en.wikipedia.org/wiki/Quaternion>`_

    :param f: Machine learning framework.
    :type f: ml_framework
    :param max_rot_ang: Absolute value of maximum rotation angle for quaternion. Default value of :math:`π`.
    :type max_rot_ang: float, optional
    :param batch_shape: Shape of batch. Shape of [1] is assumed if None.
    :type batch_shape: sequence of ints, optional
    :return: Random quaternion *[batch_shape,4]*
    """

    if f is None:
        raise Exception('framework f must be specified for calling ivy.get_random_quaternion()')

    if batch_shape is None:
        batch_shape = []

    # BS x 3
    quaternion_vector = ivy.random_uniform(0, 1, list(batch_shape) + [3], f=f)
    vec_len = ivy.norm(quaternion_vector, f=f)
    quaternion_vector /= (vec_len + MIN_DENOMINATOR)

    # BS x 1
    theta = ivy.random_uniform(-max_rot_ang, max_rot_ang, list(batch_shape) + [1], f=f)

    # BS x 4
    return axis_angle_to_quaternion(ivy.concatenate([quaternion_vector, theta], -1, f=f))


def scale_quaternion_rotation_angle(quaternion, scale, f=None):
    """
    Scale the rotation angle :math:`θ` of a given quaternion :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Quaternions>`_

    :param quaternion: Quaternion *[batch_shape,4]*
    :type quaternion: array
    :param scale: Value to scale by the rotation angle by. Can be negative to change rotation direction.
    :type scale: float
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Quaternion with rotation angle scaled *[batch_shape,4]*
    """

    f = _get_framework(quaternion, f=f)

    # BS x 4
    vector_and_angle = _ivy_aa.quaternion_to_axis_angle(quaternion, f=f)

    # BS x 1
    scaled_angle = vector_and_angle[..., -1:] * scale

    # BS x 4
    return axis_angle_to_quaternion(ivy.concatenate(
        [vector_and_angle[..., :-1], scaled_angle], -1, f=f))


def hamilton_product(quaternion1, quaternion2, f=None):
    """
    Compute hamilton product :math:`\mathbf{h}_p = \mathbf{q}_1 × \mathbf{q}_2` between
    :math:`\mathbf{q}_1 = [q_{1i}, q_{1j}, q_{1k}, q_{1r}]` and
    :math:`\mathbf{q}_2 = [q_{2i}, q_{2j}, q_{2k}, q_{2r}]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Quaternion#Hamilton_product>`_

    :param quaternion1: Quaternion 1 *[batch_shape,4]*
    :type quaternion1: array
    :param quaternion2: Quaternion 2 *[batch_shape,4]*
    :type quaternion2: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: New quaternion after product *[batch_shape,4]*
    """

    f = _get_framework(quaternion1, f=f)

    # BS x 1
    a1 = quaternion1[..., 3:4]
    a2 = quaternion2[..., 3:4]
    b1 = quaternion1[..., 0:1]
    b2 = quaternion2[..., 0:1]
    c1 = quaternion1[..., 1:2]
    c2 = quaternion2[..., 1:2]
    d1 = quaternion1[..., 2:3]
    d2 = quaternion2[..., 2:3]

    term_r = a1*a2 - b1*b2 - c1*c2 - d1*d2
    term_i = a1*b2 + b1*a2 + c1*d2 - d1*c2
    term_j = a1*c2 - b1*d2 + c1*a2 + d1*b2
    term_k = a1*d2 + b1*c2 - c1*b2 + d1*a2

    # BS x 4
    return ivy.concatenate((term_i, term_j, term_k, term_r), -1, f=f)
