"""
Collection of Pose Conversion Functions to Matrix Format
"""

# global
import ivy
from ivy.framework_handler import get_framework as _get_framework

# local
from ivy_mech.orientation import rotation_matrix as _ivy_rot_mat
from ivy_mech.orientation import quaternion as _ivy_quat


def quaternion_pose_to_mat_pose(quat_pose, f=None):
    """
    Convert quaternion pose :math:`\mathbf{p}_{q} = [\mathbf{x}_c, \mathbf{q}] = [x, y, z, q_i, q_j, q_k, q_r]` to
    matrix pose :math:`\mathbf{P}\in\mathbb{R}^{3×4}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Conversion_to_and_from_the_matrix_representation>`_

    :param quat_pose: Quaternion pose *[batch_shape,7]*
    :type quat_pose: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Matrix pose *[batch_shape,3,4]*
    """

    f = _get_framework(quat_pose, f=f)

    # BS x 3 x 3
    rot_mat = _ivy_rot_mat.quaternion_to_rot_mat(quat_pose[..., 3:], f=f)

    # BS x 3 x 1
    rhs = ivy.expand_dims(quat_pose[..., 0:3], -1, f=f)

    # BS x 3 x 4
    return ivy.concatenate((rot_mat, rhs), -1, f=f)


def euler_pose_to_mat_pose(euler_pose, convention='zyx', batch_shape=None, f=None):
    """
    Convert :math: Euler angle pose
    :math:`\mathbf{p}_{abc} = [\mathbf{x}_c, \mathbf{θ}_{xyz}] = [x, y, z, ϕ_a, ϕ_b, ϕ_c]` to matrix pose
    :math:`\mathbf{P}\in\mathbb{R}^{3×4}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix>`_

    :param euler_pose: Euler angle pose *[batch_shape,6]*
    :type euler_pose: array
    :param convention: The axes for euler rotation, in order of L.H.S. matrix multiplication.
    :type convention: str, optional
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Matrix pose *[batch_shape,3,4]*
    """

    f = _get_framework(euler_pose, f=f)

    if batch_shape is None:
        batch_shape = euler_pose.shape[:-1]

    # BS x 3 x 3
    rot_mat = _ivy_rot_mat.euler_to_rot_mat(euler_pose[..., 3:], convention, batch_shape, f=f)

    # BS x 3 x 4
    return ivy.concatenate((rot_mat, ivy.expand_dims(euler_pose[..., 0:3], -1, f=f)), -1, f=f)


def rot_vec_pose_to_mat_pose(rot_vec_pose, f=None):
    """
    Convert rotation vector pose :math:`\mathbf{p}_{rv} = [\mathbf{x}_c, \mathbf{θ}_{rv}] = [x, y, z, θe_x, θe_y, θe_z]`
    to matrix pose :math:`\mathbf{P}\in\mathbb{R}^{3×4}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_axis_and_angle_(rotation_vector)>`_

    :param rot_vec_pose: Rotation vector pose *[batch_shape,6]*
    :type rot_vec_pose: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Matrix pose *[batch_shape,3,4]*
    """

    f = _get_framework(rot_vec_pose, f=f)

    # BS x 4
    quaternion = _ivy_quat.rotation_vector_to_quaternion(rot_vec_pose[..., 3:], f=f)

    # BS x 3 x 3
    rot_mat = _ivy_rot_mat.quaternion_to_rot_mat(quaternion, f=f)

    # BS x 3 x 4
    return ivy.concatenate((rot_mat, ivy.expand_dims(rot_vec_pose[..., 0:3], -1, f=f)), -1, f=f)


def axis_angle_pose_to_mat_pose(axis_angle_pose, f=None):
    """
    Convert axis-angle pose :math:`\mathbf{p}_{aa} = [\mathbf{x}_c, \mathbf{e}, θ] = [x, y, z, e_x, e_y, e_z, θ]` to
    matrix pose :math:`\mathbf{P}\in\mathbb{R}^{3×4}`.

    :param axis_angle_pose: Quaternion pose *[batch_shape,7]*
    :type axis_angle_pose: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Matrix pose *[batch_shape,3,4]*
    """

    f = _get_framework(axis_angle_pose, f=f)

    # BS x 3 x 3
    rot_mat = _ivy_rot_mat.axis_angle_to_rot_mat(axis_angle_pose[..., 3:])

    # BS x 3 x 4
    return ivy.concatenate(
        (rot_mat, ivy.expand_dims(axis_angle_pose[..., :3], -1, f=f)), -1, f=f)
