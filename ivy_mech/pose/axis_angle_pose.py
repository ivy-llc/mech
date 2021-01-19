"""
Collection of Pose Conversion Functions to Axis-Angle Pose Format
"""

# global
from ivy.framework_handler import get_framework as _get_framework

# local
from ivy_mech.orientation import quaternion as _ivy_quat
from ivy_mech.orientation import axis_angle as _ivy_aa


# noinspection PyUnresolvedReferences
def mat_pose_to_rot_vec_pose(matrix, f=None):
    """
    Convert matrix pose :math:`\mathbf{P}\in\mathbb{R}^{3×4}` to rotation vector pose
    :math:`\mathbf{p}_{rv} = [\mathbf{x}_c, \mathbf{θ}_{rv}] = [x, y, z, θe_x, θe_y, θe_z]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_axis_and_angle_(rotation_vector)>`_

    :param matrix: Matrix pose *[batch_shape,3,4]*
    :type matrix: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Rotation vector pose *[batch_shape,6]*
    """

    f = _get_framework(matrix, f=f)

    # BS x 3
    translation = matrix[..., -1]

    # BS x 4
    quaternion = _ivy_quat.rot_mat_to_quaternion(matrix[..., 0:3], f=f)

    # BS x 3
    rot_vector = _ivy_aa.quaternion_to_rotation_vector(quaternion, f=f)

    # BS x 6
    return f.concatenate((translation, rot_vector), -1)


# noinspection PyUnresolvedReferences
def quaternion_pose_to_rot_vec_pose(quat_pose, f=None):
    """
    Convert quaternion pose :math:`\mathbf{p}_{q} = [\mathbf{x}_c, \mathbf{q}] = [x, y, z, q_i, q_j, q_k, q_r]` to
    rotation vector pose :math:`\mathbf{p}_{rv} = [\mathbf{x}_c, \mathbf{θ}_{rv}] = [x, y, z, θe_x, θe_y, θe_z]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_axis_and_angle_(rotation_vector)>`_

    :param quat_pose: Quaternion pose *[batch_shape,7]*
    :type quat_pose: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Rotation vector pose *[batch_shape,6]*
    """

    f = _get_framework(quat_pose, f=f)

    # BS x 4
    vector_and_angle = _ivy_aa.quaternion_to_axis_angle(quat_pose[..., 3:])

    # BS x 3
    rot_vec = vector_and_angle[..., :-1] * vector_and_angle[..., -1:]

    # BS x 6
    return f.concatenate((quat_pose[..., 0:3], rot_vec), -1)
