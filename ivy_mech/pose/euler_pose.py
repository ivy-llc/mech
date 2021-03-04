"""
Collection of Pose Conversion Functions to Euler Pose Format
"""

# global
import ivy as _ivy

# local
from ivy_mech.orientation import euler_angles as _ivy_ea


# noinspection PyUnresolvedReferences
def mat_pose_to_euler_pose(matrix, convention='zyx'):
    """
    Convert matrix pose :math:`\mathbf{P}\in\mathbb{R}^{3×4}` to :math:`abc` Euler angle pose
    :math:`\mathbf{p}_{xyz} = [\mathbf{x}_c, \mathbf{θ}_{xyz}] = [x, y, z, ϕ_a, ϕ_b, ϕ_c]`.\n
    `[reference] <https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/07/euler-angles1.pdf>`_

    :param matrix: Matrix pose *[batch_shape,3,4]*
    :type matrix: array
    :param convention: The axes for euler rotation, in order of L.H.S. matrix multiplication.
    :type convention: str, optional
    :return: Euler pose *[batch_shape,6]*
    """

    # BS x 3
    translation = matrix[..., 3]

    # BS x 3
    euler_angles = _ivy_ea.rot_mat_to_euler(matrix[..., 0:3], convention)

    # BS x 6
    return _ivy.concatenate((translation, euler_angles), -1)


# noinspection PyUnresolvedReferences
def quaternion_pose_to_euler_pose(quaternion_pose, convention='zyx'):
    """
    Convert quaternion pose :math:`\mathbf{p}_{q} = [\mathbf{x}_c, \mathbf{q}] = [x, y, z, q_i, q_j, q_k, q_r]`
    to :math:`abc` Euler angle pose
    :math:`\mathbf{p}_{xyz} = [\mathbf{x}_c, \mathbf{θ}_{xyz}] = [x, y, z, ϕ_a, ϕ_b, ϕ_c]`.

    :param quaternion_pose: Quaternion pose *[batch_shape,7]*
    :type quaternion_pose: array
    :param convention: The axes for euler rotation, in order of L.H.S. matrix multiplication.
    :type convention: str, optional
    :return: Euler pose *[batch_shape,6]*
    """

    # BS x 3
    translation = quaternion_pose[..., :3]

    # BS x 3
    euler_angles = _ivy_ea.quaternion_to_euler(quaternion_pose[..., 3:], convention)

    # BS x 6
    return _ivy.concatenate((translation, euler_angles), -1)


# noinspection PyUnresolvedReferences
def axis_angle_pose_to_euler_pose(axis_angle_pose, convention='zyx'):
    """
    Convert axis-angle pose :math:`\mathbf{p}_{aa} = [\mathbf{x}_c, \mathbf{e}, θ] = [x, y, z, e_x, e_y, e_z, θ]`
     to :math:`abc` Euler angle pose :math:`\mathbf{p}_{xyz} = [\mathbf{x}_c, \mathbf{θ}_{xyz}] = [x, y, z, ϕ_a, ϕ_b, ϕ_c]`.

    :param axis_angle_pose: Axis-angle pose *[batch_shape,7]*
    :type axis_angle_pose: array
    :param convention: The axes for euler rotation, in order of L.H.S. matrix multiplication.
    :type convention: str, optional
    :return: Euler pose *[batch_shape,6]*
    """

    # BS x 3
    translation = axis_angle_pose[..., :3]

    # BS x 3
    euler_angles = _ivy_ea.axis_angle_to_euler(axis_angle_pose[..., 3:], convention)

    # BS x 6
    return _ivy.concatenate((translation, euler_angles), -1)
