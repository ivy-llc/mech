"""Collection of Pose Conversion Functions to Euler Pose Format"""
# global
import ivy

# local
from ivy_mech.orientation import euler_angles as ivy_ea


# noinspection PyUnresolvedReferences
def mat_pose_to_euler_pose(matrix, convention="zyx"):
    r"""Convert matrix pose :math:`\mathbf{P}\in\mathbb{R}^{3×4}` to
    :math:`abc` Euler angle pose
    :math:`\mathbf{p}_{xyz} = [\mathbf{x}_c, \mathbf{θ}_{xyz}] = [x, y, z, ϕ_a, ϕ_b, ϕ_c]`.\n # noqa
    `[reference] <https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/07/euler-angles1.pdf>`_ # noqa

    Parameters
    ----------
    matrix
        Matrix pose *[batch_shape,3,4]*
    convention
        The axes for euler rotation, in order of L.H.S. matrix multiplication.
        (Default value = 'zyx')

    Returns
    -------
    ret
        Euler pose *[batch_shape,6]*

    """
    # BS x 3
    translation = matrix[..., 3]

    # BS x 3
    euler_angles = ivy_ea.rot_mat_to_euler(matrix[..., 0:3], convention)

    # BS x 6
    return ivy.concat([translation, euler_angles], axis=-1)


# noinspection PyUnresolvedReferences
def quaternion_pose_to_euler_pose(quaternion_pose, convention="zyx"):
    r"""Convert quaternion pose
    :math:`\mathbf{p}_{q} = [\mathbf{x}_c, \mathbf{q}] = [x, y, z, q_i, q_j, q_k, q_r]` # noqa
    to :math:`abc` Euler angle pose
    :math:`\mathbf{p}_{xyz} = [\mathbf{x}_c, \mathbf{θ}_{xyz}] = [x, y, z, ϕ_a, ϕ_b, ϕ_c]`. # noqa

    Parameters
    ----------
    quaternion_pose
        Quaternion pose *[batch_shape,7]*
    convention
        The axes for euler rotation, in order of L.H.S. matrix multiplication.
        (Default value = 'zyx')

    Returns
    -------
    ret
        Euler pose *[batch_shape,6]*

    """
    # BS x 3
    translation = quaternion_pose[..., :3]

    # BS x 3
    euler_angles = ivy_ea.quaternion_to_euler(quaternion_pose[..., 3:], convention)

    # BS x 6
    return ivy.concat([translation, euler_angles], axis=-1)


# noinspection PyUnresolvedReferences
def axis_angle_pose_to_euler_pose(axis_angle_pose, convention="zyx"):
    r"""Convert axis-angle pose
    :math:`\mathbf{p}_{aa} = [\mathbf{x}_c, \mathbf{e}, θ] = [x, y, z, e_x, e_y, e_z, θ]` # noqa
     to :math:`abc` Euler angle pose
     :math:`\mathbf{p}_{xyz} = [\mathbf{x}_c, \mathbf{θ}_{xyz}] = [x, y, z, ϕ_a, ϕ_b, ϕ_c]`. # noqa

    Parameters
    ----------
    axis_angle_pose
        Axis-angle pose *[batch_shape,7]*
    convention
        The axes for euler rotation, in order of L.H.S. matrix multiplication.
        (Default value = 'zyx')

    Returns
    -------
    ret
        Euler pose *[batch_shape,6]*

    """
    # BS x 3
    translation = axis_angle_pose[..., :3]

    # BS x 3
    euler_angles = ivy_ea.axis_angle_to_euler(axis_angle_pose[..., 3:], convention)

    # BS x 6
    return ivy.concat([translation, euler_angles], axis=-1)
