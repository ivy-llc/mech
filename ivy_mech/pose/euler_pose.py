"""
Collection of Pose Conversion Functions to Euler Pose Format
"""

# global
from ivy.framework_handler import get_framework as _get_framework

# local
from ivy_mech.orientation import euler_angles as _ivy_ea


# noinspection PyUnresolvedReferences
def mat_pose_to_euler_pose(matrix, convention='zyx', f=None):
    """
    Convert matrix pose :math:`\mathbf{P}\in\mathbb{R}^{3×4}` to :math:`abc` Euler angle pose
    :math:`\mathbf{p}_{xyz} = [\mathbf{x}_c, \mathbf{θ}_{xyz}] = [x, y, z, ϕ_a, ϕ_b, ϕ_c]`.\n
    `[reference] <https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/07/euler-angles1.pdf>`_

    :param matrix: Matrix pose *[batch_shape,3,4]*
    :type matrix: array
    :param convention: The axes for euler rotation, in order of L.H.S. matrix multiplication.
    :type convention: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Euler pose *[batch_shape,6]*
    """

    f = _get_framework(matrix, f=f)

    # BS x 3
    translation = matrix[..., 3]

    # BS x 3
    euler_angles = _ivy_ea.rot_mat_to_euler(matrix[..., 0:3], convention, f=f)

    # BS x 6
    return f.concatenate((translation, euler_angles), -1)
