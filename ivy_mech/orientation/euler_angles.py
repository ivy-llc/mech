"""
Collection of Rotation Conversion Functions to Euler Angles Format
"""

# global
import ivy as _ivy
import math as _math

# local
from ivy_mech.orientation import rotation_matrix as _ivy_rot_mat

GIMBAL_TOL = 1e-4
VALID_EULER_CONVENTIONS = ['xyx', 'yzy', 'zxz', 'xzx', 'yxy', 'zyz',
                           'xyz', 'yzx', 'zxy', 'xzy', 'yxz', 'zyx']


# Euler Helpers #
# --------------#

def _rot_mat_to_xyx_euler(rot_mat):
    # BS x 1
    euler_angles_1 = _ivy.acos(rot_mat[..., 0, 0:1])

    gimbal_validity = _ivy.abs(rot_mat[..., 0, 2:3]) > GIMBAL_TOL

    r23 = rot_mat[..., 1, 2:3]
    r22 = rot_mat[..., 1, 1:2]
    gimbal_euler_angles_0 = _ivy.atan2(-r23, r22)
    gimbal_euler_angles_2 = _ivy.zeros_like(gimbal_euler_angles_0)

    # BS x 3
    gimbal_euler_angles = _ivy.concatenate((gimbal_euler_angles_0, euler_angles_1, gimbal_euler_angles_2), -1)

    # BS x 1
    r12 = rot_mat[..., 0, 1:2]
    r13 = rot_mat[..., 0, 2:3]
    r21 = rot_mat[..., 1, 0:1]
    r31 = rot_mat[..., 2, 0:1]
    normal_euler_angles_0 = _ivy.atan2(r12, r13)
    normal_euler_angles_2 = _ivy.atan2(r21, -r31)

    # BS x 3
    normal_euler_angles = _ivy.concatenate((normal_euler_angles_0, euler_angles_1, normal_euler_angles_2), -1)

    return _ivy.where(gimbal_validity, normal_euler_angles, gimbal_euler_angles)


def _rot_mat_to_yzy_euler(rot_mat):
    # BS x 1
    euler_angles_1 = _ivy.acos(rot_mat[..., 1, 1:2])

    gimbal_validity = _ivy.abs(rot_mat[..., 1, 0:1]) > GIMBAL_TOL

    r31 = rot_mat[..., 2, 0:1]
    r33 = rot_mat[..., 2, 2:3]
    gimbal_euler_angles_0 = _ivy.atan2(-r31, r33)
    gimbal_euler_angles_2 = _ivy.zeros_like(gimbal_euler_angles_0)

    # BS x 3
    gimbal_euler_angles = _ivy.concatenate((gimbal_euler_angles_0, euler_angles_1, gimbal_euler_angles_2), -1)

    # BS x 1
    r23 = rot_mat[..., 1, 2:3]
    r21 = rot_mat[..., 1, 0:1]
    r32 = rot_mat[..., 2, 1:2]
    r12 = rot_mat[..., 0, 1:2]
    normal_euler_angles_0 = _ivy.atan2(r23, r21)
    normal_euler_angles_2 = _ivy.atan2(r32, r12)

    # BS x 3
    normal_euler_angles = _ivy.concatenate((normal_euler_angles_0, euler_angles_1, normal_euler_angles_2), -1)

    return _ivy.where(gimbal_validity, normal_euler_angles, gimbal_euler_angles)


def _rot_mat_to_zxz_euler(rot_mat):
    # BS x 1
    euler_angles_1 = _ivy.acos(rot_mat[..., 2, 2:3])

    gimbal_validity = _ivy.abs(rot_mat[..., 0, 2:3]) > GIMBAL_TOL

    r12 = rot_mat[..., 0, 1:2]
    r11 = rot_mat[..., 0, 0:1]
    gimbal_euler_angles_0 = _ivy.atan2(-r12, r11)
    gimbal_euler_angles_2 = _ivy.zeros_like(gimbal_euler_angles_0)

    # BS x 3
    gimbal_euler_angles = _ivy.concatenate((gimbal_euler_angles_0, euler_angles_1, gimbal_euler_angles_2), -1)

    # BS x 1
    r31 = rot_mat[..., 2, 0:1]
    r32 = rot_mat[..., 2, 1:2]
    r13 = rot_mat[..., 0, 2:3]
    r23 = rot_mat[..., 1, 2:3]
    normal_euler_angles_0 = _ivy.atan2(r31, r32)
    normal_euler_angles_2 = _ivy.atan2(r13, -r23)

    # BS x 3
    normal_euler_angles = _ivy.concatenate((normal_euler_angles_0, euler_angles_1, normal_euler_angles_2), -1)

    return _ivy.where(gimbal_validity, normal_euler_angles, gimbal_euler_angles)


def _rot_mat_to_xzx_euler(rot_mat):
    # BS x 1
    euler_angles_1 = _ivy.acos(rot_mat[..., 0, 0:1])

    gimbal_validity = _ivy.abs(rot_mat[..., 0, 2:3]) > GIMBAL_TOL

    r32 = rot_mat[..., 2, 1:2]
    r33 = rot_mat[..., 2, 2:3]
    gimbal_euler_angles_0 = _ivy.atan2(r32, r33)
    gimbal_euler_angles_2 = _ivy.zeros_like(gimbal_euler_angles_0)

    # BS x 3
    gimbal_euler_angles = _ivy.concatenate((gimbal_euler_angles_0, euler_angles_1, gimbal_euler_angles_2), -1)

    # BS x 1
    r13 = rot_mat[..., 0, 2:3]
    r12 = rot_mat[..., 0, 1:2]
    r31 = rot_mat[..., 2, 0:1]
    r21 = rot_mat[..., 1, 0:1]
    normal_euler_angles_0 = _ivy.atan2(r13, -r12)
    normal_euler_angles_2 = _ivy.atan2(r31, r21)

    # BS x 3
    normal_euler_angles = _ivy.concatenate((normal_euler_angles_0, euler_angles_1, normal_euler_angles_2), -1)

    return _ivy.where(gimbal_validity, normal_euler_angles, gimbal_euler_angles)


def _rot_mat_to_yxy_euler(rot_mat):
    # BS x 1
    euler_angles_1 = _ivy.acos(rot_mat[..., 1, 1:2])

    gimbal_validity = _ivy.abs(rot_mat[..., 0, 1:2]) > GIMBAL_TOL

    r13 = rot_mat[..., 0, 2:3]
    r11 = rot_mat[..., 0, 0:1]
    gimbal_euler_angles_0 = _ivy.atan2(r13, r11)
    gimbal_euler_angles_2 = _ivy.zeros_like(gimbal_euler_angles_0)

    # BS x 3
    gimbal_euler_angles = _ivy.concatenate((gimbal_euler_angles_0, euler_angles_1, gimbal_euler_angles_2), -1)

    # BS x 1
    r21 = rot_mat[..., 1, 0:1]
    r23 = rot_mat[..., 1, 2:3]
    r12 = rot_mat[..., 0, 1:2]
    r32 = rot_mat[..., 2, 1:2]
    normal_euler_angles_0 = _ivy.atan2(r21, -r23)
    normal_euler_angles_2 = _ivy.atan2(r12, r32)

    # BS x 3
    normal_euler_angles = _ivy.concatenate((normal_euler_angles_0, euler_angles_1, normal_euler_angles_2), -1)

    return _ivy.where(gimbal_validity, normal_euler_angles, gimbal_euler_angles)


def _rot_mat_to_zyz_euler(rot_mat):
    # BS x 1
    euler_angles_1 = _ivy.acos(rot_mat[..., 2, 2:3])

    gimbal_validity = _ivy.abs(rot_mat[..., 0, 2:3]) > GIMBAL_TOL

    r21 = rot_mat[..., 1, 0:1]
    r22 = rot_mat[..., 1, 1:2]
    gimbal_euler_angles_0 = _ivy.atan2(r21, r22)
    gimbal_euler_angles_2 = _ivy.zeros_like(gimbal_euler_angles_0)

    # BS x 3
    gimbal_euler_angles = _ivy.concatenate((gimbal_euler_angles_0, euler_angles_1, gimbal_euler_angles_2), -1)

    # BS x 1
    r32 = rot_mat[..., 2, 1:2]
    r31 = rot_mat[..., 2, 0:1]
    r23 = rot_mat[..., 1, 2:3]
    r13 = rot_mat[..., 0, 2:3]
    normal_euler_angles_0 = _ivy.atan2(r32, -r31)
    normal_euler_angles_2 = _ivy.atan2(r23, r13)

    # BS x 3
    normal_euler_angles = _ivy.concatenate((normal_euler_angles_0, euler_angles_1, normal_euler_angles_2), -1)

    return _ivy.where(gimbal_validity, normal_euler_angles, gimbal_euler_angles)


def _rot_mat_to_xyz_euler(rot_mat):
    # BS x 1
    euler_angles_1 = -_ivy.asin(rot_mat[..., 2, 0:1])

    gimbal_validity = _ivy.abs(rot_mat[..., 0, 0:1]) > GIMBAL_TOL

    r23 = rot_mat[..., 1, 2:3]
    r22 = rot_mat[..., 1, 1:2]
    gimbal_euler_angles_0 = _ivy.atan2(-r23, r22)
    gimbal_euler_angles_2 = _ivy.zeros_like(gimbal_euler_angles_0)

    # BS x 3
    gimbal_euler_angles = _ivy.concatenate((gimbal_euler_angles_0, euler_angles_1, gimbal_euler_angles_2), -1)

    # BS x 1
    r32 = rot_mat[..., 2, 1:2]
    r33 = rot_mat[..., 2, 2:3]
    r21 = rot_mat[..., 2, 0:1]
    r11 = rot_mat[..., 0, 0:1]
    normal_euler_angles_0 = _ivy.atan2(r32, r33)
    normal_euler_angles_2 = _ivy.atan2(r21, r11)

    # BS x 3
    normal_euler_angles = _ivy.concatenate((normal_euler_angles_0, euler_angles_1, normal_euler_angles_2), -1)

    return _ivy.where(gimbal_validity, normal_euler_angles, gimbal_euler_angles)


def _rot_mat_to_yzx_euler(rot_mat):
    # BS x 1
    euler_angles_1 = -_ivy.asin(rot_mat[..., 0, 1:2])

    gimbal_validity = _ivy.abs(rot_mat[..., 0, 0:1]) > GIMBAL_TOL

    r31 = rot_mat[..., 2, 0:1]
    r33 = rot_mat[..., 2, 2:3]
    gimbal_euler_angles_0 = _ivy.atan2(-r31, r33)
    gimbal_euler_angles_2 = _ivy.zeros_like(gimbal_euler_angles_0)

    # BS x 3
    gimbal_euler_angles = _ivy.concatenate((gimbal_euler_angles_0, euler_angles_1, gimbal_euler_angles_2), -1)

    # BS x 1
    r13 = rot_mat[..., 0, 2:3]
    r11 = rot_mat[..., 0, 0:1]
    r32 = rot_mat[..., 2, 1:2]
    r22 = rot_mat[..., 1, 1:2]
    normal_euler_angles_0 = _ivy.atan2(r13, r11)
    normal_euler_angles_2 = _ivy.atan2(r32, r22)

    # BS x 3
    normal_euler_angles = _ivy.concatenate((normal_euler_angles_0, euler_angles_1, normal_euler_angles_2), -1)

    return _ivy.where(gimbal_validity, normal_euler_angles, gimbal_euler_angles)


def _rot_mat_to_zxy_euler(rot_mat):
    # BS x 1
    euler_angles_1 = -_ivy.asin(rot_mat[..., 1, 2:3])

    gimbal_validity = _ivy.abs(rot_mat[..., 1, 0:1]) > GIMBAL_TOL

    r12 = rot_mat[..., 0, 1:2]
    r11 = rot_mat[..., 0, 0:1]
    gimbal_euler_angles_0 = _ivy.atan2(-r12, r11)
    gimbal_euler_angles_2 = _ivy.zeros_like(gimbal_euler_angles_0)

    # BS x 3
    gimbal_euler_angles = _ivy.concatenate((gimbal_euler_angles_0, euler_angles_1, gimbal_euler_angles_2), -1)

    # BS x 1
    r21 = rot_mat[..., 1, 0:1]
    r22 = rot_mat[..., 1, 1:2]
    r13 = rot_mat[..., 0, 2:3]
    r33 = rot_mat[..., 2, 0:1]
    normal_euler_angles_0 = _ivy.atan2(r21, r22)
    normal_euler_angles_2 = _ivy.atan2(r13, r33)

    # BS x 3
    normal_euler_angles = _ivy.concatenate((normal_euler_angles_0, euler_angles_1, normal_euler_angles_2), -1)

    return _ivy.where(gimbal_validity, normal_euler_angles, gimbal_euler_angles)


def _rot_mat_to_xzy_euler(rot_mat):
    # BS x 1
    euler_angles_1 = _ivy.asin(rot_mat[..., 1, 0:1])

    gimbal_validity = _ivy.abs(rot_mat[..., 0, 0:1]) > GIMBAL_TOL

    r32 = rot_mat[..., 2, 1:2]
    r33 = rot_mat[..., 2, 2:3]
    gimbal_euler_angles_0 = _ivy.atan2(r32, r33)
    gimbal_euler_angles_2 = _ivy.zeros_like(gimbal_euler_angles_0)

    # BS x 3
    gimbal_euler_angles = _ivy.concatenate((gimbal_euler_angles_0, euler_angles_1, gimbal_euler_angles_2), -1)

    # BS x 1
    r23 = rot_mat[..., 1, 2:3]
    r22 = rot_mat[..., 1, 1:2]
    r31 = rot_mat[..., 2, 0:1]
    r11 = rot_mat[..., 0, 0:1]
    normal_euler_angles_0 = _ivy.atan2(-r23, r22)
    normal_euler_angles_2 = _ivy.atan2(-r31, r11)

    # BS x 3
    normal_euler_angles = _ivy.concatenate((normal_euler_angles_0, euler_angles_1, normal_euler_angles_2), -1)

    return _ivy.where(gimbal_validity, normal_euler_angles, gimbal_euler_angles)


def _rot_mat_to_yxz_euler(rot_mat):
    # BS x 1
    euler_angles_1 = _ivy.asin(rot_mat[..., 2, 1:2])

    gimbal_validity = _ivy.abs(rot_mat[..., 1, 1:2]) > GIMBAL_TOL

    r13 = rot_mat[..., 0, 2:3]
    r11 = rot_mat[..., 0, 0:1]
    gimbal_euler_angles_0 = _ivy.atan2(r13, r11)
    gimbal_euler_angles_2 = _ivy.zeros_like(gimbal_euler_angles_0)

    # BS x 3
    gimbal_euler_angles = _ivy.concatenate((gimbal_euler_angles_0, euler_angles_1, gimbal_euler_angles_2), -1)

    # BS x 1
    r31 = rot_mat[..., 2, 0:1]
    r33 = rot_mat[..., 2, 2:3]
    r12 = rot_mat[..., 0, 1:2]
    r22 = rot_mat[..., 1, 1:2]
    normal_euler_angles_0 = _ivy.atan2(-r31, r33)
    normal_euler_angles_2 = _ivy.atan2(-r12, r22)

    # BS x 3
    normal_euler_angles = _ivy.concatenate((normal_euler_angles_0, euler_angles_1, normal_euler_angles_2), -1)

    return _ivy.where(gimbal_validity, normal_euler_angles, gimbal_euler_angles)


def _rot_mat_to_zyx_euler(rot_mat):
    # BS x 1
    euler_angles_1 = _ivy.asin(rot_mat[..., 0, 2:3])

    gimbal_validity = _ivy.abs(rot_mat[..., 1, 1:2]) > GIMBAL_TOL

    r21 = rot_mat[..., 1, 0:1]
    r22 = rot_mat[..., 1, 1:2]
    gimbal_euler_angles_0 = _ivy.atan2(r21, r22)
    gimbal_euler_angles_2 = _ivy.zeros_like(gimbal_euler_angles_0)

    # BS x 3
    gimbal_euler_angles = _ivy.concatenate((gimbal_euler_angles_0, euler_angles_1, gimbal_euler_angles_2), -1)

    # BS x 1
    r12 = rot_mat[..., 0, 1:2]
    r11 = rot_mat[..., 0, 0:1]
    r23 = rot_mat[..., 1, 2:3]
    r33 = rot_mat[..., 2, 2:3]
    normal_euler_angles_0 = _ivy.atan2(-r12, r11)
    normal_euler_angles_2 = _ivy.atan2(-r23, r33)

    # BS x 3
    normal_euler_angles = _ivy.concatenate((normal_euler_angles_0, euler_angles_1, normal_euler_angles_2), -1)

    return _ivy.where(gimbal_validity, normal_euler_angles, gimbal_euler_angles)


ROT_MAT_TO_EULER_DICT = {'xyx': _rot_mat_to_xyx_euler,
                         'yzy': _rot_mat_to_yzy_euler,
                         'zxz': _rot_mat_to_zxz_euler,
                         'xzx': _rot_mat_to_xzx_euler,
                         'yxy': _rot_mat_to_yxy_euler,
                         'zyz': _rot_mat_to_zyz_euler,
                         'xyz': _rot_mat_to_xyz_euler,
                         'yzx': _rot_mat_to_yzx_euler,
                         'zxy': _rot_mat_to_zxy_euler,
                         'xzy': _rot_mat_to_xzy_euler,
                         'yxz': _rot_mat_to_yxz_euler,
                         'zyx': _rot_mat_to_zyx_euler}


# GENERAL #
# --------#

def rot_mat_to_euler(rot_mat, convention='zyx'):
    """
    Convert rotation matrix :math:`\mathbf{R}\in\mathbb{R}^{3×3}` to :math:`zyx` Euler angles
    :math:`\mathbf{θ}_{xyz} = [ϕ_z, ϕ_y, ϕ_x]`.\n
    `[reference] <https://github.com/alisterburt/eulerangles/blob/master/eulerangles/conversions.py>`_

    :param rot_mat: Rotation matrix *[batch_shape,3,3]*
    :type rot_mat: array
    :param convention: The axes for euler rotation, in order of L.H.S. matrix multiplication.
    :type convention: str, optional`
    :return: zyx Euler angles *[batch_shape,3]*
    """
    try:
        return ROT_MAT_TO_EULER_DICT[convention](rot_mat)
    except KeyError:
        raise Exception('convention must one of: {}'.format(VALID_EULER_CONVENTIONS))


def quaternion_to_euler(quaternion, convention='zyx'):
    """
    Convert quaternion :math:`\mathbf{q} = [q_i, q_j, q_k, q_r]` to :math:`zyx` Euler angles
    :math:`\mathbf{θ}_{xyz} = [ϕ_z, ϕ_y, ϕ_x]`.\n
    `[reference] <https://github.com/alisterburt/eulerangles/blob/master/eulerangles/conversions.py>`_

    :param quaternion: Input quaternion *[batch_shape,4]*
    :type quaternion: array
    :param convention: The axes for euler rotation, in order of L.H.S. matrix multiplication.
    :type convention: str, optional`
    :return: zyx Euler angles *[batch_shape,3]*
    """

    # BS x 3
    return rot_mat_to_euler(_ivy_rot_mat.quaternion_to_rot_mat(quaternion), convention)


def axis_angle_to_euler(axis_angle, convention='zyx'):
    """
    Convert rotation axis unit vector :math:`\mathbf{e} = [e_x, e_y, e_z]` and
    rotation angle :math:`θ` to :math:`zyx` Euler angles :math:`\mathbf{θ}_{xyz} = [ϕ_z, ϕ_y, ϕ_x]`.

    :param axis_angle: Input axis_angle *[batch_shape,4]*
    :type axis_angle: array
    :param convention: The axes for euler rotation, in order of L.H.S. matrix multiplication.
    :type convention: str, optional`
    :return: zyx Euler angles *[batch_shape,3]*
    """

    # BS x 3
    return rot_mat_to_euler(_ivy_rot_mat.axis_angle_to_rot_mat(axis_angle), convention)


def get_random_euler(batch_shape=None):
    """
    Generate random :math:`zyx` Euler angles :math:`\mathbf{θ}_{xyz} = [ϕ_z, ϕ_y, ϕ_x]`.
    :param batch_shape: Shape of batch. Shape of [1] is assumed if None.
    :type batch_shape: sequence of ints, optional
    :return: Random euler *[batch_shape,3]*
    """

    if batch_shape is None:
        batch_shape = []

    # BS x 3
    return _ivy.random_uniform(0.0, _math.pi * 2, list(batch_shape) + [3])
