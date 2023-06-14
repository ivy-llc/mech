"""Collection of Pose Conversion Functions to Matrix Format"""
# global
import ivy

# local
from ivy_mech.orientation import rotation_matrix as ivy_rot_mat
from ivy_mech.orientation import quaternion as ivy_quat


def quaternion_pose_to_mat_pose(quat_pose):
    r"""Convert quaternion pose
    :math:`\mathbf{p}_{q} = [\mathbf{x}_c, \mathbf{q}] = [x, y, z, q_i, q_j, q_k, q_r]` to
    matrix pose :math:`\mathbf{P}\in\mathbb{R}^{3×4}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Conversion_to_and_from_the_matrix_representation>`_ # noqa

    Parameters
    ----------
    quat_pose
        Quaternion pose *[batch_shape,7]*

    Returns
    -------
    ret
        Matrix pose *[batch_shape,3,4]*

    """
    # BS x 3 x 3
    rot_mat = ivy_rot_mat.quaternion_to_rot_mat(quat_pose[..., 3:])

    # BS x 3 x 1
    rhs = ivy.expand_dims(quat_pose[..., 0:3], axis=-1)

    # BS x 3 x 4
    return ivy.concat([rot_mat, rhs], axis=-1)


def euler_pose_to_mat_pose(euler_pose, convention="zyx", batch_shape=None):
    r"""Convert :math: Euler angle pose
    :math:`\mathbf{p}_{abc} = [\mathbf{x}_c, \mathbf{θ}_{xyz}] = [x, y, z, ϕ_a, ϕ_b, ϕ_c]` to matrix pose # noqa
    :math:`\mathbf{P}\in\mathbb{R}^{3×4}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix>`_

    Parameters
    ----------
    euler_pose
        Euler angle pose *[batch_shape,6]*
    convention
        The axes for euler rotation, in order of L.H.S. matrix multiplication.
        (Default value = 'zyx')
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)

    Returns
    -------
    ret
        Matrix pose *[batch_shape,3,4]*

    """
    if batch_shape is None:
        batch_shape = euler_pose.shape[:-1]

    # BS x 3 x 3
    rot_mat = ivy_rot_mat.euler_to_rot_mat(euler_pose[..., 3:], convention, batch_shape)

    # BS x 3 x 4
    return ivy.concat(
        [rot_mat, ivy.expand_dims(euler_pose[..., 0:3], axis=-1)], axis=-1
    )


def rot_vec_pose_to_mat_pose(rot_vec_pose):
    r"""Convert rotation vector pose
    :math:`\mathbf{p}_{rv} = [\mathbf{x}_c, \mathbf{θ}_{rv}] = [x, y, z, θe_x, θe_y, θe_z]`
    to matrix pose :math:`\mathbf{P}\in\mathbb{R}^{3×4}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_axis_and_angle_(rotation_vector)>`_ # noqa

    Parameters
    ----------
    rot_vec_pose
        Rotation vector pose *[batch_shape,6]*

    Returns
    -------
    ret
        Matrix pose *[batch_shape,3,4]*

    """
    # BS x 4
    quaternion = ivy_quat.rotation_vector_to_quaternion(rot_vec_pose[..., 3:])

    # BS x 3 x 3
    rot_mat = ivy_rot_mat.quaternion_to_rot_mat(quaternion)

    # BS x 3 x 4
    return ivy.concat(
        [rot_mat, ivy.expand_dims(rot_vec_pose[..., 0:3], axis=-1)], axis=-1
    )


def axis_angle_pose_to_mat_pose(axis_angle_pose):
    r"""Convert axis-angle pose
    :math:`\mathbf{p}_{aa} = [\mathbf{x}_c, \mathbf{e}, θ] = [x, y, z, e_x, e_y, e_z, θ]` to # noqa
    matrix pose :math:`\mathbf{P}\in\mathbb{R}^{3×4}`.

    Parameters
    ----------
    axis_angle_pose
        Quaternion pose *[batch_shape,7]*

    Returns
    -------
    ret
        Matrix pose *[batch_shape,3,4]*

    """
    # BS x 3 x 3
    rot_mat = ivy_rot_mat.axis_angle_to_rot_mat(axis_angle_pose[..., 3:])

    # BS x 3 x 4
    return ivy.concat(
        [rot_mat, ivy.expand_dims(axis_angle_pose[..., :3], axis=-1)], axis=-1
    )
