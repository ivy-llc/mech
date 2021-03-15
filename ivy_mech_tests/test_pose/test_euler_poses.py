"""
Collection of tests for euler pose functions
"""

# global
import ivy_mech
import ivy.numpy
import numpy as np

# local
from ivy_mech_tests.test_pose.pose_data import PoseTestData

ptd = PoseTestData()


def test_mat_pose_to_euler_pose(dev_str, call):
    for conv in ivy_mech.VALID_EULER_CONVENTIONS:
        with ivy.numpy.use:
            euler_pose = ivy_mech.mat_pose_to_euler_pose(ptd.matrix_pose, conv)
        assert np.allclose(call(ivy_mech.mat_pose_to_euler_pose, ptd.matrix_pose, conv), euler_pose, atol=1e-6)
        assert np.allclose(call(ivy_mech.mat_pose_to_euler_pose, ptd.batched_matrix_pose, conv)[0], euler_pose,
                           atol=1e-6)


def test_quaternion_pose_to_euler_pose(dev_str, call):
    for conv in ivy_mech.VALID_EULER_CONVENTIONS:
        with ivy.numpy.use:
            euler_pose = ivy_mech.mat_pose_to_euler_pose(ptd.matrix_pose, conv)
        assert np.allclose(call(ivy_mech.quaternion_pose_to_euler_pose, ptd.quaternion_pose, conv), euler_pose, atol=1e-6)
        assert np.allclose(call(ivy_mech.quaternion_pose_to_euler_pose, ptd.batched_quaternion_pose, conv)[0], euler_pose,
                           atol=1e-6)


def test_axis_angle_pose_to_euler_pose(dev_str, call):
    for conv in ivy_mech.VALID_EULER_CONVENTIONS:
        with ivy.numpy.use:
            euler_pose = ivy_mech.mat_pose_to_euler_pose(ptd.matrix_pose, conv)
        assert np.allclose(call(ivy_mech.axis_angle_pose_to_euler_pose, ptd.axis_angle_pose, conv), euler_pose, atol=1e-6)
        assert np.allclose(call(ivy_mech.axis_angle_pose_to_euler_pose, ptd.batched_axis_angle_pose, conv)[0], euler_pose,
                           atol=1e-6)
