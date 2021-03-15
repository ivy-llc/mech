"""
Collection of tests for quaternion pose functions
"""

# global
import ivy_mech
import numpy as np

# local
from ivy_mech_tests.test_pose.pose_data import PoseTestData

ptd = PoseTestData()


def test_axis_angle_pose_to_quaternion_pose(dev_str, call):
    assert np.allclose(call(ivy_mech.axis_angle_pose_to_quaternion_pose, ptd.axis_angle_pose), ptd.quaternion_pose, atol=1e-6)
    assert np.allclose(call(ivy_mech.axis_angle_pose_to_quaternion_pose, ptd.batched_axis_angle_pose)[0],
                       ptd.quaternion_pose, atol=1e-6)


def test_mat_pose_to_quaternion_pose(dev_str, call):
    assert np.allclose(call(ivy_mech.mat_pose_to_quaternion_pose, ptd.matrix_pose), ptd.quaternion_pose, atol=1e-6)
    assert np.allclose(call(ivy_mech.mat_pose_to_quaternion_pose, ptd.batched_matrix_pose)[0],
                       ptd.quaternion_pose, atol=1e-6)


def test_euler_pose_to_quaternion_pose(dev_str, call):
    assert np.allclose(call(ivy_mech.euler_pose_to_quaternion_pose, ptd.euler_pose), ptd.quaternion_pose, atol=1e-6)
    assert np.allclose(call(ivy_mech.euler_pose_to_quaternion_pose, ptd.batched_euler_pose)[0],
                       ptd.quaternion_pose, atol=1e-6)


def test_increment_quaternion_pose_with_velocity(dev_str, call):
    assert np.allclose(call(ivy_mech.increment_quaternion_pose_with_velocity, ptd.quaternion_pose, ptd.velocity,
                            ptd.control_dt), ptd.incremented_quaternion, atol=1e-6)
    assert np.allclose(call(ivy_mech.increment_quaternion_pose_with_velocity, ptd.batched_quaternion_pose,
                            ptd.batched_velocity, ptd.batched_control_dt)[0], ptd.incremented_quaternion, atol=1e-6)
