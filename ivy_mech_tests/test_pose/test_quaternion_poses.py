"""Collection of tests for quaternion pose functions"""
# global
import ivy
import ivy_mech
import numpy as np

# local
from ivy_mech_tests.test_pose.pose_data import PoseTestData

ptd = PoseTestData()


def test_axis_angle_pose_to_quaternion_pose(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_mech.axis_angle_pose_to_quaternion_pose(ivy.array(ptd.axis_angle_pose)),
        ptd.quaternion_pose,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_mech.axis_angle_pose_to_quaternion_pose(
            ivy.array(ptd.batched_axis_angle_pose)
        )[0],
        ptd.quaternion_pose,
        atol=1e-6,
    )
    ivy.previous_backend()


def test_mat_pose_to_quaternion_pose(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_mech.mat_pose_to_quaternion_pose(ivy.array(ptd.matrix_pose)),
        ptd.quaternion_pose,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_mech.mat_pose_to_quaternion_pose(ivy.array(ptd.batched_matrix_pose))[0],
        ptd.quaternion_pose,
        atol=1e-6,
    )
    ivy.previous_backend()


def test_euler_pose_to_quaternion_pose(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_mech.euler_pose_to_quaternion_pose(ivy.array(ptd.euler_pose)),
        ptd.quaternion_pose,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_mech.euler_pose_to_quaternion_pose(ivy.array(ptd.batched_euler_pose))[0],
        ptd.quaternion_pose,
        atol=1e-6,
    )
    ivy.previous_backend()


def test_increment_quaternion_pose_with_velocity(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_mech.increment_quaternion_pose_with_velocity(
            ivy.array(ptd.quaternion_pose),
            ivy.array(ptd.velocity),
            ivy.array(ptd.control_dt),
        ),
        ptd.incremented_quaternion,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_mech.increment_quaternion_pose_with_velocity(
            ivy.array(ptd.batched_quaternion_pose),
            ivy.array(ptd.batched_velocity),
            ivy.array(ptd.batched_control_dt),
        )[0],
        ptd.incremented_quaternion,
        atol=1e-6,
    )
    ivy.previous_backend()
