"""
Collection of tests for matrix pose functions
"""

# global
import ivy
import ivy_mech
import ivy.functional.backends.numpy as ivy_np
import numpy as np

# local
from ivy_mech_tests.test_pose.pose_data import PoseTestData

ptd = PoseTestData()


def test_axis_angle_pose_to_mat_pose(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.axis_angle_pose_to_mat_pose(ivy.array(ptd.axis_angle_pose)), ptd.matrix_pose, atol=1e-6)
    assert np.allclose(ivy_mech.axis_angle_pose_to_mat_pose(ivy.array(ptd.batched_axis_angle_pose))[0],
                       ptd.matrix_pose, atol=1e-6)
    ivy.previous_backend()
    

def test_quaternion_pose_to_mat_pose(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.quaternion_pose_to_mat_pose(ivy.array(ptd.quaternion_pose)), ptd.matrix_pose, atol=1e-6)
    assert np.allclose(ivy_mech.quaternion_pose_to_mat_pose(ivy.array(ptd.batched_quaternion_pose))[0],
                       ptd.matrix_pose, atol=1e-6)
    ivy.previous_backend()


def test_euler_pose_to_mat_pose(device, fw):
    with ivy_np.use:
        matrix_pose = ivy_mech.euler_pose_to_mat_pose(ptd.euler_pose)
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.euler_pose_to_mat_pose(ivy.array(ptd.euler_pose)), matrix_pose, atol=1e-6)
    assert np.allclose(ivy_mech.euler_pose_to_mat_pose(ivy.array(ptd.batched_euler_pose))[0], matrix_pose, atol=1e-6)
    ivy.previous_backend()


def test_rot_vec_pose_to_mat_pose(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.rot_vec_pose_to_mat_pose(ivy.array(ptd.rot_vec_pose)), ptd.matrix_pose, atol=1e-6)
    assert np.allclose(ivy_mech.rot_vec_pose_to_mat_pose(ivy.array(ptd.batched_rot_vec_pose))[0], ptd.matrix_pose, atol=1e-6)
    ivy.previous_backend()
