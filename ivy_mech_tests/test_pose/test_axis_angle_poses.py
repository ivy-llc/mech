"""
Collection of tests for axis-angle pose functions
"""

# global
import ivy
import ivy_mech
import numpy as np

# local
from ivy_mech_tests.test_pose.pose_data import PoseTestData

ptd = PoseTestData()


def test_euler_pose_to_axis_angle_pose(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.euler_pose_to_axis_angle_pose(ivy.array(ptd.euler_pose)), ptd.axis_angle_pose, atol=1e-6)
    assert np.allclose(ivy_mech.euler_pose_to_axis_angle_pose(ivy.array(ptd.batched_euler_pose))[0], ptd.axis_angle_pose, atol=1e-6)
    ivy.unset_backend()
    

def test_mat_pose_to_rot_vec_pose(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.mat_pose_to_rot_vec_pose(ivy.array(ptd.matrix_pose)), ptd.rot_vec_pose, atol=1e-6)
    assert np.allclose(ivy_mech.mat_pose_to_rot_vec_pose(ivy.array(ptd.batched_matrix_pose))[0], ptd.rot_vec_pose, atol=1e-6)
    ivy.unset_backend()


def test_quaternion_pose_to_rot_vec_pose(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.quaternion_pose_to_rot_vec_pose(ivy.array(ptd.quaternion_pose)), ptd.rot_vec_pose, atol=1e-6)
    assert np.allclose(ivy_mech.quaternion_pose_to_rot_vec_pose(ivy.array(ptd.batched_quaternion_pose))[0], ptd.rot_vec_pose, atol=1e-6)
    ivy.unset_backend()
