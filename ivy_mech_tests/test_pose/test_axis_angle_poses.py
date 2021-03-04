"""
Collection of tests for axis-angle pose functions
"""

# global
import ivy_mech
import numpy as np
import ivy_mech_tests.helpers as helpers
from ivy.framework_handler import set_framework, unset_framework

# local
from ivy_mech_tests.test_pose.pose_data import PoseTestData

ptd = PoseTestData()


def test_euler_pose_to_axis_angle_pose():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        set_framework(lib)
        assert np.allclose(call(ivy_mech.euler_pose_to_axis_angle_pose, ptd.euler_pose), ptd.axis_angle_pose, atol=1e-6)
        assert np.allclose(call(ivy_mech.euler_pose_to_axis_angle_pose, ptd.batched_euler_pose)[0], ptd.axis_angle_pose, atol=1e-6)
        unset_framework()


def test_mat_pose_to_rot_vec_pose():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        set_framework(lib)
        assert np.allclose(call(ivy_mech.mat_pose_to_rot_vec_pose, ptd.matrix_pose), ptd.rot_vec_pose, atol=1e-6)
        assert np.allclose(call(ivy_mech.mat_pose_to_rot_vec_pose, ptd.batched_matrix_pose)[0], ptd.rot_vec_pose, atol=1e-6)
        unset_framework()


def test_quaternion_pose_to_rot_vec_pose():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        set_framework(lib)
        assert np.allclose(call(ivy_mech.quaternion_pose_to_rot_vec_pose, ptd.quaternion_pose), ptd.rot_vec_pose, atol=1e-6)
        assert np.allclose(call(ivy_mech.quaternion_pose_to_rot_vec_pose, ptd.batched_quaternion_pose)[0], ptd.rot_vec_pose, atol=1e-6)
        unset_framework()
