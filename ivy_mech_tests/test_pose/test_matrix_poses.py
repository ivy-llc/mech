"""
Collection of tests for matrix pose functions
"""

# global
import ivy_mech
import ivy.numpy
import numpy as np
import ivy_mech_tests.helpers as helpers
from ivy.framework_handler import set_framework, unset_framework

# local
from ivy_mech_tests.test_pose.pose_data import PoseTestData

ptd = PoseTestData()


def test_axis_angle_pose_to_mat_pose():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        set_framework(lib)
        assert np.allclose(call(ivy_mech.axis_angle_pose_to_mat_pose, ptd.axis_angle_pose), ptd.matrix_pose, atol=1e-6)
        assert np.allclose(call(ivy_mech.axis_angle_pose_to_mat_pose, ptd.batched_axis_angle_pose)[0],
                           ptd.matrix_pose, atol=1e-6)
        unset_framework()


def test_quaternion_pose_to_mat_pose():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        set_framework(lib)
        assert np.allclose(call(ivy_mech.quaternion_pose_to_mat_pose, ptd.quaternion_pose), ptd.matrix_pose, atol=1e-6)
        assert np.allclose(call(ivy_mech.quaternion_pose_to_mat_pose, ptd.batched_quaternion_pose)[0],
                           ptd.matrix_pose, atol=1e-6)
        unset_framework()


def test_euler_pose_to_mat_pose():
    with ivy.numpy.use:
        matrix_pose = ivy_mech.euler_pose_to_mat_pose(ptd.euler_pose)
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        set_framework(lib)
        assert np.allclose(call(ivy_mech.euler_pose_to_mat_pose, ptd.euler_pose), matrix_pose, atol=1e-6)
        assert np.allclose(call(ivy_mech.euler_pose_to_mat_pose, ptd.batched_euler_pose)[0], matrix_pose, atol=1e-6)
        unset_framework()


def test_rot_vec_pose_to_mat_pose():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        set_framework(lib)
        assert np.allclose(call(ivy_mech.rot_vec_pose_to_mat_pose, ptd.rot_vec_pose), ptd.matrix_pose, atol=1e-6)
        assert np.allclose(call(ivy_mech.rot_vec_pose_to_mat_pose, ptd.batched_rot_vec_pose)[0], ptd.matrix_pose, atol=1e-6)
        unset_framework()
