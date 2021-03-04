"""
Collection of tests for euler functions
"""

# global
import ivy_mech
import ivy.numpy
import numpy as np
import ivy_mech_tests.helpers as helpers
from ivy.framework_handler import set_framework, unset_framework

# local
from ivy_mech_tests.test_orientation.orientation_data import OrientationTestData

otd = OrientationTestData()


def test_rot_mat_to_euler():
    for lib, call in helpers.calls:
        set_framework(lib)
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        for conv in ivy_mech.VALID_EULER_CONVENTIONS:
            with ivy.numpy.use:
                euler_angles = ivy_mech.rot_mat_to_euler(otd.rotation_matrix, conv)
            assert np.allclose(call(ivy_mech.rot_mat_to_euler, otd.rotation_matrix, conv),
                               euler_angles, atol=1e-6)
            assert np.allclose(call(ivy_mech.rot_mat_to_euler, otd.batched_rotation_matrix, conv)[0],
                               euler_angles, atol=1e-6)
        unset_framework()


def test_quaternion_to_euler():
    for lib, call in helpers.calls:
        set_framework(lib)
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        for conv in ivy_mech.VALID_EULER_CONVENTIONS:
            with ivy.numpy.use:
                euler_angles = ivy_mech.quaternion_to_euler(otd.quaternion, conv)
            assert np.allclose(call(ivy_mech.quaternion_to_euler, otd.quaternion, conv), euler_angles, atol=1e-6)
            assert np.allclose(call(ivy_mech.quaternion_to_euler, otd.batched_quaternion, conv)[0], euler_angles,
                               atol=1e-6)
        unset_framework()


def test_axis_angle_to_euler():
    for lib, call in helpers.calls:
        set_framework(lib)
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        for conv in ivy_mech.VALID_EULER_CONVENTIONS:
            with ivy.numpy.use:
                euler_angles = ivy_mech.quaternion_to_euler(otd.quaternion, conv)
            assert np.allclose(call(ivy_mech.axis_angle_to_euler, otd.axis_angle, conv), euler_angles, atol=1e-6)
            assert np.allclose(call(ivy_mech.axis_angle_to_euler, otd.batched_axis_angle, conv)[0], euler_angles,
                               atol=1e-6)
        unset_framework()


def test_get_random_euler():
    for lib, call in helpers.calls:
        set_framework(lib)
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        assert call(ivy_mech.get_random_euler).shape == (3,)
        assert call(ivy_mech.get_random_euler, batch_shape=(1, 1)).shape == (1, 1, 3)
        unset_framework()
