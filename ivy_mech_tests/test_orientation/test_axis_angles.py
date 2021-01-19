"""
Collection of tests for axis-angle functions
"""

# global
import ivy_mech
import numpy as np
import ivy_mech_tests.helpers as helpers

# local
from ivy_mech_tests.test_orientation.orientation_data import OrientationTestData

otd = OrientationTestData()


def test_quaternion_to_vector_and_angle():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        vector_and_angle = call(ivy_mech.quaternion_to_axis_angle, otd.quaternion)
        assert np.allclose(vector_and_angle, otd.axis_angle, atol=1e-6)
        vector_and_angle = call(ivy_mech.quaternion_to_axis_angle, otd.batched_quaternion)
        assert np.allclose(vector_and_angle, otd.axis_angle, atol=1e-6)


def test_quaternion_to_polar_axis_angle():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        assert np.allclose(call(ivy_mech.quaternion_to_polar_axis_angle, otd.quaternion),
                           otd.polar_axis_angle, atol=1e-6)
        assert np.allclose(call(ivy_mech.quaternion_to_polar_axis_angle, otd.batched_quaternion)[0],
                           otd.polar_axis_angle, atol=1e-6)


def test_quaternion_to_rotation_vector():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        assert np.allclose(call(ivy_mech.quaternion_to_rotation_vector, otd.quaternion, dev='cpu'), otd.rotation_vector, atol=1e-6)
        assert np.allclose(call(ivy_mech.quaternion_to_rotation_vector, otd.batched_quaternion, dev='cpu')[0],
                           otd.rotation_vector, atol=1e-6)
