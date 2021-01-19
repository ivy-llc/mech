"""
Collection of tests for euler functions
"""

# global
import ivy_mech
import numpy as np
import ivy_mech_tests.helpers as helpers

# local
from ivy_mech_tests.test_orientation.orientation_data import OrientationTestData

otd = OrientationTestData()


def test_rot_mat_to_euler():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        for conv in ivy_mech.VALID_EULER_CONVENTIONS:
            euler_angles = ivy_mech.rot_mat_to_euler(otd.rotation_matrix, conv, f=helpers._ivy_np)
            assert np.allclose(call(ivy_mech.rot_mat_to_euler, otd.rotation_matrix, conv),
                               euler_angles, atol=1e-6)
            assert np.allclose(call(ivy_mech.rot_mat_to_euler, otd.batched_rotation_matrix, conv)[0],
                               euler_angles, atol=1e-6)


def test_quaternion_to_euler():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        for conv in ivy_mech.VALID_EULER_CONVENTIONS:
            euler_angles = ivy_mech.quaternion_to_euler(otd.quaternion, conv, f=helpers._ivy_np)
            assert np.allclose(call(ivy_mech.quaternion_to_euler, otd.quaternion, conv), euler_angles, atol=1e-6)
            assert np.allclose(call(ivy_mech.quaternion_to_euler, otd.batched_quaternion, conv)[0], euler_angles,
                               atol=1e-6)
