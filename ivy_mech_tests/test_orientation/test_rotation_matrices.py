"""
Collection of tests for rotation matrix functions
"""

# global
import ivy_mech
import numpy as np
import ivy_mech_tests.helpers as helpers
import ivy.core.general as ivy_gen

# local
from ivy_mech_tests.test_orientation.orientation_data import OrientationTestData

otd = OrientationTestData()


def test_rot_vec_to_rot_mat():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        assert np.allclose(call(ivy_mech.rot_vec_to_rot_mat, otd.rotation_vector), otd.rotation_matrix, atol=1e-6)
        assert np.allclose(call(ivy_mech.rot_vec_to_rot_mat, otd.batched_rotation_vector)[0],
                           otd.rotation_matrix, atol=1e-6)


def test_quaternion_to_rot_mat():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        assert np.allclose(call(ivy_mech.quaternion_to_rot_mat, otd.quaternion), otd.rotation_matrix, atol=1e-6)
        assert np.allclose(call(ivy_mech.quaternion_to_rot_mat, otd.batched_quaternion)[0], otd.rotation_matrix, atol=1e-6)


def test_euler_to_rot_mat():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        for conv in ivy_mech.VALID_EULER_CONVENTIONS:
            rotation_matrix = ivy_mech.euler_to_rot_mat(otd.euler_angles, conv, f=helpers._ivy_np)
            assert np.allclose(call(ivy_mech.euler_to_rot_mat, otd.euler_angles, conv), rotation_matrix, atol=1e-6)
            assert np.allclose(call(ivy_mech.euler_to_rot_mat, otd.batched_euler_angles, conv)[0], rotation_matrix,
                               atol=1e-6)


def test_target_facing_rotation_vector():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        assert np.allclose(call(ivy_mech.target_facing_rotation_matrix, ivy_gen.array([0., 0., 0.], f=lib),
                                ivy_gen.array([0., 1., 0.], f=lib)),
                           np.array([[-1., 0., 0.],
                                     [0., 0., 1.],
                                     [0., 1., 0.]]), atol=1e-6)
        assert np.allclose(call(ivy_mech.target_facing_rotation_matrix, ivy_gen.array([[[0., 0., 0.]]], f=lib),
                                ivy_gen.array([[[0., 1., 0.]]], f=lib)),
                           np.array([[[[-1., 0., 0.],
                                       [0., 0., 1.],
                                       [0., 1., 0.]]]]), atol=1e-6)
