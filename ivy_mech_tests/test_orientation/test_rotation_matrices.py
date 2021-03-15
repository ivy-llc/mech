"""
Collection of tests for rotation matrix functions
"""

# global
import ivy_mech
import ivy.numpy
import numpy as np
import ivy.core.general as ivy_gen

# local
from ivy_mech_tests.test_orientation.orientation_data import OrientationTestData

otd = OrientationTestData()


def test_axis_angle_to_rot_mat(dev_str, call):
    assert np.allclose(call(ivy_mech.axis_angle_to_rot_mat, otd.axis_angle), otd.rotation_matrix, atol=1e-6)
    assert np.allclose(call(ivy_mech.axis_angle_to_rot_mat, otd.batched_axis_angle)[0],
                       otd.rotation_matrix, atol=1e-6)


def test_rot_vec_to_rot_mat(dev_str, call):
    assert np.allclose(call(ivy_mech.rot_vec_to_rot_mat, otd.rotation_vector), otd.rotation_matrix, atol=1e-6)
    assert np.allclose(call(ivy_mech.rot_vec_to_rot_mat, otd.batched_rotation_vector)[0],
                       otd.rotation_matrix, atol=1e-6)


def test_quaternion_to_rot_mat(dev_str, call):
    assert np.allclose(call(ivy_mech.quaternion_to_rot_mat, otd.quaternion), otd.rotation_matrix, atol=1e-6)
    assert np.allclose(call(ivy_mech.quaternion_to_rot_mat, otd.batched_quaternion)[0], otd.rotation_matrix, atol=1e-6)


def test_euler_to_rot_mat(dev_str, call):
    for conv in ivy_mech.VALID_EULER_CONVENTIONS:
        with ivy.numpy.use:
            rotation_matrix = ivy_mech.euler_to_rot_mat(otd.euler_angles, conv)
        assert np.allclose(call(ivy_mech.euler_to_rot_mat, otd.euler_angles, conv), rotation_matrix, atol=1e-6)
        assert np.allclose(call(ivy_mech.euler_to_rot_mat, otd.batched_euler_angles, conv)[0], rotation_matrix,
                           atol=1e-6)


def test_target_facing_rotation_vector(dev_str, call):
    assert np.allclose(call(ivy_mech.target_facing_rotation_matrix, ivy_gen.array([0., 0., 0.]),
                            ivy_gen.array([0., 1., 0.])),
                       np.array([[-1., 0., 0.],
                                 [0., 0., 1.],
                                 [0., 1., 0.]]), atol=1e-6)
    assert np.allclose(call(ivy_mech.target_facing_rotation_matrix, ivy_gen.array([[[0., 0., 0.]]]),
                            ivy_gen.array([[[0., 1., 0.]]])),
                       np.array([[[[-1., 0., 0.],
                                   [0., 0., 1.],
                                   [0., 1., 0.]]]]), atol=1e-6)


def test_get_random_rot_mat(dev_str, call):
    assert call(ivy_mech.get_random_rot_mat).shape == (3, 3)
    assert call(ivy_mech.get_random_rot_mat, batch_shape=(1, 1)).shape == (1, 1, 3, 3)
