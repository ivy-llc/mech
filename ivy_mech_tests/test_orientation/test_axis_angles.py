"""
Collection of tests for axis-angle functions
"""

# global
import ivy_mech
import numpy as np

# local
from ivy_mech_tests.test_orientation.orientation_data import OrientationTestData

otd = OrientationTestData()


def test_rot_mat_to_axis_angle(device, call):
    assert np.allclose(call(ivy_mech.rot_mat_to_axis_angle, otd.rotation_matrix),
                       otd.axis_angle, atol=1e-6)
    assert np.allclose(call(ivy_mech.rot_mat_to_axis_angle, otd.batched_rotation_matrix)[0],
                       otd.axis_angle, atol=1e-6)


def test_euler_to_axis_angle(device, call):
    assert np.allclose(call(ivy_mech.euler_to_axis_angle, otd.euler_angles),
                       otd.axis_angle, atol=1e-6)
    assert np.allclose(call(ivy_mech.euler_to_axis_angle, otd.batched_euler_angles)[0],
                       otd.axis_angle, atol=1e-6)


def test_quaternion_to_vector_and_angle(device, call):
    vector_and_angle = call(ivy_mech.quaternion_to_axis_angle, otd.quaternion)
    assert np.allclose(vector_and_angle, otd.axis_angle, atol=1e-6)
    vector_and_angle = call(ivy_mech.quaternion_to_axis_angle, otd.batched_quaternion)
    assert np.allclose(vector_and_angle, otd.axis_angle, atol=1e-6)


def test_quaternion_to_polar_axis_angle(device, call):
    assert np.allclose(call(ivy_mech.quaternion_to_polar_axis_angle, otd.quaternion),
                       otd.polar_axis_angle, atol=1e-6)
    assert np.allclose(call(ivy_mech.quaternion_to_polar_axis_angle, otd.batched_quaternion)[0],
                       otd.polar_axis_angle, atol=1e-6)


def test_quaternion_to_rotation_vector(device, call):
    assert np.allclose(call(ivy_mech.quaternion_to_rotation_vector, otd.quaternion, device='cpu'),
                       otd.rotation_vector, atol=1e-6)
    assert np.allclose(call(ivy_mech.quaternion_to_rotation_vector, otd.batched_quaternion, device='cpu')[0],
                       otd.rotation_vector, atol=1e-6)


def test_get_random_axis_angle(device, call):
    assert call(ivy_mech.get_random_axis_angle).shape == (4,)
    assert call(ivy_mech.get_random_axis_angle, batch_shape=(1, 1)).shape == (1, 1, 4)
