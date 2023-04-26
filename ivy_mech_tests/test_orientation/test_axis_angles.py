"""
Collection of tests for axis-angle functions
"""

# global
import ivy
import ivy_mech
import numpy as np

# local
from ivy_mech_tests.test_orientation.orientation_data import OrientationTestData

otd = OrientationTestData()


def test_rot_mat_to_axis_angle(device, fw):
    assert np.allclose(
        ivy_mech.rot_mat_to_axis_angle(ivy.array(otd.rotation_matrix)),
        otd.axis_angle,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_mech.rot_mat_to_axis_angle(ivy.array(otd.batched_rotation_matrix))[0],
        otd.axis_angle,
        atol=1e-6,
    )


def test_euler_to_axis_angle(device, fw):
    assert np.allclose(
        ivy_mech.euler_to_axis_angle(ivy.array(otd.euler_angles)),
        otd.axis_angle,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_mech.euler_to_axis_angle(ivy.array(otd.batched_euler_angles))[0],
        otd.axis_angle,
        atol=1e-6,
    )


def test_quaternion_to_vector_and_angle(device, fw):
    vector_and_angle = ivy_mech.quaternion_to_axis_angle(ivy.array(otd.quaternion))
    assert np.allclose(vector_and_angle, otd.axis_angle, atol=1e-6)
    vector_and_angle = ivy_mech.quaternion_to_axis_angle(
        ivy.array(otd.batched_quaternion)
    )
    assert np.allclose(vector_and_angle, otd.axis_angle, atol=1e-6)


def test_quaternion_to_polar_axis_angle(device, fw):
    assert np.allclose(
        ivy_mech.quaternion_to_polar_axis_angle(ivy.array(otd.quaternion)),
        otd.polar_axis_angle,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_mech.quaternion_to_polar_axis_angle(ivy.array(otd.batched_quaternion))[0],
        otd.polar_axis_angle,
        atol=1e-6,
    )


def test_quaternion_to_rotation_vector(device, fw):
    assert np.allclose(
        ivy_mech.quaternion_to_rotation_vector(ivy.array(otd.quaternion), device="cpu"),
        otd.rotation_vector,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_mech.quaternion_to_rotation_vector(
            ivy.array(otd.batched_quaternion), device="cpu"
        )[0],
        otd.rotation_vector,
        atol=1e-6,
    )


def test_get_random_axis_angle(device, fw):
    assert ivy_mech.get_random_axis_angle().shape == (4,)
    assert ivy_mech.get_random_axis_angle(batch_shape=(1, 1)).shape == (1, 1, 4)
