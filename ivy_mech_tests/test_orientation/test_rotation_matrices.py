"""
Collection of tests for rotation matrix functions
"""

# global
import ivy_mech
import ivy.functional.backends.numpy as ivy_np
import numpy as np
import ivy

# local
from ivy_mech_tests.test_orientation.orientation_data import OrientationTestData

otd = OrientationTestData()


def test_axis_angle_to_rot_mat(device, fw):
    assert np.allclose(
        ivy_mech.axis_angle_to_rot_mat(ivy.array(otd.axis_angle)),
        otd.rotation_matrix,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_mech.axis_angle_to_rot_mat(ivy.array(otd.batched_axis_angle))[0],
        otd.rotation_matrix,
        atol=1e-6,
    )


def test_rot_vec_to_rot_mat(device, fw):
    assert np.allclose(
        ivy_mech.rot_vec_to_rot_mat(ivy.array(otd.rotation_vector)),
        otd.rotation_matrix,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_mech.rot_vec_to_rot_mat(ivy.array(otd.batched_rotation_vector))[0],
        otd.rotation_matrix,
        atol=1e-6,
    )


def test_quaternion_to_rot_mat(device, fw):
    assert np.allclose(
        ivy_mech.quaternion_to_rot_mat(ivy.array(otd.quaternion)),
        otd.rotation_matrix,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_mech.quaternion_to_rot_mat(ivy.array(otd.batched_quaternion))[0],
        otd.rotation_matrix,
        atol=1e-6,
    )


def test_euler_to_rot_mat(device, fw):
    for conv in ivy_mech.VALID_EULER_CONVENTIONS:
        with ivy_np.use:
            rotation_matrix = ivy_mech.euler_to_rot_mat(otd.euler_angles, conv)
        assert np.allclose(
            ivy_mech.euler_to_rot_mat(ivy.array(otd.euler_angles), conv),
            rotation_matrix,
            atol=1e-6,
        )
        assert np.allclose(
            ivy_mech.euler_to_rot_mat(ivy.array(otd.batched_euler_angles), conv)[0],
            rotation_matrix,
            atol=1e-6,
        )


def test_target_facing_rotation_vector(device, fw):
    assert np.allclose(
        ivy_mech.target_facing_rotation_matrix(
            ivy.array([0.0, 0.0, 0.0]), ivy.array([0.0, 1.0, 0.0])
        ),
        np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        atol=1e-6,
    )
    assert np.allclose(
        ivy_mech.target_facing_rotation_matrix(
            ivy.array([[[0.0, 0.0, 0.0]]]), ivy.array([[[0.0, 1.0, 0.0]]])
        ),
        np.array([[[[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
        atol=1e-6,
    )


def test_get_random_rot_mat(device, fw):
    assert ivy_mech.get_random_rot_mat().shape == (3, 3)
    assert ivy_mech.get_random_rot_mat(batch_shape=(1, 1)).shape == (1, 1, 3, 3)
