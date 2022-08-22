"""
Collection of tests for euler functions
"""

# global
import ivy
import ivy_mech
import ivy.functional.backends.numpy as ivy_np
import numpy as np

# local
from ivy_mech_tests.test_orientation.orientation_data import OrientationTestData

otd = OrientationTestData()


def test_rot_mat_to_euler(device, fw):
    for conv in ivy_mech.VALID_EULER_CONVENTIONS:
        with ivy_np.use:
            euler_angles = ivy_mech.rot_mat_to_euler(otd.rotation_matrix, conv)
        ivy.set_backend(fw)
        assert np.allclose(ivy_mech.rot_mat_to_euler(otd.rotation_matrix, conv),
                           euler_angles, atol=1e-6)
        assert np.allclose(ivy_mech.rot_mat_to_euler(otd.batched_rotation_matrix, conv)[0],
                           euler_angles, atol=1e-6)
        ivy.unset_backend()


def test_quaternion_to_euler(device, fw):
    for conv in ivy_mech.VALID_EULER_CONVENTIONS:
        with ivy_np.use:
            euler_angles = ivy_mech.quaternion_to_euler(otd.quaternion, conv)
        ivy.set_backend(fw)
        assert np.allclose(ivy_mech.quaternion_to_euler(otd.quaternion, conv), euler_angles, atol=1e-6)
        assert np.allclose(ivy_mech.quaternion_to_euler(otd.batched_quaternion, conv)[0], euler_angles,
                           atol=1e-6)
        ivy.unset_backend()


def test_axis_angle_to_euler(device, fw):
    for conv in ivy_mech.VALID_EULER_CONVENTIONS:
        with ivy_np.use:
            euler_angles = ivy_mech.quaternion_to_euler(otd.quaternion, conv)
        ivy.set_backend(fw)
        assert np.allclose(ivy_mech.axis_angle_to_euler(otd.axis_angle, conv), euler_angles, atol=1e-6)
        assert np.allclose(ivy_mech.axis_angle_to_euler(otd.batched_axis_angle, conv)[0], euler_angles,
                           atol=1e-6)
        ivy.unset_backend()


def test_get_random_euler(device, fw):
    ivy.set_backend(fw)
    assert ivy_mech.get_random_euler().shape == (3,)
    assert ivy_mech.get_random_euler(batch_shape=(1, 1)).shape == (1, 1, 3)
    ivy.unset_backend()
