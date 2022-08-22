"""
Collection of tests for quaternion functions
"""

# global
import ivy
import ivy_mech
import ivy.functional.backends.numpy as ivy_np
import numpy as np

# local
from ivy_mech_tests.test_orientation.quaternion_data import QuaternionTestData

qtd = QuaternionTestData()


# Representation Conversions #
# ---------------------------#

def test_vector_and_angle_to_quaternion(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.axis_angle_to_quaternion(qtd.axis_angle), qtd.quaternion, atol=1e-6)
    assert np.allclose(ivy_mech.axis_angle_to_quaternion(qtd.batched_axis_angle)[0], qtd.quaternion, atol=1e-6)
    ivy.unset_backend()
    

def test_axis_angle_to_quaternion(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.polar_axis_angle_to_quaternion(qtd.polar_axis_angle), qtd.quaternion, atol=1e-6)
    assert np.allclose(ivy_mech.polar_axis_angle_to_quaternion(qtd.batched_polar_axis_angle)[0], qtd.quaternion, atol=1e-6)
    ivy.unset_backend()


def test_rot_mat_to_quaternion(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.rot_mat_to_quaternion(qtd.rotation_matrix), qtd.quaternion, atol=1e-6)
    assert np.allclose(ivy_mech.rot_mat_to_quaternion(qtd.batched_rotation_matrix)[0], qtd.quaternion, atol=1e-6)
    ivy.unset_backend()


def test_euler_to_quaternion(device, fw):
    for conv in ivy_mech.VALID_EULER_CONVENTIONS:
        with ivy_np.use:
            quaternion = ivy_mech.euler_to_quaternion(qtd.euler_angles, conv)
        ivy.set_backend(fw)
        assert np.allclose(ivy_mech.euler_to_quaternion(qtd.euler_angles, conv), quaternion, atol=1e-6)
        assert np.allclose(ivy_mech.euler_to_quaternion(qtd.batched_euler_angles, conv)[0], quaternion,
                           atol=1e-6)
        ivy.unset_backend()


def test_rotation_vector_to_quaternion(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.rotation_vector_to_quaternion(qtd.rotation_vector), qtd.quaternion, atol=1e-6)
    assert np.allclose(ivy_mech.rotation_vector_to_quaternion(qtd.batched_rotation_vector)[0],
                       qtd.quaternion, atol=1e-6)
    ivy.unset_backend()


# Quaternion Operations #
# ----------------------#

def test_inverse_quaternion(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.inverse_quaternion(
                            ivy_mech.inverse_quaternion(qtd.quaternion)),
                       qtd.quaternion, atol=1e-6)
    ivy.unset_backend()


def test_get_random_quaternion(device, fw):
    ivy.set_backend(fw)
    assert ivy_mech.get_random_quaternion().shape == (4,)
    assert ivy_mech.get_random_quaternion(batch_shape=(1, 1)).shape == (1, 1, 4)
    ivy.unset_backend()


def test_scale_quaternion_theta(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.scale_quaternion_rotation_angle(qtd.quaternion, qtd.scale_factor), qtd.scaled_quaternion,
                       atol=1e-6)
    assert np.allclose(ivy_mech.scale_quaternion_rotation_angle(qtd.batched_quaternion, qtd.batched_scale_factor)[0],
                       qtd.scaled_quaternion, atol=1e-6)
    ivy.unset_backend()


def test_hamilton_product(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.hamilton_product(qtd.quaternion, qtd.quaternion), qtd.hp_quaternion, atol=1e-6)
    assert np.allclose(ivy_mech.hamilton_product(qtd.batched_quaternion, qtd.batched_quaternion)[0],
                       qtd.hp_quaternion, atol=1e-6)
    ivy.unset_backend()
