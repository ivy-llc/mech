"""
Collection of tests for co-ordinate conversion functions
"""

# global
import ivy
import ivy_mech
import numpy as np

# local
from ivy_mech_tests.test_position.position_data import PositionTestData

ptd = PositionTestData()


def test_polar_to_cartesian_coords(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.polar_to_cartesian_coords(ivy.array(ptd.polar_coords)), ptd.cartesian_coords, atol=1e-6)
    assert np.allclose(ivy_mech.polar_to_cartesian_coords(ivy.array(ptd.batched_polar_coords))[0],
                       ptd.cartesian_coords, atol=1e-6)
    ivy.previous_backend()


def test_cartesian_to_polar_coords(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.cartesian_to_polar_coords(ivy.array(ptd.cartesian_coords)), ptd.polar_coords, atol=1e-6)
    assert np.allclose(ivy_mech.cartesian_to_polar_coords(ivy.array(ptd.batched_cartesian_coords))[0],
                       ptd.polar_coords, atol=1e-6)
    ivy.previous_backend()


def test_cartesian_to_polar_coords_and_back(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.polar_to_cartesian_coords(
                            ivy_mech.cartesian_to_polar_coords(ivy.array(ptd.cartesian_coords))),
                       ptd.cartesian_coords, atol=1e-6)
    assert np.allclose(ivy_mech.polar_to_cartesian_coords(
                            ivy_mech.cartesian_to_polar_coords(ivy.array(ptd.batched_cartesian_coords)))[0],
                       ptd.cartesian_coords, atol=1e-6)
    ivy.previous_backend()


def test_polar_to_cartesian_coords_and_back(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_mech.cartesian_to_polar_coords(
                            ivy_mech.polar_to_cartesian_coords(ivy.array(ptd.polar_coords))),
                       ptd.polar_coords, atol=1e-6)
    assert np.allclose(ivy_mech.cartesian_to_polar_coords(
                            ivy_mech.polar_to_cartesian_coords(ivy.array(ptd.batched_polar_coords)))[0],
                       ptd.polar_coords, atol=1e-6)
    ivy.previous_backend()
