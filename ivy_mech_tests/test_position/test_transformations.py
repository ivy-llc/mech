"""
Collection of tests for homogeneous co-ordinate functions
"""

# global
import ivy
import ivy_mech
import numpy as np

# local
from ivy_mech_tests.test_position.position_data import PositionTestData

ptd = PositionTestData()


def test_make_coordinates_homogeneous(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_mech.make_coordinates_homogeneous(ivy.array(ptd.cartesian_coords)),
        ptd.cartesian_coords_homo,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_mech.make_coordinates_homogeneous(ivy.array(ptd.batched_cartesian_coords)),
        ptd.batched_cartesian_coords_homo,
        atol=1e-6,
    )
    ivy.previous_backend()


def test_make_transformation_homogeneous(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_mech.make_transformation_homogeneous(ivy.array(ptd.ext_mat)),
        ptd.ext_mat_homo,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_mech.make_transformation_homogeneous(ivy.array(ptd.batched_ext_mat)),
        ptd.batched_ext_mat_homo,
        atol=1e-6,
    )
    ivy.previous_backend()
