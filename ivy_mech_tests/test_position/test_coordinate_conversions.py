"""
Collection of tests for co-ordinate conversion functions
"""

# global
import ivy_mech
import numpy as np
import ivy_mech_tests.helpers as helpers

# local
from ivy_mech_tests.test_position.position_data import PositionTestData

ptd = PositionTestData()


def test_polar_to_cartesian_coords():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        assert np.allclose(call(ivy_mech.polar_to_cartesian_coords, ptd.polar_coords), ptd.cartesian_coords, atol=1e-6)
        assert np.allclose(call(ivy_mech.polar_to_cartesian_coords, ptd.batched_polar_coords)[0],
                           ptd.cartesian_coords, atol=1e-6)


def test_cartesian_to_polar_coords():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        assert np.allclose(call(ivy_mech.cartesian_to_polar_coords, ptd.cartesian_coords), ptd.polar_coords, atol=1e-6)
        assert np.allclose(call(ivy_mech.cartesian_to_polar_coords, ptd.batched_cartesian_coords)[0],
                           ptd.polar_coords, atol=1e-6)


def test_cartesian_to_polar_coords_and_back():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        assert np.allclose(call(ivy_mech.polar_to_cartesian_coords,
                                call(ivy_mech.cartesian_to_polar_coords, ptd.cartesian_coords)),
                           ptd.cartesian_coords, atol=1e-6)
        assert np.allclose(call(ivy_mech.polar_to_cartesian_coords,
                                call(ivy_mech.cartesian_to_polar_coords, ptd.batched_cartesian_coords))[0],
                           ptd.cartesian_coords, atol=1e-6)


def test_polar_to_cartesian_coords_and_back():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        assert np.allclose(call(ivy_mech.cartesian_to_polar_coords,
                                call(ivy_mech.polar_to_cartesian_coords, ptd.polar_coords)),
                           ptd.polar_coords, atol=1e-6)
        assert np.allclose(call(ivy_mech.cartesian_to_polar_coords,
                                call(ivy_mech.polar_to_cartesian_coords, ptd.batched_polar_coords))[0],
                           ptd.polar_coords, atol=1e-6)
