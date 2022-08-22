"""
Collection of tests for ivy mechanics demos
"""

# global
import pytest
import ivy_tests.test_ivy.helpers as helpers



def test_demo_run_through(device, f, fw):
    from ivy_mech_demos.run_through import main
    if fw == 'tensorflow_graph':
        # these particular demos are only implemented in eager mode, without compilation
        pytest.skip()
    main(f=f,fw=fw)


@pytest.mark.parametrize(
    "with_sim", [False])
def test_demo_target_facing_rotation_vector(with_sim, device, f, fw):
    from ivy_mech_demos.interactive.target_facing_rotation_matrix import main
    if fw == 'tensorflow_graph':
        # these particular demos are only implemented in eager mode, without compilation
        pytest.skip()
    main(False, with_sim, f=f, fw=fw)


@pytest.mark.parametrize(
    "with_sim", [False])
def test_demo_polar_to_cartesian_coords(with_sim, device, f, fw):
    from ivy_mech_demos.interactive.polar_to_cartesian_coords import main
    if fw == 'tensorflow_graph':
        # these particular demos are only implemented in eager mode, without compilation
        pytest.skip()
    main(False, with_sim, f=f, fw=fw)
