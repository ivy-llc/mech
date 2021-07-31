"""
Collection of tests for ivy mechanics demos
"""

# global
import pytest
import ivy_tests.helpers as helpers


def test_demo_run_through(dev_str, f, call):
    from demos.run_through import main
    if call in [helpers.tf_graph_call]:
        # these particular demos are only implemented in eager mode, without compilation
        pytest.skip()
    main(f=f)


@pytest.mark.parametrize(
    "with_sim", [False])
def test_demo_target_facing_rotation_vector(with_sim, dev_str, f, call):
    from demos.interactive.target_facing_rotation_matrix import main
    if call in [helpers.tf_graph_call]:
        # these particular demos are only implemented in eager mode, without compilation
        pytest.skip()
    main(False, with_sim, f=f)


@pytest.mark.parametrize(
    "with_sim", [False])
def test_demo_polar_to_cartesian_coords(with_sim, dev_str, f, call):
    from demos.interactive.polar_to_cartesian_coords import main
    if call in [helpers.tf_graph_call]:
        # these particular demos are only implemented in eager mode, without compilation
        pytest.skip()
    main(False, with_sim, f=f)
