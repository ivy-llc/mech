"""
Collection of tests for ivy mechanics demos
"""

# global
import ivy_mech_tests.helpers as helpers


def test_demo_run_through():
    from demos.run_through import main
    for lib, call in helpers.calls:
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # these particular demos are only implemented in eager mode, without compilation
            continue
        main(f=lib)


def test_demo_target_facing_rotation_vector():
    from demos.interactive.target_facing_rotation_matrix import main
    for lib, call in helpers.calls:
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # these particular demos are only implemented in eager mode, without compilation
            continue
        main(False, False, f=lib)


def test_demo_polar_to_cartesian_coords():
    from demos.interactive.polar_to_cartesian_coords import main
    for lib, call in helpers.calls:
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # these particular demos are only implemented in eager mode, without compilation
            continue
        main(False, False, f=lib)
