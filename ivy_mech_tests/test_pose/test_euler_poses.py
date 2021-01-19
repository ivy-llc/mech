"""
Collection of tests for euler pose functions
"""

# global
import ivy_mech
import numpy as np
import ivy_mech_tests.helpers as helpers

# local
from ivy_mech_tests.test_pose.pose_data import PoseTestData

ptd = PoseTestData()


def test_mat_pose_to_euler_pose():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not fully support array slicing
            continue
        for conv in ivy_mech.VALID_EULER_CONVENTIONS:
            euler_pose = ivy_mech.mat_pose_to_euler_pose(ptd.matrix_pose, conv, f=helpers._ivy_np)
            assert np.allclose(call(ivy_mech.mat_pose_to_euler_pose, ptd.matrix_pose, conv), euler_pose, atol=1e-6)
            assert np.allclose(call(ivy_mech.mat_pose_to_euler_pose, ptd.batched_matrix_pose, conv)[0], euler_pose,
                               atol=1e-6)
