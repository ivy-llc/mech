"""
Collection of tests for euler pose functions
"""

# global
import ivy
import ivy_mech
import ivy.functional.backends.numpy as ivy_np
import numpy as np

# local
from ivy_mech_tests.test_pose.pose_data import PoseTestData

ptd = PoseTestData()


def test_mat_pose_to_euler_pose(device, fw):
    for conv in ivy_mech.VALID_EULER_CONVENTIONS:
        with ivy_np.use:
            euler_pose = ivy_mech.mat_pose_to_euler_pose(ptd.matrix_pose, conv)
        ivy.set_backend(fw)
        assert np.allclose(
            ivy_mech.mat_pose_to_euler_pose(ivy.array(ptd.matrix_pose), conv),
            euler_pose,
            atol=1e-6,
        )
        assert np.allclose(
            ivy_mech.mat_pose_to_euler_pose(ivy.array(ptd.batched_matrix_pose), conv)[
                0
            ],
            euler_pose,
            atol=1e-6,
        )
        ivy.previous_backend()


def test_quaternion_pose_to_euler_pose(device, fw):
    for conv in ivy_mech.VALID_EULER_CONVENTIONS:
        with ivy_np.use:
            euler_pose = ivy_mech.mat_pose_to_euler_pose(ptd.matrix_pose, conv)
        ivy.set_backend(fw)
        assert np.allclose(
            ivy_mech.quaternion_pose_to_euler_pose(
                ivy.array(ptd.quaternion_pose), conv
            ),
            euler_pose,
            atol=1e-6,
        )
        assert np.allclose(
            ivy_mech.quaternion_pose_to_euler_pose(
                ivy.array(ptd.batched_quaternion_pose), conv
            )[0],
            euler_pose,
            atol=1e-6,
        )
        ivy.previous_backend()


def test_axis_angle_pose_to_euler_pose(device, fw):
    for conv in ivy_mech.VALID_EULER_CONVENTIONS:
        with ivy_np.use:
            euler_pose = ivy_mech.mat_pose_to_euler_pose(ptd.matrix_pose, conv)
        ivy.set_backend(fw)
        assert np.allclose(
            ivy_mech.axis_angle_pose_to_euler_pose(
                ivy.array(ptd.axis_angle_pose), conv
            ),
            euler_pose,
            atol=1e-6,
        )
        assert np.allclose(
            ivy_mech.axis_angle_pose_to_euler_pose(
                ivy.array(ptd.batched_axis_angle_pose), conv
            )[0],
            euler_pose,
            atol=1e-6,
        )
        ivy.previous_backend()
