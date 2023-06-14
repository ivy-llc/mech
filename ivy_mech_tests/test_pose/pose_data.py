"""test data for pose functions"""
# global
import ivy.functional.backends.numpy as ivy_np
import numpy as np

# local
import ivy_mech
from ivy_mech_tests.test_orientation.orientation_data import OrientationTestData


class PoseTestData(OrientationTestData):
    def __init__(self):
        super(PoseTestData, self).__init__()

        # cartesian co-ordinates
        self.cartesian_coords = np.array([[1.0, 2.0, 3.0]])
        self.batched_cartesian_coords = np.expand_dims(self.cartesian_coords, 0)

        # axis-angle pose
        self.axis_angle_pose = np.concatenate(
            (self.cartesian_coords, self.axis_angle), 1
        )
        self.batched_axis_angle_pose = np.expand_dims(self.axis_angle_pose, 0)

        # rotation vector pose
        self.rot_vec_pose = np.concatenate(
            (self.cartesian_coords, self.rotation_vector), 1
        )
        self.batched_rot_vec_pose = np.expand_dims(self.rot_vec_pose, 0)

        # euler pose
        self.euler_pose = np.concatenate((self.cartesian_coords, self.euler_angles), 1)
        self.batched_euler_pose = np.expand_dims(self.euler_pose, 0)

        # matrix pose
        self.matrix_pose = np.concatenate(
            (self.rotation_matrix, np.expand_dims(self.cartesian_coords, -1)), 2
        )
        self.batched_matrix_pose = np.expand_dims(self.matrix_pose, 0)

        # quaternion pose
        self.quaternion_pose = np.concatenate(
            (self.cartesian_coords, self.quaternion), 1
        )
        self.batched_quaternion_pose = np.expand_dims(self.quaternion_pose, 0)

        # incremented quaternion pose
        current_quaternion = self.quaternion_pose[:, 3:7]
        self.velocity = self.quaternion_pose
        self.batched_velocity = np.expand_dims(self.velocity, 0)
        self.control_dt = np.array([0.05])
        self.batched_control_dt = np.expand_dims(self.control_dt, 0)
        quaternion_vel = self.velocity[:, 3:7]
        with ivy_np.use:
            quaternion_transform = ivy_mech.scale_quaternion_rotation_angle(
                quaternion_vel, self.control_dt
            )
        new_quaternion = ivy_mech.hamilton_product(
            current_quaternion, quaternion_transform
        )
        self.incremented_quaternion = np.concatenate(
            (
                self.quaternion_pose[:, 0:3] + self.velocity[:, 0:3] * self.control_dt,
                new_quaternion,
            ),
            -1,
        )
        self.batched_incremented_quaternion = np.expand_dims(
            self.incremented_quaternion, 0
        )
