"""test data for orientation functions"""
# global
import numpy as np


class OrientationTestData:
    def __init__(self):
        # axis
        axis = np.array([[1.0, 2.0, 3.0]])
        self.axis = axis / np.linalg.norm(axis)
        self.batched_axis = np.expand_dims(self.axis, 0)

        # angle
        self.angle = np.array([[np.pi / 3]])
        self.batched_angle = np.expand_dims(self.angle, 0)

        # rotation vector
        self.rotation_vector = self.axis * self.angle
        self.batched_rotation_vector = np.expand_dims(self.rotation_vector, 0)

        # axis angle
        self.axis_angle = np.concatenate((self.axis, self.angle), -1)
        self.batched_axis_angle = np.expand_dims(self.axis_angle, 0)

        # polar axis angle
        theta = np.arccos(self.axis[:, 2:3])
        phi = np.arctan2(self.axis[:, 1:2], self.axis[:, 0:1])
        self.polar_axis_angle = np.concatenate((theta, phi, self.angle), -1)
        self.batched_polar_axis_angle = np.expand_dims(self.polar_axis_angle, 0)

        # quaternion
        n = np.cos(self.angle / 2)
        e1 = np.sin(self.angle / 2) * self.axis[:, 0:1]
        e2 = np.sin(self.angle / 2) * self.axis[:, 1:2]
        e3 = np.sin(self.angle / 2) * self.axis[:, 2:3]
        self.quaternion = np.concatenate((e1, e2, e3, n), -1)
        self.batched_quaternion = np.expand_dims(self.quaternion, 0)

        # rotation matrix
        a = np.expand_dims(self.quaternion[:, 3:4], -1)
        b = np.expand_dims(self.quaternion[:, 0:1], -1)
        c = np.expand_dims(self.quaternion[:, 1:2], -1)
        d = np.expand_dims(self.quaternion[:, 2:3], -1)

        top_left = a**2 + b**2 - c**2 - d**2
        top_middle = 2 * b * c - 2 * a * d
        top_right = 2 * b * d + 2 * a * c
        middle_left = 2 * b * c + 2 * a * d
        middle_middle = a**2 - b**2 + c**2 - d**2
        middle_right = 2 * c * d - 2 * a * b
        bottom_left = 2 * b * d - 2 * a * c
        bottom_middle = 2 * c * d + 2 * a * b
        bottom_right = a**2 - b**2 - c**2 + d**2

        top_row = np.concatenate((top_left, top_middle, top_right), -1)
        middle_row = np.concatenate((middle_left, middle_middle, middle_right), -1)
        bottom_row = np.concatenate((bottom_left, bottom_middle, bottom_right), -1)

        self.rotation_matrix = np.concatenate((top_row, middle_row, bottom_row), -2)
        self.batched_rotation_matrix = np.expand_dims(self.rotation_matrix, 0)

        # euler
        x_angle = np.arctan2(
            self.rotation_matrix[:, 1:2, 2:3], self.rotation_matrix[:, 2:3, 2:3]
        )
        c2 = np.sqrt(
            self.rotation_matrix[:, 0:1, 0:1] ** 2
            + self.rotation_matrix[:, 0:1, 1:2] ** 2
        )
        y_angle = np.arctan2(-self.rotation_matrix[:, 0:1, 2:3], c2)
        s1 = np.sin(x_angle)
        c1 = np.cos(x_angle)
        z_angle = np.arctan2(
            s1 * self.rotation_matrix[:, 2:3, 0:1]
            - c1 * self.rotation_matrix[:, 1:2, 0:1],
            c1 * self.rotation_matrix[:, 1:2, 1:2]
            - s1 * self.rotation_matrix[:, 2:3, 1:2],
        )

        x_angle = -x_angle[:, :, 0]
        y_angle = -y_angle[:, :, 0]
        z_angle = -z_angle[:, :, 0]

        self.euler_angles = np.concatenate((z_angle, y_angle, x_angle), 1)
        self.batched_euler_angles = np.expand_dims(self.euler_angles, 0)
