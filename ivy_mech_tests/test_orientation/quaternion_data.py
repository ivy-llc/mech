"""test data for quaternion functions"""
# global
import numpy as np

# local
import ivy_mech
from ivy_mech_tests.test_orientation.orientation_data import OrientationTestData


class QuaternionTestData(OrientationTestData):
    def __init__(self):
        super(QuaternionTestData, self).__init__()

        # scaled angle
        self.scale_factor = np.array([2.0])
        self.batched_scale_factor = np.expand_dims(self.scale_factor, 0)
        self.scaled_angle = self.angle * self.scale_factor
        self.batched_scaled_angle = np.expand_dims(self.scaled_angle, 0)

        # scaled quaternion
        n = np.cos(self.scaled_angle / 2)
        e1 = np.sin(self.scaled_angle / 2) * self.axis[:, 0:1]
        e2 = np.sin(self.scaled_angle / 2) * self.axis[:, 1:2]
        e3 = np.sin(self.scaled_angle / 2) * self.axis[:, 2:3]
        self.scaled_quaternion = np.concatenate((e1, e2, e3, n), -1)
        self.batched_scaled_quaternion = np.expand_dims(self.scaled_quaternion, 0)

        # hamilton product quaternion
        q1 = self.quaternion
        q2 = self.quaternion

        a1 = q1[..., 3:4]
        a2 = q2[..., 3:4]
        b1 = q1[..., 0:1]
        b2 = q2[..., 0:1]
        c1 = q1[..., 1:2]
        c2 = q2[..., 1:2]
        d1 = q1[..., 2:3]
        d2 = q2[..., 2:3]

        term_r = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        term_i = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
        term_j = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
        term_k = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

        self.hp_quaternion = np.concatenate((term_i, term_j, term_k, term_r), -1)
        self.batched_hp_quaternion = np.expand_dims(self.hp_quaternion, 0)

        # target facing quaternion
        self.body_pos = np.array([0.0, 1.0, 2.0])
        self.batched_body_pos = np.expand_dims(self.body_pos, 0)
        self.target_pos = np.array([-1.0, -2.0, -3.0])
        self.batched_target_pos = np.expand_dims(self.target_pos, 0)

        rel_body_pos = self.body_pos - self.target_pos

        zeros = np.zeros([1])
        rel_body_pos_x = rel_body_pos[0:1]
        rel_body_pos_y = rel_body_pos[1:2]
        rel_body_pos_z = rel_body_pos[2:3]

        axis_vector = np.concatenate((rel_body_pos_y, -rel_body_pos_x, zeros), -1)
        axis_vector /= np.linalg.norm(axis_vector)

        rel_body_pos_xy_dist = np.linalg.norm(rel_body_pos[..., 0:2])
        sign = rel_body_pos[..., 1:2] >= 0
        theta = np.arctan(sign * rel_body_pos_xy_dist / rel_body_pos_z)

        downward_facing_quaternion = np.array([0, 1, 0, 0])
        quaternion = ivy_mech.axis_angle_to_quaternion(
            np.concatenate([axis_vector, theta], -1)
        )
        self.target_facing_quaternion = ivy_mech.hamilton_product(
            downward_facing_quaternion, quaternion
        )
