"""test data for position functions"""
# global
import numpy as np


class PositionTestData:
    def __init__(self):
        # Co-ordinate Conversions #

        # cartesian co-ordinates
        self.cartesian_coords = np.array([[-1.0, 2.0, 3.0]])
        self.batched_cartesian_coords = np.expand_dims(self.cartesian_coords, 0)

        # polar co-ordinates
        x = self.cartesian_coords[:, 0:1]
        y = self.cartesian_coords[:, 1:2]
        z = self.cartesian_coords[:, 2:3]

        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)
        theta = np.arccos(z / r)

        # B x 3
        self.polar_coords = np.concatenate((phi, theta, r), 1)
        self.batched_polar_coords = np.expand_dims(self.polar_coords, 0)

        # Homogeneous Co-ordinates #

        # homogeneous coords
        self.cartesian_coords_homo = np.array([[-1.0, 2.0, 3.0, 1.0]])
        self.batched_cartesian_coords_homo = np.expand_dims(
            self.cartesian_coords_homo, 0
        )

        # transformation matrices
        self.ext_mat = np.eye(4)[0:3]
        self.batched_ext_mat = np.expand_dims(self.ext_mat, 0)

        # homogeneous transformation matrices
        self.ext_mat_homo = np.eye(4)
        self.batched_ext_mat_homo = np.expand_dims(self.ext_mat_homo, 0)
