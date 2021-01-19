"""
Collection of Functions for Homogeneous Co-ordinates
"""

# global
from ivy.framework_handler import get_framework as _get_framework


def make_coordinates_homogeneous(coords, batch_shape=None, f=None):
    """
    Append ones to array of non-homogeneous co-ordinates to make them homogeneous.

    :param coords: Array of non-homogeneous co-ordinates *[batch_shape,d]*
    :type coords: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Array of Homogeneous co-ordinates *[batch_shape,d+1]*
    """

    f = _get_framework(coords, f=f)

    if batch_shape is None:
        batch_shape = coords.shape[:-1]

    # shapes as list
    batch_shape = list(batch_shape)

    # BS x 1
    ones = f.ones_like(coords[..., 0:1])

    # BS x (D+1)
    return f.concatenate((coords, ones), -1)


def make_transformation_homogeneous(matrices, batch_shape=None, dev=None, f=None):
    """
    Append to set of 3x4 non-homogeneous matrices to make them homogeneous.

    :param matrices: set of 3x4 non-homogeneous matrices *[batch_shape,3,4]*
    :type matrices: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: 4x4 Homogeneous matrices *[batch_shape,4,4]*
    """

    f = _get_framework(matrices, f=f)

    if batch_shape is None:
        batch_shape = matrices.shape[:-2]

    if dev is None:
        dev = f.get_device(matrices)

    # shapes as list
    batch_shape = list(batch_shape)
    num_batch_dims = len(batch_shape)

    # BS x 1 x 4
    last_row = f.tile(f.reshape(f.array([0., 0., 0., 1.], dev=dev), [1] * num_batch_dims + [1, 4]),
                         batch_shape + [1, 1])

    # BS x 4 x 4
    return f.concatenate((matrices, last_row), -2)
