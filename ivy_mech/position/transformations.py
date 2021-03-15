"""
Collection of Functions for Homogeneous Co-ordinates
"""

# global
import ivy as _ivy


def make_coordinates_homogeneous(coords, batch_shape=None):
    """
    Append ones to array of non-homogeneous co-ordinates to make them homogeneous.

    :param coords: Array of non-homogeneous co-ordinates *[batch_shape,d]*
    :type coords: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :return: Array of Homogeneous co-ordinates *[batch_shape,d+1]*
    """

    if batch_shape is None:
        batch_shape = coords.shape[:-1]

    # shapes as list
    batch_shape = list(batch_shape)

    # BS x 1
    ones = _ivy.ones_like(coords[..., 0:1])

    # BS x (D+1)
    return _ivy.concatenate((coords, ones), -1)


def make_transformation_homogeneous(matrices, batch_shape=None, dev_str=None):
    """
    Append to set of 3x4 non-homogeneous matrices to make them homogeneous.

    :param matrices: set of 3x4 non-homogeneous matrices *[batch_shape,3,4]*
    :type matrices: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :return: 4x4 Homogeneous matrices *[batch_shape,4,4]*
    """

    if batch_shape is None:
        batch_shape = matrices.shape[:-2]

    if dev_str is None:
        dev_str = _ivy.dev_str(matrices)

    # shapes as list
    batch_shape = list(batch_shape)
    num_batch_dims = len(batch_shape)

    # BS x 1 x 4
    last_row = _ivy.tile(_ivy.reshape(_ivy.array([0., 0., 0., 1.], dev_str=dev_str), [1] * num_batch_dims + [1, 4]),
                         batch_shape + [1, 1])

    # BS x 4 x 4
    return _ivy.concatenate((matrices, last_row), -2)
