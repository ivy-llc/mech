"""Collection of Functions for Homogeneous Co-ordinates"""
# global
import ivy


def make_coordinates_homogeneous(coords, batch_shape=None):
    """Append ones to array of non-homogeneous co-ordinates to make them homogeneous.

    Parameters
    ----------
    coords
        Array of non-homogeneous co-ordinates *[batch_shape,d]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)

    Returns
    -------
    ret
        Array of Homogeneous co-ordinates *[batch_shape,d+1]*

    """
    if batch_shape is None:
        batch_shape = coords.shape[:-1]

    # shapes as list
    batch_shape = list(batch_shape)

    # BS x 1
    ones = ivy.ones_like(coords[..., 0:1])

    # BS x (D+1)
    return ivy.concat([coords, ones], axis=-1)


def make_transformation_homogeneous(matrices, batch_shape=None, device=None):
    """Append to set of 3x4 non-homogeneous matrices to make them homogeneous.

    Parameters
    ----------
    matrices
        set of 3x4 non-homogeneous matrices *[batch_shape,3,4]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        Same as x if None. (Default value = None)

    Returns
    -------
    ret
        4x4 Homogeneous matrices *[batch_shape,4,4]*

    """
    if batch_shape is None:
        batch_shape = matrices.shape[:-2]

    if device is None:
        device = ivy.dev(matrices)

    # shapes as list
    batch_shape = list(batch_shape)
    num_batch_dims = len(batch_shape)

    # BS x 1 x 4
    last_row = ivy.tile(
        ivy.reshape(
            ivy.array([0.0, 0.0, 0.0, 1.0], device=device),
            [1] * num_batch_dims + [1, 4],
        ),
        batch_shape + [1, 1],
    )

    # BS x 4 x 4
    return ivy.concat([matrices, last_row], axis=-2)
