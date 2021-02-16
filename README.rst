.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='docs/partial_source/logos/logo.png'>
    </p>

.. raw:: html

    <br/>
    <a href="https://pypi.org/project/ivy-mech">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://badge.fury.io/py/ivy-mech.svg">
    </a>
    <a href="https://www.apache.org/licenses/LICENSE-2.0">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/pypi/l/ivy-mech">
    </a>
    <a href="https://github.com/ivy-dl/mech/actions?query=workflow%3Adocs">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/workflow/status/ivy-dl/mech/docs?label=docs">
    </a>
    <a href="https://github.com/ivy-dl/mech/actions?query=workflow%3Anightly-tests">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/workflow/status/ivy-dl/mech/nightly-tests?label=nightly">
    </a>
    <a href="https://github.com/ivy-dl/mech/actions?query=workflow%3Apypi-tests">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/workflow/status/ivy-dl/mech/pypi-tests?label=pypi">
    </a>
    <a href="https://discord.gg/EN9YS3QW8w">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/799879767196958751?color=blue&label=%20&logo=discord&logoColor=white">
    </a>
    <br clear="all" />

**Mechanics functions with end-to-end support for deep learning developers, written in Ivy.**

.. raw:: html

    <div style="display: block;">
        <img width="4%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://jax.readthedocs.io">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/jax_logo.png">
        </a>
        <img width="6.66%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://www.tensorflow.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/tensorflow_logo.png">
        </a>
        <img width="6.66%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://pytorch.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/pytorch_logo.png">
        </a>
        <img width="6.66%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://mxnet.apache.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/mxnet_logo.png">
        </a>
        <img width="6.66%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://numpy.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/numpy_logo.png">
        </a>
    </div>

Contents
--------

* `Overview`_
* `Run Through`_
* `Interactive Demos`_
* `Get Involed`_

Overview
--------

.. _docs: https://ivy-dl.org/mech

**What is Ivy Mechanics?**

Ivy mechanics provides functions for conversions of orientation, pose, and positional representations,
as well as frame-of-reference transformations, and other more applied functions. Check out the docs_ for more info!

The library is built on top of the Ivy deep learning framework.
This means all functions simultaneously support:
Jax, Tensorflow, PyTorch, MXNet, and Numpy.

**A Family of Libraries**

Ivy mechanis is one library in a family of Ivy libraries.
There are also Ivy libraries for 3D vision, robotics, differentiable memory, and differentiable gym environments.
Click on the icons below for their respective github pages.

.. raw:: html

    <div style="display: block;">
        <img width="8%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/mech">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_mech.png">
        </a>
        <img width="2%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/vision">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_vision.png">
        </a>
        <img width="2%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/robot">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_robot.png">
        </a>
        <img width="2%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/memory">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_memory.png">
        </a>
        <img width="2%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/gym">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_gym.png">
        </a>
    </div>
    <br clear="all" />

**Quick Start**

Ivy mechanics can be installed like so: ``pip install ivy-mech``

.. _demos: https://github.com/ivy-dl/mech/tree/master/demos
.. _interactive: https://github.com/ivy-dl/mech/tree/master/demos/interactive

To quickly see the different aspects of the library, we suggest you check out the demos_!
We suggest you start by running the script ``run_through.py``,
and read the "Run Through" section below which explains this script.

For more interactive demos, we suggest you run either
``target_facing_rotation_matrix.py`` or ``polar_to_cartesian_coords.py`` in the interactive_ demos folder.

Run Through
-----------

We run through some of the different parts of the library via a simple ongoing example script.
The full script is available in the demos_ folder, as file ``run_through.py``.
First, we select a random backend framework ``f`` to use for the examples, from the options
``ivy.jax``, ``ivy.tensorflow``, ``ivy.torch``, ``ivy.mxnd`` or ``ivy.numpy``.

.. code-block:: python

    from ivy_demo_utils.framework_utils import choose_random_framework
    f = choose_random_framework()

**Orientation Module**

The orientation module is the most comprehensive, with conversions to and from all euler conventions, quaternions,
rotation matrices, rotation vectors, and axis-angle representations.

A few of the orientation functions are outlined below.

.. code-block:: python

    # rotation representations

    # 3
    rot_vec = f.array([0., 1., 0.])

    # 3 x 3
    rot_mat = ivy_mech.rot_vec_to_rot_mat(rot_vec)

    # 3
    euler_angles = ivy_mech.rot_mat_to_euler(rot_mat, 'zyx')

    # 4
    quat = ivy_mech.euler_to_quaternion(euler_angles)

    # 3, 1
    axis, angle = ivy_mech.quaternion_to_axis_and_angle(quat)

    # 3
    rot_vec_again = axis * angle

**Pose Module**

The pose representations mirror the orientation representations, with the addition of 3 values to
represent the cartesian position. Again, we give some examples below.

.. code-block:: python

    # pose representations

    # 3
    position = f.ones_like(rot_vec)

    # 6
    rot_vec_pose = f.concatenate((position, rot_vec), 0)

    # 3 x 4
    mat_pose = ivy_mech.rot_vec_pose_to_mat_pose(rot_vec_pose)

    # 6
    euler_pose = ivy_mech.mat_pose_to_euler_pose(mat_pose)

    # 7
    quat_pose = ivy_mech.euler_pose_to_quaternion_pose(euler_pose)

    # 6
    rot_vec_pose_again = ivy_mech.quaternion_pose_to_rot_vec_pose(quat_pose)

**Position Module**

The position module includes functions for converting between positional representations,
such as cartesian and polar conventions,
and for applying frame-of-reference transformations to cartesian co-ordinates.

We give some examples for conversion of positional representation below.

.. code-block:: python

    # conversions of positional representation

    # 3
    cartesian_coord = f.random_uniform(0., 1., (3,))

    # 3
    polar_coord = ivy_mech.cartesian_to_polar_coords(
        cartesian_coord)

    # 3
    cartesian_coord_again = ivy_mech.polar_to_cartesian_coords(
        polar_coord)

Assuming cartesian form, we give an example of a frame-of-reference transformations below.

.. code-block:: python

    # cartesian co-ordinate frame-of-reference transformations

    # 3 x 4
    trans_mat = f.random_uniform(0., 1., (3, 4))

    # 4
    cartesian_coord_homo = ivy_mech.make_coordinates_homogeneous(
        cartesian_coord)

    # 3
    trans_cartesian_coord = f.matmul(
        trans_mat, f.expand_dims(cartesian_coord_homo, -1))[:, 0]

    # 4
    trans_cartesian_coord_homo = ivy_mech.make_coordinates_homogeneous(
        trans_cartesian_coord)

    # 4 x 4
    trans_mat_homo = ivy_mech.make_transformation_homogeneous(
        trans_mat)

    # 3 x 4
    inv_trans_mat = f.inv(trans_mat_homo)[0:3]

    # 3
    cartesian_coord_again = f.matmul(
        inv_trans_mat, f.expand_dims(trans_cartesian_coord_homo, -1))[:, 0]

Interactive Demos
-----------------

In addition to the run through above, we provide two further demo scripts,
which are more visual and interactive, and are each built around a particular function.

Rather than presenting the code here, we show visualizations of the demos.
The scripts for these demos can be found in the interactive_ demos folder.

**Target Facing Rotation Matrix**

The first demo uses the method ``ivy_mech.target_facing_rotation_matrix`` to dynamically
track a moving target plant pot with a camera, as shown below:

.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_mech/demo_a.gif?raw=true'>
    </p>

**Polar to Cartesian Co-ordinates**

The second demo uses the method ``ivy_mech.polar_to_cartesian_coords`` to convert a polar depth image
acquired from an omni-directional camera into cartesian co-ordinates,
enabling the surrounding geometry to be represented as a point cloud, for interactive visualization.

.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_mech/demo_b.gif?raw=true'>
    </p>

Get Involed
-----------

We hope the functions in this library are useful to a wide range of deep learning developers.
However, there are many more areas of mechanics which could be covered by this library.

If there are any particular mechanics functions you feel are missing,
and your needs are not met by the functions currently on offer,
then we are very happy to accept pull requests!

We look forward to working with the community on expanding and improving the Ivy mechanics library.

Citation
--------

::

    @article{lenton2021ivy,
      title={Ivy: Templated Deep Learning for Inter-Framework Portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }