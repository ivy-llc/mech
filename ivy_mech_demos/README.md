# Ivy Mechanics Demos

We provide a simple set of interactive demos for the Ivy Mechanics library.
Running these demos is quick and simple.

## Install

First, clone this repo:

```bash
git clone https://github.com/unifyai/mech.git ~/ivy_mech
```

The interactive demos optionally make use of the simulator
[CoppeliaSim](https://www.coppeliarobotics.com/),
and the python wrapper [PyRep](https://github.com/stepjam/PyRep).

If these are not installed, the demos will all still run, but will display pre-rendered images from the simultator.

### Local

For a local installation, first install the dependencies:

```bash
cd ~/ivy_mech
python3 -m pip install -r requirements.txt
cd ~/ivy_mech/ivy_mech_demos
python3 -m pip install -r requirements.txt
```

To run interactive demos inside a simulator, CoppeliaSim and PyRep should then be installed following the installation [intructions](https://github.com/stepjam/PyRep#install).

### Docker

For a docker installation, first ensure [docker](https://docs.docker.com/get-docker/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) are installed.

Then simply pull the ivy mechanics image:

```bash
docker pull ivydl/ivy-mech:latest
```

## Demos

All demos can be run by executing the python scripts directly.
If a demo script is run without command line arguments, then a random backend framework will be selected from those installed.
Alternatively, the `--framework` argument can be used to manually specify a framework from the options
`jax`, `tensorflow`, `torch`, `mxnd` or `numpy`.

The examples below assume a docker installation, but the demo scripts can also
be run with python directly for local installations.

### Run Through

For a basic run through the library:

```bash
cd ~/ivy_mech/ivy_mech_demos
./run_demo.sh run_through
```

This script, and the various parts of the library, are further discussed in the [Run Through](https://github.com/unifyai/mech#run-through) section of the main README.
We advise following along with this section for maximum effect. The demo script should also be opened locally,
and breakpoints added to step in at intermediate points to further explore.

To run the script using a specific backend, tensorflow for example, then run like so:

```bash
./run_demo.sh run_through --framework tensorflow
```

### Target Facing Rotation Matrix

In this demo, a plant pot is dragged around the scene, and a camera is set to
dynamically track the plant pot using the function ivy_mech.target_facing_rotation_matrix.

```bash
cd ~/ivy_mech/ivy_mech_demos
./run_demo.sh interactive.target_facing_rotation_matrix
```

Example output from the simulator is given below:

<p align="center">
    <img width="75%" style="display: block;" src='https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_mech/demo_a.gif?raw=true'>
</p>

### Polar to Cartesian Co-ordinates

In this demo, an omni-directional camera is dragged around the scene,
and a cartesian point cloud reconstruction is dynamically generated from the omni camera polar depth maps,
using the method ivy_mech.polar_to_cartesian_coords.

```bash
cd ~/ivy_mech/ivy_mech_demos
./run_demo.sh interactive.polar_to_cartesian_coords
```
Example output from the simulator, and Open3D renderings, are given below:

<p align="center">
    <img width="75%" style="display: block;" src='https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_mech/demo_b.gif?raw=true'>
</p>

## Get Involved

If you have any issues running any of the demos, would like to request further demos, or would like to implement your own, then get it touch.
Feature requests, pull requests, and [tweets](https://twitter.com/unify_ai) all welcome!