# global
import os
import ivy
import math
import argparse
import ivy_mech
import ivy.functional.backends.numpy as ivy_np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ivy_demo_utils.open3d_utils import Visualizer
from ivy.framework_handler import set_framework, unset_framework
from ivy_demo_utils.ivy_scene.scene_utils import SimCam, BaseSimulator
from ivy_demo_utils.framework_utils import choose_random_framework, get_framework_from_str


class DummyOmCam:

    def __init__(self):
        self._pos = ivy.array([0., 0., 1.5])

    @staticmethod
    def cap():
        this_dir = os.path.dirname(os.path.realpath(__file__))
        return ivy.array(np.load(os.path.join(this_dir, 'ptcc_no_sim', 'omni_depth.npy'))),\
               ivy.array(np.load(os.path.join(this_dir, 'ptcc_no_sim', 'omni_rgb.npy')))

    @staticmethod
    def get_inv_ext_mat():
        return ivy.array([[0., -1., 0., 0.],
                          [1., 0., 0., 0.],
                          [0., 0., 1., 1.5]])

    def set_pos(self, pos):
        self._pos = pos

    def get_pos(self):
        return self._pos


class Simulator(BaseSimulator):

    def __init__(self, interactive, try_use_sim):
        super().__init__(interactive, try_use_sim)

        # initialize scene
        if self.with_pyrep:
            for i in range(6):
                self._vision_sensors[i].remove()
                self._vision_sensor_bodies[i].remove()
                [item.remove() for item in self._vision_sensor_rays[i]]
            self._default_camera.set_position(np.array([-2.3518, 4.3953, 2.8949]))
            self._default_camera.set_orientation(np.array([i*np.pi/180 for i in [112.90, 27.329, -10.978]]))
            inv_ext_mat = ivy.reshape(ivy.array(self._default_vision_sensor.get_matrix(), 'float32')[0:3, :], (3, 4))
            self.default_camera_ext_mat_homo = ivy.inv(ivy_mech.make_transformation_homogeneous(inv_ext_mat))

            # public objects
            self.omcam = SimCam(self._spherical_vision_sensor)

            # wait for user input
            self._user_prompt('\nInitialized scene with an omni-directional camera in the centre.\n\n'
                              'You can click on the omni directional camera, which appears as a small floating black sphere, '
                              'then select the box icon with four arrows in the top panel of the simulator, '
                              'and then drag the camera around dynamically.\n'
                              'Starting to drag and then holding ctrl allows you to also drag the camera up and down. \n\n'
                              'This demo enables you to capture 10 different omni-directional images from the camera, '
                              'and render the associated 10 point clouds in an open3D visualizer.\n\n'
                              'Both visualizers can be translated and rotated by clicking either the left mouse button or the wheel, '
                              'and then dragging the mouse.\n'
                              'Scrolling the mouse wheel zooms the view in and out.\n\n'
                              'Both visualizers can be rotated and zoomed by clicking either the left mouse button or the wheel, '
                              'and then dragging with the mouse.\n\n'
                              'Press enter in the terminal to use method ivy_mech.polar_coords_to_cartesian_coords and '
                              'show the first cartesian point cloud reconstruction of the scene, '
                              'converted from the polar co-ordinates captured from the omni-directional camera.\n\n')

        else:
            # public objects
            self.omcam = DummyOmCam()
            self.default_camera_ext_mat_homo = ivy.array(
                [[-0.872, -0.489,  0., 0.099],
                 [-0.169,  0.301, -0.938, 0.994],
                 [0.459, -0.818, -0.346, 5.677],
                 [0., 0., 0., 1.]])

            # message
            print('\nInitialized dummy scene with an omni-directional camera in the centre.'
                  '\nClose the visualization window to use method ivy_mech.polar_coords_to_cartesian_coords and show'
                  'a cartesian point cloud reconstruction of the scene, '
                  'converted from the omni-directional camera polar co-ordinates\n')

            # plot scene before rotation
            if interactive:
                plt.imshow(mpimg.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                     'ptcc_no_sim', 'before_capture.png')))
                plt.show()


def main(interactive=True, try_use_sim=True, f=None):
    f = choose_random_framework() if f is None else f
    set_framework(f)
    sim = Simulator(interactive, try_use_sim)
    vis = Visualizer(ivy.to_numpy(sim.default_camera_ext_mat_homo))
    pix_per_deg = 2
    om_pix = sim.get_pix_coords()
    plr_degs = om_pix / pix_per_deg
    plr_rads = plr_degs * math.pi / 180
    iterations = 10 if sim.with_pyrep else 1
    for _ in range(iterations):
        depth, rgb = sim.omcam.cap()
        plr = ivy.concat([plr_rads, depth], -1)
        xyz_wrt_cam = ivy_mech.polar_to_cartesian_coords(plr)
        xyz_wrt_cam = ivy.reshape(xyz_wrt_cam, (-1, 3))
        xyz_wrt_cam_homo = ivy_mech.make_coordinates_homogeneous(xyz_wrt_cam)
        inv_ext_mat_trans = ivy.matrix_transpose(sim.omcam.get_inv_ext_mat(), (1, 0))
        xyz_wrt_world = ivy.matmul(xyz_wrt_cam_homo, inv_ext_mat_trans)[..., 0:3]
        with ivy_np.use:
            omni_cam_inv_ext_mat = ivy_mech.make_transformation_homogeneous(
                ivy.to_numpy(sim.omcam.get_inv_ext_mat()))
        vis.show_point_cloud(xyz_wrt_world, rgb, interactive,
                             sphere_inv_ext_mats=[omni_cam_inv_ext_mat], sphere_radii=[0.025])
        if not interactive:
            sim.omcam.set_pos(sim.omcam.get_pos()
                               + ivy.array([-0.01, 0.01, 0.]))
    sim.close()
    unset_framework()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--non_interactive', action='store_true',
                        help='whether to run the demo in non-interactive mode.')
    parser.add_argument('--no_sim', action='store_true',
                        help='whether to run the demo without attempt to use the PyRep simulator.')
    parser.add_argument('--framework', type=str, default=None,
                        help='which framework to use. Chooses a random framework if unspecified.')
    parsed_args = parser.parse_args()
    framework = None if parsed_args.framework is None else get_framework_from_str(parsed_args.framework)
    main(not parsed_args.non_interactive, not parsed_args.no_sim, framework)
