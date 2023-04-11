# global
import os
import ivy
import time
import math
import ivy_mech
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ivy_demo_utils.ivy_scene.scene_utils import SimObj, BaseSimulator


class DummyCam:

    def __init__(self, interactive):
        self._pos = ivy.array([-1, 0.6, 1.45])
        self._interactive = interactive

    def get_pos(self):
        return self._pos

    def set_rot_mat(self, _):
        print('\nCamera now facing towards plant plot...'
              '\nClose the visualization window to end the demo.\n')
        if self._interactive:
            plt.imshow(mpimg.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 'tfrm_no_sim', 'after_rotation.png')))
            plt.show()


class DummyTarget:

    def __init__(self):
        self._pos = ivy.array([0.17179595, -0.01713575,  1.02739596])

    def get_pos(self):
        return self._pos

    def set_pos(self, _):
        return


class Simulator(BaseSimulator):

    def __init__(self, interactive, try_use_sim):
        super().__init__(interactive, try_use_sim)

        # initialize scene
        if self.with_pyrep:
            self._spherical_vision_sensor.remove()
            for i in range(1, 6):
                self._vision_sensors[i].remove()
                self._vision_sensor_bodies[i].remove()
                [item.remove() for item in self._vision_sensor_rays[i]]
            self._vision_sensor_body_0.set_position([-1, 0.6, 1.45])
            self._vision_sensor_body_0.set_orientation([i*math.pi/180 for i in [-145, -25, -180]])
            self._default_camera.set_position(np.array([-1.1013, -2.1, 1.9484]))
            self._default_camera.set_orientation(np.array([i*np.pi/180 for i in [-114.69, 13.702, -173.78]]))

            # public objects
            self.cam = SimObj(self._vision_sensor_body_0)
            self.target = SimObj(self._plant)

            # prompt input
            self._user_prompt('\nInitialized scene with a camera facing away from the plant plot.\n\n'
                              'The visualizer can be translated and rotated by clicking either the left mouse button or the wheel, '
                              'and then dragging the mouse.\n'
                              'Scrolling the mouse wheel zooms the view in and out.\n\n'
                              'You can click on the plant pot, '
                              'then select the box icon with four arrows in the top panel of the simulator, '
                              'and then drag the plant pot around dynamically.\n'
                              'Starting to drag and then holding ctrl allows you to also drag the pot up and down. \n\n'
                              'Press enter in the terminal to use method ivy_mech.target_facing_rotation_vector '
                              'to rotate the camera to track the plant pot as you move it...\n\n')

        else:
            # public objects
            self.cam = DummyCam(interactive)
            self.target = DummyTarget()

            # message
            print('\nInitialized dummy scene with a camera facing away from the plant plot.'
                  '\nClose the visualization window to use method ivy_mech.target_facing_rotation_vector '
                  'to rotate the camera to face towards the plant pot...\n')

            # plot scene before rotation
            if interactive:
                plt.imshow(mpimg.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                     'tfrm_no_sim', 'before_rotation.png')))
                plt.show()


def main(interactive=True, try_use_sim=True, f=None, fw=None):
    fw = ivy.choose_random_backend() if fw is None else fw
    ivy.set_backend(fw)
    f = ivy.with_backend(backend=fw)
    sim = Simulator(interactive, try_use_sim)
    cam_pos = sim.cam.get_pos()
    iterations = 250 if sim.with_pyrep else 1
    msg = 'tracking plant pot for 250 simulator steps...' if sim.with_pyrep else ''
    print(msg)
    for i in range(iterations):
        target_pos = sim.target.get_pos()
        tfrm = ivy_mech.target_facing_rotation_matrix(cam_pos, target_pos)
        sim.cam.set_rot_mat(tfrm)
        if not interactive:
            sim.target.set_pos(sim.target.get_pos()
                               + ivy.array([-0.01, 0.01, 0.]))
        time.sleep(0.05)
    sim.close()
    ivy.previous_backend()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--non_interactive', action='store_true',
                        help='whether to run the demo in non-interactive mode.')
    parser.add_argument('--no_sim', action='store_true',
                        help='whether to run the demo without attempt to use the PyRep simulator.')
    parser.add_argument('--backend', type=str, default=None,
                        help='which backend to use. Chooses a random backend if unspecified.')
    parsed_args = parser.parse_args()
    fw = parsed_args.backend()
    f = None if fw is None else ivy.with_backend(backend=fw)
    main(not parsed_args.non_interactive, not parsed_args.no_sim, f, fw)
