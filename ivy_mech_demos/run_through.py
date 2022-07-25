# global
import ivy
import argparse
import ivy_mech


def main(f=None, fw=None):

    # Framework Setup #
    # ----------------#

    # choose random framework

    fw = ivy.choose_random_backend(excluded='mxnet') if fw is None else fw
    ivy.set_backend(fw)
    f=ivy.get_backend(fw) if f is None else f

    # Orientation #
    # ------------#

    # rotation representations

    # 3
    rot_vec = ivy.array([0., 1., 0.])

    # 3 x 3
    rot_mat = ivy_mech.rot_vec_to_rot_mat(rot_vec)

    # 3
    euler_angles = ivy_mech.rot_mat_to_euler(rot_mat, 'zyx')

    # 4
    quat = ivy_mech.euler_to_quaternion(euler_angles)

    # 4
    axis_and_angle = ivy_mech.quaternion_to_axis_angle(quat)

    # 3
    rot_vec_again = axis_and_angle[..., :-1] * axis_and_angle[..., -1:]

    # Pose #
    # -----#

    # pose representations

    # 3
    position = ivy.ones_like(rot_vec)

    # 6
    rot_vec_pose = ivy.concat([position, rot_vec], 0)

    # 3 x 4
    mat_pose = ivy_mech.rot_vec_pose_to_mat_pose(rot_vec_pose)

    # 6
    euler_pose = ivy_mech.mat_pose_to_euler_pose(mat_pose)

    # 7
    quat_pose = ivy_mech.euler_pose_to_quaternion_pose(euler_pose)

    # 6
    rot_vec_pose_again = ivy_mech.quaternion_pose_to_rot_vec_pose(quat_pose)

    # Position #
    # ---------#

    # conversions of positional representation

    # 3
    cartesian_coord = ivy.random_uniform(0., 1., (3,))

    # 3
    polar_coord = ivy_mech.cartesian_to_polar_coords(
        cartesian_coord)

    # 3
    cartesian_coord_again = ivy_mech.polar_to_cartesian_coords(
        polar_coord)

    # cartesian co-ordinate frame-of-reference transformations

    # 3 x 4
    trans_mat = ivy.random_uniform(0., 1., (3, 4))

    # 4
    cartesian_coord_homo = ivy_mech.make_coordinates_homogeneous(
        cartesian_coord)

    # 3
    trans_cartesian_coord = ivy.matmul(
        trans_mat, ivy.expand_dims(cartesian_coord_homo, -1))[:, 0]

    # 4
    trans_cartesian_coord_homo = ivy_mech.make_coordinates_homogeneous(
        trans_cartesian_coord)

    # 4 x 4
    trans_mat_homo = ivy_mech.make_transformation_homogeneous(
        trans_mat)

    # 3 x 4
    inv_trans_mat = ivy.inv(trans_mat_homo)[0:3]

    # 3
    cartesian_coord_again = ivy.matmul(
        inv_trans_mat, ivy.expand_dims(trans_cartesian_coord_homo, -1))[:, 0]

    # message
    print('End of Run Through Demo!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default=None,
                        help='which backend to use. Chooses a random backend if unspecified.')
    parsed_args = parser.parse_args()
    fw = parsed_args.backend()
    f = None if fw is None else ivy.get_backend(fw)
    main(f, fw)
