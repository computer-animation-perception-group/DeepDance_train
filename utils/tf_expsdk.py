import tensorflow as tf
import numpy as np


def exp_mul(e0, e1):
    original_shape = e0.get_shape().as_list()
    e0 = tf.reshape(e0, [-1, 3])
    e1 = tf.reshape(e1, [-1, 3])
    r0 = exp2rot(e0)
    r1 = exp2rot(e1)
    rm = tf.matmul(r0, r1)
    em = rot2exp(rm)
    em = tf.reshape(em, original_shape)

    return em


def rot2exp(r):
    return quat2exp(rot2quat(r))


def rot2quat(r, eps=1e-8):
    original_shape = r.get_shape().as_list()
    out_shape = original_shape[:-1]
    out_shape[-1] = 4
    r = tf.reshape(r, [-1, 3, 3])
    d = r - tf.transpose(r, [0, 2, 1])
    r_ = tf.stack([-d[:, 1, 2], d[:, 0, 2], -d[:, 0, 1]], axis=-1)
    sin_theta = tf.norm(r_, axis=-1) / 2
    r0 = tf.divide(r_, tf.norm(r_, axis=-1, keep_dims=True) + eps)
    cos_theta = (tf.trace(r) - 1) / 2
    theta = tf.atan2(sin_theta, cos_theta)
    theta = tf.reshape(theta, [-1, 1])
    w = tf.cos(theta / 2)
    v = r0 * tf.sin(theta / 2)
    q = tf.concat([w, v], axis=-1)
    q = tf.reshape(q, out_shape)
    return q


def revert_coordinate_space(chls, init_t, init_r):
    org_shape = chls.get_shape().as_list()

    init_t = tf.reshape(init_t, [1, 3])
    init_r = tf.reshape(init_r, [1, 3, 3])
    init_r = tf.tile(init_r, [org_shape[0], 1, 1])
    r_prev = init_r
    t_prev = init_t
    rec_chls = []
    for i in range(org_shape[1]):
        print(i)
        r_diff = exp2rot(chls[:, i, 3:6])
        r = tf.matmul(r_diff, r_prev)
        rec_r = rot2exp(r)
        rec_t = t_prev + tf.squeeze(tf.matmul(tf.transpose(r_prev, [0, 2, 1]),
                                    tf.transpose(chls[:, i:i+1, :3], [0, 2, 1])))
        rec_frame = tf.concat([rec_t, rec_r, chls[:, i, 6:]], axis=-1)
        rec_chls.append(rec_frame)

        t_prev = rec_t
        r_prev = r

    rec_chls = tf.stack(rec_chls, axis=1)
    print('revert_coordinate_space')
    return rec_chls


def quat2exp(q, eps=1e-8):
    original_shape = q.get_shape().as_list()
    out_shape = original_shape.copy()
    out_shape[-1] = int(out_shape[-1] / 4 * 3)

    q = tf.reshape(q, [-1, 4])
    sin_half_theta = tf.norm(q[:, 1:], axis=-1)
    cos_half_theta = q[:, 0]
    r0 = q[:, 1:] / (tf.norm(q[:, 1:], axis=-1, keep_dims=True) + eps)
    theta = 2 * tf.atan2(sin_half_theta, cos_half_theta)
    theta = tf.mod(theta + 2 * np.pi, 2 * np.pi)
    pi = tf.constant(np.pi, dtype=tf.float32, shape=theta.get_shape().as_list())
    theta = tf.where(theta > pi,  2 * np.pi - theta,  theta)
    r0 = tf.where(theta > pi,  -r0,  r0)

    e = r0 * tf.reshape(theta, [-1, 1])
    e = tf.reshape(e, out_shape)
    return e


def exp2rot(e, eps=1e-32):
    original_shape = e.get_shape().as_list()
    out_shape = original_shape.copy()
    out_shape.extend([3])

    e = tf.reshape(e, [-1, 3])
    theta = tf.norm(e, axis=-1, keep_dims=True)
    r0 = tf.divide(e, theta + eps)

    c_0 = tf.constant(0, dtype=tf.float32, shape=[e.get_shape().as_list()[0]])
    # row0 = tf.stack([c_0, -r0[:, 2], r0[:, 1]], axis=1)
    # row1 = tf.stack([c_0, c_0, -r0[:, 0]], axis=1)
    # row2 = tf.stack([c_0, c_0, c_0], axis=1)
    # r0x = tf.stack([row0, row1, row2], axis=1)

    c0 = tf.stack([c_0, c_0, c_0], axis=1)
    c1 = tf.stack([-r0[:, 2], c_0, c_0], axis=1)
    c2 = tf.stack([r0[:, 1], -r0[:, 0], c_0], axis=1)
    r0x = tf.stack([c0, c1, c2], axis=2)

    r0x = r0x - tf.transpose(r0x, perm=[0, 2, 1])

    eye_matrix = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
    rot_0 = tf.reshape(eye_matrix, [1, 3, 3])
    rot_1 = tf.reshape(tf.sin(theta), [-1, 1, 1]) * r0x
    rot_2 = tf.matmul(tf.reshape(1-tf.cos(theta), [-1, 1, 1]) * r0x, r0x)
    rot = rot_0 + rot_1 + rot_2

    rot = tf.reshape(rot, out_shape)
    return rot


def rotation_matrix(x_angle, y_angle, z_angle, order='zxy'):
    # TODO: only 'zxy' implementation for now
    # order = order.lower()
    original_shape = x_angle.get_shape().as_list()
    out_shape = original_shape.copy()
    out_shape.extend([3, 3])

    x_angle = tf.reshape(x_angle, [-1])
    y_angle = tf.reshape(y_angle, [-1])
    z_angle = tf.reshape(z_angle, [-1])

    c1 = tf.cos(x_angle)
    c2 = tf.cos(y_angle)
    c3 = tf.cos(z_angle)
    s1 = tf.sin(x_angle)
    s2 = tf.sin(y_angle)
    s3 = tf.sin(z_angle)

    r0 = tf.stack([c2*c3-s1*s2*s3, c2*s3+s1*s2*c3, -s2*c1], axis=1)
    r1 = tf.stack([-c1*s3, c1*c3, s1], axis=1)
    r2 = tf.stack([s2*c3+c2*s1*s3, s2*s3-c2*s1*c3, c2*c1], axis=1)
    rm = tf.stack([r0, r1, r2], axis=1)

    rm = tf.reshape(rm, original_shape)
    return rm


def rot_vector(r, v):
    original_shape = v.get_shape().as_list()
    r = tf.reshape(r, [-1, 3, 3])
    v = tf.reshape(v, [-1, 1, 3])
    rv = tf.matmul(v, r)
    return tf.reshape(rv, original_shape)


def rot_mul(r0, r1):
    original_shape = r0.get_shape().as_list()
    r0 = tf.reshape(r0, [-1, 3, 3])
    r1 = tf.reshape(r1, [-1, 3, 3])

    r = tf.matmul(r0, r1)
    return tf.reshape(r, original_shape)


def exp2xyz(skel, rotations, root_positions, scale):
    """
    :param skel: capg skel
    :param rotations: batch_size * num_steps * num_joints * 3
    :param root_positions: batch_size * num_steps * 3
    :param scale: meter, scale = 100.0
    :return: positions: batch_size * num_steps * num_joints * 3
    """
    positions_world = []
    rotations_world = []
    rot_shape = rotations.get_shape().as_list()

    for i in range(len(skel)):
        if i == 0:
            this_pos = root_positions
            this_rot = exp2rot(rotations[:, :, 0, :])
        else:
            parent = skel[i].parent
            offset = tf.constant(np.asarray(skel[i].offset) / scale, dtype=tf.float32)
            offset = tf.expand_dims(offset, 0)
            offset = tf.expand_dims(offset, 0)
            offset = tf.tile(offset, [rot_shape[0], rot_shape[1], 1])
            this_pos = rot_vector(rotations_world[parent], offset) + positions_world[parent]
            if skel[i].quat_idx:
                # print(skel[i].quat_idx)
                this_rot = exp2rot(rotations[:, :, skel[i].quat_idx, :])
                this_rot = rot_mul(this_rot, rotations_world[parent])
            else:
                this_rot = None

        positions_world.append(this_pos)
        rotations_world.append(this_rot)

    points = tf.stack(positions_world, axis=3)
    points = tf.transpose(points, [0, 1, 3, 2])

    return points
