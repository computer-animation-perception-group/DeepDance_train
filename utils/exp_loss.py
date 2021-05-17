import pickle

import tensorflow as tf

from utils import tf_expsdk as tk


def unnorm_chls(chls, config):
    mot_data_info = config.mot_data_info
    norm_way = config.norm_way
    mot_ignore_dims = config.mot_ignore_dims
    nd = mot_data_info[0].shape[0]
    all_dims = list(range(nd))
    useful_dims = [d for d in all_dims if d not in mot_ignore_dims]
    original_shape = chls.get_shape().as_list()

    if norm_way == 'no':
        print('unnorm no')
        unnorm_data = chls
    else:
        print('unnorm zscore')
        chls = tf.reshape(chls, [-1, original_shape[-1]])
        data_mean = tf.constant(mot_data_info[2][useful_dims], dtype=tf.float32, shape=[1, len(useful_dims)])
        data_std = tf.constant(mot_data_info[3][useful_dims], dtype=tf.float32, shape=[1, len(useful_dims)])
        unnorm_data = tf.multiply(chls, data_std) + data_mean
        unnorm_data = tf.reshape(unnorm_data, original_shape)

    return unnorm_data


def norm_chls(chls, config, eps=1e-6):
    mot_data_info = config.mot_data_info
    norm_way = config.norm_way
    mot_ignore_dims = config.mot_ignore_dims
    nd = mot_data_info[0].shape[0]
    all_dims = list(range(nd))
    useful_dims = [d for d in all_dims if d not in mot_ignore_dims]
    original_shape = chls.get_shape().as_list()

    if norm_way == 'no':
        print('norm no')
        norm_data = chls
    else:
        print('norm zscore')
        chls = tf.reshape(chls, [-1, original_shape[-1]])
        data_mean = tf.constant(mot_data_info[2][useful_dims], dtype=tf.float32, shape=[1, len(useful_dims)])
        data_std = tf.constant(mot_data_info[3][useful_dims], dtype=tf.float32, shape=[1, len(useful_dims)])
        norm_data = (chls - data_mean) / (data_std + eps)
        norm_data = tf.reshape(norm_data, original_shape)

    return norm_data


def _normalize(chls, axis, eps=1e-6):
    unit_chls = tf.nn.l2_normalize(chls, dim=axis, epsilon=eps)
    return unit_chls


def _dot2ang(dot_product, eps=1e-6):
    dot_product = tf.clip_by_value(dot_product, -1.+eps, 1.-eps)
    angle = tf.acos(dot_product)
    # angle = 1 - dot_product
    return angle


def mse_loss_impl(pre_chls, tru_chls):
    return tf.reduce_mean(tf.squared_difference(pre_chls, tru_chls))


def mse_trans_loss_impl(pre_chls, tru_chls):
    pre_trans_chls = pre_chls[:, :, :3]
    tru_trans_chls = tru_chls[:, :, :3]

    return tf.reduce_mean(tf.squared_difference(pre_trans_chls, tru_trans_chls))


def mse_exp_loss_impl(pre_chls, tru_chls):
    pre_exp_chls = pre_chls[:, :, 3:]
    tru_exp_chls = tru_chls[:, :, 3:]
    return tf.reduce_mean(tf.squared_difference(pre_exp_chls, tru_exp_chls))


def path_loss_impl(pre_chls, tru_chls):
    """
    path loss
    :param pre_chls: batch_size * num_steps * 79
    :param tru_chls: batch_size * num_steps * 79
    :return: loss value
    """
    print('path_loss')
    pre_root_positions = tf.stack([pre_chls[:, :, 0], pre_chls[:, :, 2]], axis=-1)
    tru_root_positions = tf.stack([tru_chls[:, :, 0], tru_chls[:, :, 2]], axis=-1)
    pre_path = pre_root_positions[:, 1:, :] - pre_root_positions[:, :-1, :]
    tru_path = tru_root_positions[:, 1:, :] - tru_root_positions[:, :-1, :]

    # pre_path = pre_root_positions - pre_root_positions[:, 0:1, :]
    # tru_path = tru_root_positions - tru_root_positions[:, 0:1, :]

    # pre_path = tf.norm(pre_path, axis=-1)
    # tru_path = tf.norm(tru_path, axis=-1)

    # path_loss = tf.squared_difference(pre_path, tru_path)
    path_loss = tf.norm(pre_path-tru_path, axis=-1)
    path_loss = tf.reduce_mean(path_loss)

    return path_loss


def height_loss_impl(pre_chls, tru_chls):
    pre_height = pre_chls[:, :, 1]
    tru_height = tru_chls[:, :, 1]
    height_loss = tf.reduce_mean(tf.abs(pre_height-tru_height))

    return height_loss


def pos_loss_impl(pre_chls, tru_chls):

    chls_shape = pre_chls.get_shape().as_list()
    num_joints = int((chls_shape[2] - 3) / 3)
    pre_pos_chls = tf.reshape(pre_chls[:, :, 3:], [chls_shape[0], chls_shape[1], num_joints, 3])
    tru_pos_chls = tf.reshape(tru_chls[:, :, 3:], [chls_shape[0], chls_shape[1], num_joints, 3])
    pos_loss = tf.norm(pre_pos_chls-tru_pos_chls, axis=-1)
    pos_loss = tf.reduce_mean(pos_loss)

    pre_delta_chls = pre_pos_chls[:, 1:, :] - pre_pos_chls[:, :-1, :]
    tru_delta_chls = tru_pos_chls[:, 1:, :] - tru_pos_chls[:, :-1, :]
    pos_delta_loss = tf.norm(pre_delta_chls-tru_delta_chls, axis=-1)
    pos_delta_loss = tf.reduce_mean(pos_delta_loss)

    return pos_loss, pos_delta_loss


def pos_delta_loss_impl(pre_chls, tru_chls):
    chls_shape = pre_chls.get_shape().as_list()
    num_joints = int((chls_shape[2] - 3) / 3)
    pre_pos_chls = tf.reshape(pre_chls[:, :, 3:], [chls_shape[0], chls_shape[1], num_joints, 3])
    tru_pos_chls = tf.reshape(tru_chls[:, :, 3:], [chls_shape[0], chls_shape[1], num_joints, 3])

    pre_delta_chls = pre_pos_chls[:, 1:, :] - pre_pos_chls[:, :-1, :]
    tru_delta_chls = tru_pos_chls[:, 1:, :] - tru_pos_chls[:, :-1, :]

    pos_loss = tf.norm(pre_delta_chls-tru_delta_chls, axis=-1)
    pos_loss = tf.reduce_mean(pos_loss)

    return pos_loss


def _get_angle(root_positions):

    path_former = root_positions[:, 1:-1, :] - root_positions[:, :-2, :]
    unit_former = _normalize(path_former, axis=-1)

    path_latter = root_positions[:, 2:, :] - root_positions[:, 1:-1, :]
    unit_latter = _normalize(path_latter, axis=-1)

    dot_product = tf.reduce_sum(tf.multiply(unit_former, unit_latter), axis=-1)
    angle = _dot2ang(dot_product)

    return angle


def dir_loss_impl(pre_chls, tru_chls):
    """
    direction loss
    :param pre_chls: batch_size * num_steps * 79
    :param tru_chls: batch_size * num_steps * 79
    :return: loss value
    """
    pre_root_positions = tf.stack([pre_chls[:, :, 0], pre_chls[:, :, 2]], axis=-1)
    tru_root_positions = tf.stack([tru_chls[:, :, 0], tru_chls[:, :, 2]], axis=-1)

    pre_angle = _get_angle(pre_root_positions)
    tru_angle = _get_angle(tru_root_positions)
    angle_dist = pre_angle - tru_angle
    dir_loss = tf.abs(angle_dist)
    dir_loss = tf.reduce_mean(dir_loss)
    return dir_loss


def _get_root_ori(chls):
    """
             CHip(Chest)
             /\
            /  \
      Rhip /____\ Lhip
    :param chls:
    :return:
    """
    c_hip_chl = chls[:, :, 1, :]
    l_hip_chl = chls[:, :, 16, :]
    r_hip_chl = chls[:, :, 20, :]

    c_ori = _normalize(tf.cross(r_hip_chl-c_hip_chl, l_hip_chl-c_hip_chl), axis=-1)
    # r_ori = tf.nn.l2_normalize(tf.cross(l_hip_chl-r_hip_chl, c_hip_chl-r_hip_chl), dim=-1)
    # l_ori = tf.nn.l2_normalize(tf.cross(c_hip_chl-l_hip_chl, r_hip_chl-l_hip_chl), dim=-1)
    # root_ori = tf.nn.l2_normalize((c_ori + r_ori + l_ori) / 3, dim=-1)

    return c_ori


def ori_loss_impl(pre_chls, tru_chls):
    """
    orientation loss
    :param pre_chls: batch_size * num_steps * 72
    :param tru_chls: batch_size * num_steps * 72
    :return: loss value
    """
    chls_shape = pre_chls.get_shape().as_list()
    num_joints = int((chls_shape[2]) / 3)
    pre_chls = tf.reshape(pre_chls, [chls_shape[0], chls_shape[1], num_joints, 3])
    tru_chls = tf.reshape(tru_chls, [chls_shape[0], chls_shape[1], num_joints, 3])

    pre_ori = _get_root_ori(pre_chls)
    tru_ori = _get_root_ori(tru_chls)

    dot_product = tf.reduce_sum(tf.multiply(pre_ori, tru_ori), axis=-1)
    angle_dist = _dot2ang(dot_product)
    ori_loss = tf.reduce_mean(tf.abs(angle_dist))

    pre_dot_product = tf.reduce_sum(tf.multiply(pre_ori[:, 1:, :], pre_ori[:, :-1, :]), axis=-1)
    pre_angle = _dot2ang(pre_dot_product)
    tru_dot_product = tf.reduce_sum(tf.multiply(tru_ori[:, 1:, :], tru_ori[:, :-1, :]), axis=-1)
    tru_angle = _dot2ang(tru_dot_product)
    ori_delta_loss = tf.reduce_mean(tf.abs(pre_angle - tru_angle))

    return ori_loss, ori_delta_loss


def get_pos_chls(pre_chls, tru_chls, config):
    mot_scale = config.mot_scale

    pre_chls = unnorm_chls(pre_chls, config)
    tru_chls = unnorm_chls(tru_chls, config)

    init_t = tf.constant([0, 0, 0], dtype=tf.float32)
    init_r = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
    pre_chls = tk.revert_coordinate_space(pre_chls, init_t, init_r)
    tru_chls = tk.revert_coordinate_space(tru_chls, init_t, init_r)

    with open('./utils/capg_exp_skel.pkl', 'rb') as fh:
        skel = pickle.load(fh)

    chls_shape = pre_chls.get_shape().as_list()
    num_joints = int((chls_shape[2] - 3) / 3)

    pre_exp_chls = tf.reshape(pre_chls[:, :, 3:], [chls_shape[0], chls_shape[1], num_joints, 3])
    # if add height in trans, will result in nan loss
    pre_trans_chls = pre_chls[:, :, :3]

    tru_exp_chls = tf.reshape(tru_chls[:, :, 3:], [chls_shape[0], chls_shape[1], num_joints, 3])
    tru_trans_chls = tru_chls[:, :, :3]

    pre_positions = tk.exp2xyz(skel, pre_exp_chls, pre_trans_chls, mot_scale)
    tru_positions = tk.exp2xyz(skel, tru_exp_chls, tru_trans_chls, mot_scale)
    pre_positions = tf.reshape(pre_positions, [chls_shape[0], chls_shape[1], -1])
    tru_positions = tf.reshape(tru_positions, [chls_shape[0], chls_shape[1], -1])

    return pre_positions, tru_positions


def loss_impl(pre_chls, tru_chls, pre_pos_chls, tru_pos_chls, config):
    rate = config.loss_rate_list

    mse_loss = mse_loss_impl(pre_chls, tru_chls)
    # pos_loss = pos_loss_impl(pre_chls, tru_chls, mot_scale)
    pos_loss, pos_delta_loss = pos_loss_impl(pre_pos_chls, tru_pos_chls)
    path_loss = path_loss_impl(pre_pos_chls, tru_pos_chls)
    height_loss = height_loss_impl(pre_pos_chls, tru_pos_chls)
    dir_loss = dir_loss_impl(pre_pos_chls, tru_pos_chls)
    ori_loss, ori_delta_loss = ori_loss_impl(pre_pos_chls, tru_pos_chls)

    path_loss = path_loss + height_loss
    loss_list = [mse_loss, pos_loss, path_loss, dir_loss, ori_loss, pos_delta_loss, ori_delta_loss]
    loss_res_list = []
    loss = tf.constant(0, dtype=tf.float32, name='loss')
    for i in range(len(loss_list)):
        if i == 0:
            rate_idx = 0
        elif 1 <= i <= 4:
            rate_idx = 1
        else:
            rate_idx = 2
        loss_res_list.append(loss_list[i])
        if rate[rate_idx] != 0:
            loss += rate[rate_idx] * loss_list[i]
    loss_res_list.append(loss)

    return loss, loss_res_list


