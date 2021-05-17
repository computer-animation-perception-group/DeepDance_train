import tensorflow as tf
import inlib.models as md
import numpy as np


class DisGraph(object):
    def __init__(self, inputs, cond_inputs, config, name, is_reuse):
        self.inputs = inputs
        self.cond_inputs = cond_inputs
        self.name = name
        self.is_reuse = is_reuse
        self.act_type = config.act_type
        self.kernel_size = config.kernel_size
        self.cond_axis = config.cond_axis
        self.stride = config.stride
        self.mus_ebd_dim = config.mus_ebd_dim
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.is_training = config.is_training
        self.is_shuffle = config.is_shuffle

    def build_dis_graph(self):
        if self.name == 'mlp':
            outputs = self._build_dis_mlp_graph()
        elif self.name == 'cnn':
            outputs = self._build_dis_cnn_graph()
        elif self.name == 'sig_cnn':
            outputs = self._build_dis_sig_cnn_graph()
        elif self.name == 'cond_cnn':
            outputs = self._build_dis_cond_cnn_graph()
        elif self.name == 'time_cond_cnn':
            outputs = self._build_dis_time_cond_cnn_graph()
        elif self.name == 'tgan_cond_cnn':
            outputs = self._build_dis_tgan_cond_cnn_graph()
        elif self.name == 'time_tgan_cond_cnn':
            outputs = self._build_dis_time_tgan_cond_cnn_graph()
        else:
            raise ValueError('Not valid discriminator name')

        return outputs

    def _build_dis_mlp_graph(self):
        return []

    def _build_dis_cnn_graph(self):
        return []

    def _build_dis_cond_cnn_graph(self):
        return []

    def _build_dis_time_cond_cnn_graph(self):
        return []

    def _build_dis_tgan_cond_cnn_graph(self):
        return []

    def _build_dis_time_tgan_cond_cnn_graph(self):
        return []

    def _build_dis_sig_cnn_graph(self):
        return []


class DisFrameGraph(DisGraph):
    def __init__(self, inputs, cond_inputs, config, name='cnn', is_reuse=False):
        super(DisFrameGraph, self).__init__(inputs, cond_inputs, config, name, is_reuse)

    def _build_dis_mlp_graph(self):
        fc_list_d = [[100, self.act_type], [256, self.act_type], [500, self.act_type], [1, '']]
        # [batch_size*num_steps, mus_ebd_dim]
        mot_input = tf.reshape(self.inputs, [-1, 60])
        outputs = md.mlp(mot_input, fc_list_d, 'discriminator', reuse=self.is_reuse)
        return outputs

    def _build_dis_cnn_graph(self):
        print('frame_cnn_graph')
        mot_input = tf.reshape(self.inputs, [-1, 20, 1, 3])
        conv_list_d = [[64, self.kernel_size, self.stride, 'SAME', self.act_type],
                       [128, self.kernel_size, self.stride, 'SAME', self.act_type]]
        fc_list_d = [[1, '']]
        outputs = md.cnn(mot_input, conv_list_d, fc_list_d, name='discriminator', reuse=self.is_reuse)
        return outputs

    def _build_dis_cond_cnn_graph(self):
        print('frame_cond_cnn_graph')
        cond_input = tf.reshape(self.cond_inputs, [self.batch_size, self.num_steps, self.mus_ebd_dim, 1])
        mot_input = tf.reshape(self.inputs, [self.batch_size, self.num_steps, self.mus_ebd_dim, 1])
        # bs * mus_ebd_dim * num_steps * 1
        mot_input = tf.transpose(mot_input, [0, 2, 1, 3])
        cond_input = tf.transpose(cond_input, [0, 2, 1, 3])
        all_input = tf.concat([mot_input, cond_input], axis=self.cond_axis, name='concat_cond')

        [batch_size, m_dim, num_steps, chl] = all_input.get_shape()
        all_input = tf.transpose(all_input, [0, 2, 1, 3])
        all_input = tf.reshape(all_input, [int(batch_size)*int(num_steps), int(m_dim), 1, int(chl)])
        print('all_input: ', all_input)

        conv_list_d = [[64, self.kernel_size, self.stride, 'SAME', self.act_type],
                       [128, self.kernel_size, self.stride, 'SAME', self.act_type]]
        fc_list_d = [[1, '']]
        outputs = md.cnn(mot_input, conv_list_d, fc_list_d, name='discriminator', reuse=self.is_reuse)
        return outputs

    def _build_dis_sig_cnn_graph(self):
        print('frame_sig_cnn_graph')
        inputs = tf.reshape(self.inputs, [-1, 20, 1, 3])
        idx_lists = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                     18, 19, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 2, 4, 6, 8, 10,
                     12, 14, 16, 18, 1, 4, 7, 10, 13, 16, 19, 3, 6, 9, 12, 15, 18,
                     2, 5, 8, 11, 14, 17, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3,
                     7, 11, 15, 19, 4, 8, 12, 16, 1, 6, 11, 16, 2, 7, 12, 17, 3,
                     8, 13, 18, 4, 9, 14, 19, 5, 10, 15, 1, 7, 13, 19, 6, 12, 18,
                     5, 11, 17, 4, 10, 16, 3, 9, 15, 2, 8, 14, 1, 8, 15, 3, 10,
                     17, 5, 12, 19, 7, 14, 2, 9, 16, 4, 11, 18, 6, 13, 1, 9, 17,
                     6, 14, 3, 11, 19, 8, 16, 5, 13, 2, 10, 18, 7, 15, 4, 12, 1,
                     10, 19, 9, 18, 8, 17, 7, 16, 6, 15, 5, 14, 4, 13, 3, 12, 2,
                     11, 1]

        # TODO: need to check
        mot_input = []
        for i, idx in enumerate(idx_lists):
            mot_input.append(inputs[:, idx, :, :])
        mot_input = tf.reshape(tf.concat(mot_input, axis=1), [-1, 173, 1, 3])
        # [3, 1], [2, 1]
        conv_list_d = [[64,  self.kernel_size, self.stride, 'SAME', self.act_type],
                       [128, self.kernel_size, self.stride, 'SAME', self.act_type],
                       [256, self.kernel_size, self.stride, 'SAME', self.act_type],
                       [512, self.kernel_size, self.stride, 'SAME', self.act_type]]

        fc_list_d = [[1, '']]
        outputs = md.cnn(mot_input, conv_list_d, fc_list_d, name='discriminator', reuse=self.is_reuse)
        return outputs


class DisSegGraph(DisGraph):
    def __init__(self, inputs, cond_inputs, config, name='mlp', is_reuse=False):
        super(DisSegGraph, self).__init__(inputs, cond_inputs, config, name, is_reuse)

    def _build_dis_mlp_graph(self):
        outputs = []
        return outputs

    def _build_dis_cnn_graph(self):
        print('seg_cnn_graph')
        mot_input = tf.reshape(self.inputs, [self.batch_size, self.num_steps, 20, 3])
        # bs * 20 * num_steps * 3
        tf.transpose(mot_input, [0, 2, 1, 3])
        # [3, 3] [2, 2]
        conv_list_d = [[64,  self.kernel_size, self.stride, 'SAME', self.act_type],
                       [128, self.kernel_size, self.stride, 'SAME', self.act_type],
                       [256, self.kernel_size, self.stride, 'SAME', self.act_type],
                       [512, self.kernel_size, self.stride, 'SAME', self.act_type]]
        fc_list_d = [[1, '']]
        outputs = md.cnn(mot_input, conv_list_d, fc_list_d, name='discriminator', reuse=self.is_reuse)
        return outputs

    def _build_dis_cond_cnn_graph(self):
        print('seg_cond_cnn_graph')
        cond_input = tf.reshape(self.cond_inputs, [self.batch_size, self.num_steps, self.mus_ebd_dim, 1])
        mot_input = tf.reshape(self.inputs, [self.batch_size, self.num_steps, self.mus_ebd_dim, 1])
        # bs * mus_ebd_dim * num_steps * 1
        # cond_input = tf.transpose(cond_input, [0, 2, 1, 3])
        # mot_input = tf.transpose(mot_input, [0, 2, 1, 3])
        all_input = tf.concat([mot_input, cond_input], axis=self.cond_axis, name='concat_cond')
        if self.is_shuffle:
            original_shape = all_input.get_shape().as_list()
            np.random.seed(1234567890)
            shuffle_list = list(np.random.permutation(original_shape[0]))
            all_inputs = []
            for i, idx in enumerate(shuffle_list):
                all_inputs.append(all_input[idx:idx+1, :, :, :])
            all_input = tf.concat(all_inputs, axis=0)
        print('all_input: ', all_input)
        # [3, 3] [2, 2]
        conv_list_d = [[32,  self.kernel_size, self.stride, 'SAME', self.act_type],
                       [64,  self.kernel_size, self.stride, 'SAME', self.act_type, 'bn'],
                       [128, self.kernel_size, self.stride, 'SAME', self.act_type, 'bn'],
                       [256, self.kernel_size, self.stride, 'SAME', self.act_type, 'bn']]
        fc_list_d = [[1, '']]
        outputs = md.cnn(all_input, conv_list_d, fc_list_d, name='discriminator', reuse=self.is_reuse)
        return outputs

    def _build_dis_time_cond_cnn_graph(self):
        print('seg_time_cond_cnn_graph')
        # bs * 1 * num_steps * 72
        cond_input = tf.reshape(self.cond_inputs, [self.batch_size, 1, self.num_steps, self.mus_ebd_dim])
        mot_input = tf.reshape(self.inputs, [self.batch_size, 1, self.num_steps, self.mus_ebd_dim])

        all_input = tf.concat([mot_input, cond_input], axis=self.cond_axis, name='concat_cond')
        if self.is_shuffle:
            original_shape = all_input.get_shape().as_list()
            np.random.seed(1234567890)
            shuffle_list = list(np.random.permutation(original_shape[0]))
            all_inputs = []
            for i, idx in enumerate(shuffle_list):
                all_inputs.append(all_input[idx:idx+1, :, :, :])
            all_input = tf.concat(all_inputs, axis=0)
        print('all_input: ', all_input)
        # [1, 3] [1, 2]
        conv_list_d = [[32,  self.kernel_size, self.stride, 'SAME', self.act_type],
                       [64,  self.kernel_size, self.stride, 'SAME', self.act_type, 'bn'],
                       [128, self.kernel_size, self.stride, 'SAME', self.act_type, 'bn'],
                       [256, self.kernel_size, self.stride, 'SAME', self.act_type, 'bn']]
        fc_list_d = [[1, '']]
        outputs = md.cnn(all_input, conv_list_d, fc_list_d, name='discriminator', reuse=self.is_reuse)
        return outputs

    def _build_dis_tgan_cond_cnn_graph(self):
        print('tgan_cond_cnn_graph')
        cond_input = tf.reshape(self.cond_inputs, [self.batch_size, self.num_steps, self.mus_ebd_dim, 1])
        mot_input = tf.reshape(self.inputs, [self.batch_size, self.num_steps, self.mus_ebd_dim, 1])
        all_input = tf.concat([mot_input, cond_input], axis=self.cond_axis, name='concat_cond')
        if self.is_shuffle:
            print('shuffle')
            original_shape = all_input.get_shape().as_list()
            np.random.seed(1234567890)
            shuffle_list = list(np.random.permutation(original_shape[0]))
            all_inputs = []
            for i, idx in enumerate(shuffle_list):
                all_inputs.append(all_input[idx:idx+1, :, :, :])
            all_input = tf.concat(all_inputs, axis=0)
        print('all_input: ', all_input)
        # [3, 3] [2, 2]
        conv_list_d = [[32,  self.kernel_size, self.stride, 'SAME', self.act_type],
                       [64,  self.kernel_size, self.stride, 'SAME', self.act_type, 'bn'],
                       [128, self.kernel_size, self.stride, 'SAME', self.act_type, 'bn'],
                       [256, self.kernel_size, self.stride, 'SAME', self.act_type, 'bn']]
        outputs = md.cnn(all_input, conv_list_d, [], name='discriminator',
                         is_training=self.is_training, reuse=self.is_reuse)
        return outputs

    def _build_dis_time_tgan_cond_cnn_graph(self):
        print('time_tgan_cond_cnn_graph')
        # bs * 1 * num_steps * 72
        cond_input = tf.reshape(self.cond_inputs, [self.batch_size, 1, self.num_steps, self.mus_ebd_dim])
        mot_input = tf.reshape(self.inputs, [self.batch_size, 1, self.num_steps, self.mus_ebd_dim])

        all_input = tf.concat([mot_input, cond_input], axis=self.cond_axis, name='concat_cond')
        if self.is_shuffle:
            original_shape = all_input.get_shape().as_list()
            np.random.seed(1234567890)
            shuffle_list = list(np.random.permutation(original_shape[0]))
            all_inputs = []
            for i, idx in enumerate(shuffle_list):
                all_inputs.append(all_input[idx:idx+1, :, :, :])
            all_input = tf.concat(all_inputs, axis=0)
        print('all_input: ', all_input)
        # [1, 3] [1, 2]
        conv_list_d = [[32,  self.kernel_size, self.stride, 'SAME', self.act_type],
                       [64,  self.kernel_size, self.stride, 'SAME', self.act_type, 'bn'],
                       [128, self.kernel_size, self.stride, 'SAME', self.act_type, 'bn'],
                       [256, self.kernel_size, self.stride, 'SAME', self.act_type, 'bn']]
        outputs = md.cnn(all_input, conv_list_d, [], name='discriminator', reuse=self.is_reuse)
        return outputs


