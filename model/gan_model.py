from model.base_model import *
from model import discriminator_model as dm
from utils import exp_loss as es, gan_loss as gls


class GanModel(BaseModel):
    """The Generative adversarial model"""
    def __init__(self, train_type, config):
        super(GanModel, self).__init__(train_type, config)

        mot_predictions = self.mot_predictions
        mot_truth = self.mot_truth
        mus_ebd_outputs = self.mus_ebd_outputs

        tru_pos, pre_pos = es.get_pos_chls(mot_predictions, mot_truth, config)

        dis_name = config.dis_name
        dis_graph = getattr(dm, config.dis_type)

        if self.mus_ebd_dim == 60:
            real_data = mot_truth
            fake_data = mot_predictions
        elif self.mus_ebd_dim == 72:
            real_data = tru_pos
            fake_data = pre_pos
        else:
            real_data = tf.concat([mot_truth, tru_pos], axis=-1)
            fake_data = tf.concat([mot_predictions, pre_pos], axis=-1)

        print('real_data:', real_data)
        print('fake_data:', fake_data)

        g_sig_loss, d_loss, clip_d_weights = \
            gls.gan_loss(dis_graph, dis_name, real_data=real_data, fake_data=fake_data,
                         cond_inputs=mus_ebd_outputs, config=config)

        # generator loss
        g_loss, loss_list = es.loss_impl(mot_predictions, mot_truth, pre_pos, tru_pos, config)
        # g_mse_loss = tf.reduce_mean(tf.squared_difference(mot_predictions, mot_truth),
        #                             name='mean_square_loss')
        g_loss = config.mse_rate * g_loss + config.dis_rate * g_sig_loss
        self.g_loss = [loss_list, g_sig_loss]
        self.d_loss = d_loss

        # if test, return
        if not self.is_training:
            return

        tvars = tf.trainable_variables()
        d_vars = [v for v in tvars if 'discriminator' in v.name]
        g_vars = [v for v in tvars if 'generator' in v.name]

        # add reg
        if config.is_reg:
            reg_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in g_vars
                                     if 'bias' not in v.name]) * config.reg_scale
            g_loss = g_loss + reg_cost

        gen_learning_rate = config.learning_rate
        dis_learning_rate = config.dis_learning_rate

        if config.optimizer.lower() == 'adam':
            print('Adam optimizer')
            g_optimizer = tf.train.AdamOptimizer(learning_rate=gen_learning_rate)
            d_optimizer = tf.train.AdamOptimizer(learning_rate=dis_learning_rate)
        else:
            print('Rmsprop optimizer')
            g_optimizer = tf.train.RMSPropOptimizer(learning_rate=gen_learning_rate)
            d_optimizer = tf.train.RMSPropOptimizer(learning_rate=dis_learning_rate)

        # for batch_norm op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        g_grads = tf.gradients(g_loss, g_vars, aggregation_method=2)
        d_grads = tf.gradients(d_loss, d_vars, aggregation_method=2)
        with tf.control_dependencies(update_ops):
            self.train_g_op = g_optimizer.apply_gradients(zip(g_grads, g_vars))
        print('train_g_op')

        if clip_d_weights:
            with tf.control_dependencies([clip_d_weights, update_ops]):
                self.train_d_op = d_optimizer.apply_gradients(zip(d_grads, d_vars))
                # self._train_d_op = optimizer.minimize(d_loss, var_list=d_vars)
        else:
            with tf.control_dependencies(update_ops):
                self.train_d_op = d_optimizer.apply_gradients(zip(d_grads, d_vars))
        print('train_d_op')