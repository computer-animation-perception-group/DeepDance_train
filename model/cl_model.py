from model.base_model import *
from utils import exp_loss as es


class CLModel(BaseModel):
    def __init__(self, train_type, config):
        super(CLModel, self).__init__(train_type, config)

        mot_predictions = self.mot_predictions
        mot_truth = self.mot_truth

        tru_pos, pre_pos = es.get_pos_chls(mot_predictions, mot_truth, config)

        # generator loss
        g_loss, loss_list = es.loss_impl(mot_predictions, mot_truth, pre_pos, tru_pos, config)
        self.g_loss = loss_list

        # if test, return
        if not self.is_training:
            return

        tvars = tf.trainable_variables()
        g_vars = [v for v in tvars if 'generator' in v.name]

        # add reg
        if config.is_reg:
            reg_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in g_vars
                                     if 'bias' not in v.name]) * config.reg_scale
            g_loss = g_loss + reg_cost

        gen_learning_rate = config.learning_rate

        if config.optimizer.lower() == 'adam':
            print('Adam optimizer')
            g_optimizer = tf.train.AdamOptimizer(learning_rate=gen_learning_rate)
        else:
            print('Rmsprop optimizer')
            g_optimizer = tf.train.RMSPropOptimizer(learning_rate=gen_learning_rate)

        # for batch_norm op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        g_grads = tf.gradients(g_loss, g_vars, aggregation_method=2)
        with tf.control_dependencies(update_ops):
            self.train_g_op = g_optimizer.apply_gradients(zip(g_grads, g_vars))
        print('train_g_op')
