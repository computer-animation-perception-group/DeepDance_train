import argparse
import copy
import json
import os
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
from model.gan_model import *

import m2m_config as cfg
from utils import reader as rd


def run_epoch(session, model, data_info, config, teacher_forcing_ratio,
              path=None, train_type=0, verbose=False, epoch=0):
    """Runs the model on the given data"""
    d_losses = 0.0
    g_exp_losses = 0.0
    g_losses = 0.0

    mot_state = session.run(model.mot_initial_state)
    mus_state = session.run(model.mus_initial_state)

    step = 0
    np.random.seed(123456789)

    for batch_x, batch_y, batch_f in rd.capg_seq_generator(epoch, train_type, data_info, config):
        feed_dict = dict()
        if config.is_use_pre_mot:
            for i, (c, h) in enumerate(model.mot_initial_state):
                feed_dict[c] = mot_state[i].c
                feed_dict[h] = mot_state[i].h

            for i, (c, h) in enumerate(model.mus_initial_state):
                feed_dict[c] = mus_state[i].c
                feed_dict[h] = mus_state[i].h

        tf_mask = np.random.uniform(size=config.num_steps) < teacher_forcing_ratio
        # print(tf_mask)
        feed_dict[model.tf_mask] = tf_mask

        last_step_mot = copy.deepcopy(batch_f)
        last_step_mot[:, :6] = 0

        feed_dict[model.init_step_mot] = last_step_mot
        feed_dict[model.input_x] = batch_x
        feed_dict[model.input_y] = batch_y

        g_fetches = {
            "last_step_mot": model.last_step_mot,
            "g_loss": model.g_loss,
            "eval_op": model.train_g_op
        }

        d_fetches = {
            "d_loss": model.d_loss,
            "eval_op": model.train_d_op
        }

        d_vals = session.run(d_fetches, feed_dict)
        g_vals = session.run(g_fetches, feed_dict)

        d_loss = d_vals["d_loss"]
        g_loss = g_vals["g_loss"]

        d_losses += d_loss
        g_exp_losses += g_loss[0][-1]
        g_losses += g_loss[1]
        step += 1

        if verbose:
            info = "Epoch {0}: {1} d_loss: {2} g_loss: {3}, exp_loss: {4}\n".format(
                epoch, step, d_loss, g_loss[1], g_loss[0])
            print(info)
            with open(path, 'a') as fh:
                fh.write(info)

    return [d_losses/step, g_losses/step, g_exp_losses/step]


def generate_motion(session, model, data_info, gen_str, test_config, hop, epoch=0,
                    time_dir=None, use_pre_mot=True, prefix='test', is_save=True):
    """Runs the model on the given data"""
    g_exp_losses = 0.0
    g_losses = 0.0
    d_losses = 0.0

    fetches = {
        "prediction": model.mot_predictions,
        "last_step_mot": model.last_step_mot,
        "g_loss": model.g_loss,
        "d_loss": model.d_loss,
        "mot_final_state": model.mot_final_state,
        "mus_final_state": model.mus_final_state,
    }

    step = 0
    num_steps = test_config.num_steps
    pre_mot = []
    mus_data = data_info[gen_str][0]
    mot_data = copy.deepcopy(data_info[gen_str][1])

    seq_keys = list(mus_data.keys())
    seq_keys.sort()
    mus_delay = test_config.mus_delay

    for file_name in seq_keys:
        predictions = []
        mus_file_data = mus_data[file_name]
        mot_file_data = mot_data[file_name]
        test_len = min(mus_file_data.shape[1]+mus_delay, mot_file_data.shape[1])
        test_num = int((test_len - 1 - num_steps) / hop + 1)

        mot_state = session.run(model.mot_initial_state)
        mus_state = session.run(model.mus_initial_state)

        for t in range(test_num):
            batch_x = mus_file_data[:, t * hop + 1 - mus_delay: t * hop + num_steps + 1 - mus_delay, :]
            batch_y = mot_file_data[:, t * hop + 1: t * hop + num_steps + 1, :]
            batch_f = mot_file_data[:, t * hop, :]  # first frame

            feed_dict = dict()
            if use_pre_mot:
                for i, (c, h) in enumerate(model.mot_initial_state):
                    feed_dict[c] = mot_state[i].c
                    feed_dict[h] = mot_state[i].h

                for i, (c, h) in enumerate(model.mus_initial_state):
                    feed_dict[c] = mus_state[i].c
                    feed_dict[h] = mus_state[i].h

            if t > 0 and use_pre_mot:
                last_step_mot = copy.deepcopy(pre_mot)
            else:
                last_step_mot = copy.deepcopy(batch_f)
                last_step_mot[:, :6] = 0

            feed_dict[model.init_step_mot] = last_step_mot
            feed_dict[model.input_x] = batch_x
            feed_dict[model.input_y] = batch_y
            feed_dict[model.tf_mask] = [False] * test_config.num_steps

            vals = session.run(fetches, feed_dict)

            prediction = vals["prediction"]
            g_loss = vals["g_loss"]
            d_loss = vals["d_loss"]
            mot_state = vals["mot_final_state"]
            mus_state = vals["mus_final_state"]
            pre_mot = vals["last_step_mot"]

            d_losses += d_loss
            g_exp_losses += g_loss[0][-1]
            g_losses += g_loss[1]

            step += 1
            prediction = np.reshape(prediction, [test_config.num_steps, test_config.mot_dim])
            predictions.append(prediction)

        if is_save and ((epoch+1) % test_config.save_data_epoch == 0 or epoch == 0):
            test_pred_path = os.path.join(time_dir, prefix, str(epoch+1), file_name + ".csv")
            if len(predictions):
                predictions = np.concatenate(predictions, 0)
                rd.save_predict_data(predictions, test_pred_path, data_info,
                                     test_config.norm_way, test_config.mot_ignore_dims,
                                     test_config.mot_scale)

    return [d_losses/step, g_losses/step, g_exp_losses/step]


def save_arg(config, path):
    config_dict = dict()
    for name, value in vars(config).items():
        config_dict[name] = value
    json.dump(config_dict, open(path, 'w'), indent=4, sort_keys=True)


def run_main(config, test_config, data_info):

    with tf.Graph().as_default():
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None):
                train_model = GanModel(config=config,
                                       train_type=0)

        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=True):
                test_model = GanModel(config=test_config,
                                      train_type=2)

        # allowing gpu memory growth
        gpu_config = tf.ConfigProto()
        saver = tf.train.Saver(max_to_keep=20)
        gpu_config.gpu_options.allow_growth = True

        with tf.Session(config=gpu_config) as session:

            # initialize all variables
            if config.is_load_model:
                saver.restore(session, config.model_path)
            else:
                session.run(tf.global_variables_initializer())

            # start queue
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=session, coord=coord)

            time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            save_dir = config.save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_save_dir = os.path.join(save_dir, 'model')
            if not os.path.exists(save_dir):
                os.makedirs(model_save_dir)
            train_loss_dict = OrderedDict()
            test_loss_dict = OrderedDict()
            start_time = time.time()
            train_loss_path = os.path.join(save_dir, "train_loss.txt")
            train_step_loss_path = os.path.join(save_dir, "train_step_loss.txt")
            config_path = os.path.join(save_dir, "config.txt")
            time_path = os.path.join(save_dir, "time.txt")
            config.save_config(config_path)
            arg_path = os.path.join(save_dir, "args.txt")
            save_arg(args, arg_path)

            teacher_forcing_ratio = config.teacher_forcing_ratio

            for i in range(config.max_max_epoch):
                train_loss = \
                    run_epoch(session, train_model,
                              data_info, config,
                              teacher_forcing_ratio,
                              path=train_step_loss_path,
                              train_type=0,
                              epoch=i,
                              verbose=True)

                print("---Epoch {0} train_loss: {1}\n".format(i, train_loss))
                train_loss_dict[str(i+1)] = train_loss
                json.dump(train_loss_dict, open(train_loss_path, 'w'), indent=4)

                if (i + 1) % test_config.save_data_epoch == 0:
                    _ = \
                        generate_motion(session, test_model,
                                        data_info, 'test', test_config, hop=test_config.num_steps,
                                        epoch=i, time_dir=save_dir,
                                        use_pre_mot=True, prefix='seq')

                    if test_config.is_save_train:
                        _ = \
                            generate_motion(session, test_model,
                                            data_info, 'train', test_config, hop=test_config.num_steps,
                                            epoch=i, time_dir=save_dir,
                                            use_pre_mot=True, prefix='seq_train')

                if (i == 0 or (i + 1) % config.save_model_epoch == 0) and config.is_save_model:
                    model_save_path = os.path.join(model_save_dir, 'cnn-erd_'+str(i)+'_model.ckpt')
                    saver.save(session, model_save_path)

                time_info = "Epoch: {0} Elapsed Time : {1}\n".format(i + 1, time.time()-start_time)
                print(time_info)
                with open(time_path, 'a') as fh:
                    fh.write(time_info)

                teacher_forcing_ratio *= config.tf_decay

            coord.request_stop()
            coord.join()


def main(_):
    type_list = args.type_list
    fold_list = args.fold_list
    seg_list = args.seg_list

    for seg_len in seg_list:
        for fold_idx in fold_list:
            for i, m_type in enumerate(type_list):
                seg_str = str(seg_len)
                fold_str = 'fold_' + str(fold_idx)
                print(m_type, seg_str, fold_str)
                if fold_idx != 0 and m_type in ['hiphop', 'salsa']:
                    continue
                if fold_idx == 3 and m_type == 'groovenet':
                    continue
                config = cfg.get_config(m_type, fold_str, seg_str)
                cfg_list = []
                care_list = ['add_info', 'mse_rate', 'dis_rate', 'dis_learning_rate',
                             'reg_scale', 'rnn_keep_list', 'is_reg', 'cond_axis']
                for k, v in sorted(vars(args).items()):
                    print(k, v)
                    setattr(config, k, v)
                    if k in care_list:
                        v_str = str(v)
                        if isinstance(v, bool):
                            v_str = v_str[0]
                        cfg_list.append(v_str)
                config.save_dir = os.path.join(args.add_info, m_type)

                args.care_list = care_list
                test_config = copy.deepcopy(config)
                test_config.batch_size = config.test_batch_size
                test_config.num_steps = config.test_num_steps

                print(config.save_dir)
                data_info = rd.run_all(config)
                config.mot_data_info = data_info['mot']
                test_config.mot_data_info = data_info['mot']
                run_main(config, test_config, data_info)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--add_info', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--is_load_model', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--fold_list', nargs='+', type=int,  help='0, 1, 2, 3')
    parser.add_argument('--seg_list', nargs='+',  type=int, help='150, 90')
    parser.add_argument('--type_list', nargs='+', type=str, help='music type')
    parser.add_argument('--model_path', type=str, help='model_path')
    parser.add_argument('--max_max_epoch', type=int, help='training epoch number')
    parser.add_argument('--save_model_epoch', type=int, help='save_model_epoch_number')
    parser.add_argument('--save_data_epoch', type=int, help='save_data_epoch_number')
    parser.add_argument('--is_reg', type=lambda x: (str(x).lower() == 'true'), help='if add regularization')
    parser.add_argument('--reg_scale', type=float, help='5e-4')
    parser.add_argument('--rnn_keep_list', nargs='+', type=float, help='rnn_keep_probability list, [1.0, 1.0, 1.0]')
    parser.add_argument('--batch_size', type=int, help='32 or 64')
    parser.add_argument('--has_random_seed', type=lambda x: (str(x).lower() == 'true'), help='')
    parser.add_argument('--teacher_forcing_ratio', type=float, help='')
    parser.add_argument('--tf_decay', type=float, help='')
    parser.add_argument('--norm_way', type=str, help='zscore, maxmin, no')
    parser.add_argument('--seq_shift', type=int, help='seq_shift')
    parser.add_argument('--gen_hop', type=int, help='gen_hop')
    parser.add_argument('--mot_scale', type=float, help='motion scale')
    parser.add_argument('--is_save_train', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--cond_axis', type=int, help='1: height, 3: channel', default=3)
    parser.add_argument('--act_type', type=str, default='lrelu')
    parser.add_argument('--kernel_size', nargs='+', type=int)
    parser.add_argument('--stride', nargs='+', type=int)
    parser.add_argument('--dis_learning_rate', type=float, default=1e-4)
    parser.add_argument('--dis_type', type=str, help='DisFrameGraph or DisSegGraph')
    parser.add_argument('--dis_name', type=str, default='cond_cnn')
    parser.add_argument('--mse_rate', type=float, default=0.99)
    parser.add_argument('--dis_rate', type=float, default=0.01)
    parser.add_argument('--loss_mode', type=str, default='gan')
    parser.add_argument('--clip_value', type=float, default=0.01)
    parser.add_argument('--pen_lambda', type=float, default=10)
    parser.add_argument('--mus_ebd_dim', type=int)
    parser.add_argument('--is_all_norm', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--loss_rate_list', nargs='+', type=float, default=[1., 0., 0.])

    args = parser.parse_args()

    tf.app.run()
