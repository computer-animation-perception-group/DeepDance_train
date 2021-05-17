import tensorflow as tf
import inlib.models as md


def gan_loss(dis_graph, dis_name, real_data, fake_data, cond_inputs, config):
    config.is_shuffle = False
    d_real_model = dis_graph(real_data, cond_inputs, name=dis_name, is_reuse=False, config=config)
    config.is_shuffle = True
    d_fake_model = dis_graph(fake_data, cond_inputs, name=dis_name, is_reuse=True, config=config)
    real_logits = d_real_model.build_dis_graph()
    fake_logits = d_fake_model.build_dis_graph()

    mode = config.loss_mode
    clip_disc_weights = []

    if mode == 'wgan':
        gen_loss = -tf.reduce_mean(fake_logits)
        disc_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

        clip_ops = []
        disc_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        for var in disc_vars:
            if not hasattr(config, 'clip_value'):
                raise ValueError('wgan must set the clip_value argument!')
            clip_value = config.clip_value
            clip_bounds = [-clip_value, clip_value]
            clip_ops.append(
                tf.assign(var, tf.clip_by_value(
                    var, clip_bounds[0], clip_bounds[1]))
            )
        clip_disc_weights = tf.group(*clip_ops)

    elif mode == 'wgan-gp':
        gen_loss = -tf.reduce_mean(fake_logits)
        disc_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

        alpha = tf.random_uniform(
            shape=[real_data.get_shape()[0].value, 1, 1], minval=0., maxval=1.)
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        gradients = tf.gradients(dis_graph(interpolates, cond_inputs, config=config,
                                           name=dis_name, is_reuse=True).build_dis_graph(),
                                 [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        if not hasattr(config, 'pen_lambda'):
            raise ValueError('wgan-gp must have lambda argument')
        disc_loss += config.pen_lambda * gradient_penalty

    elif mode == 'gan':
        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits, labels=tf.ones_like(fake_logits)))
        disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits, labels=tf.zeros_like(fake_logits)))
        disc_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_logits, labels=tf.ones_like(real_logits)))
        disc_loss /= 2.0

    elif mode == 'rsgan':
        gen_loss = tf.reduce_mean(-tf.log(tf.sigmoid(fake_logits - real_logits) + 1e-9))
        disc_loss = tf.reduce_mean(-tf.log(tf.sigmoid(real_logits - fake_logits) + 1e-9))

    elif 'tgan' in mode:
        x_real_fake = tf.subtract(real_logits, fake_logits)
        x_fake_real = tf.subtract(fake_logits, real_logits)
        fc_list = [[1, '']]
        x_real_fake_score = md.mlp(x_real_fake, fc_list, 'discriminator', reuse=False)
        x_fake_real_score = md.mlp(x_fake_real, fc_list, 'discriminator', reuse=True)
        loss_type = mode.split('-')[1]
        gen_loss = tgan_gen_loss(loss_type, real=x_real_fake_score, fake=x_fake_real_score)
        disc_loss = tgan_diss_loss(loss_type, real=x_real_fake_score, fake=x_fake_real_score)
    else:
        raise ValueError('Not implemented loss mode.')

    return gen_loss, disc_loss, clip_disc_weights


def tgan_diss_loss(loss_type, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_type == 'wgan':
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if loss_type == 'lsgan':
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if loss_type == 'sgan' or loss_type == 'dragan':
        print('sgan')
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real)+1e-9)
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake)+1e-9)

    if loss_type == 'hinge':
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss


def tgan_gen_loss(loss_type, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_type == 'wgan':
        real_loss = tf.reduce_mean(real)
        fake_loss = -tf.reduce_mean(fake)

    if loss_type == 'lsgan':
        real_loss = tf.reduce_mean(tf.square(real))
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if loss_type == 'sgan' or loss_type == 'dragan':
        print('sgan')
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real), logits=real)+1e-9)
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake)+1e-9)

    if loss_type == 'hinge':
        fake_loss = -tf.reduce_mean(fake)

    loss = real_loss + fake_loss

    return loss

