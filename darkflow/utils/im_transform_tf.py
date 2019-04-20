import tensorflow as tf

def imcv2_recolor(ims, a = .1):
    channel_noise = tf.random_uniform(tf.shape(ims), -1, 1)
    ims = ims * (1 + channel_noise * a)
    mx = 255 * (1 + a)

    up = tf.reshape(tf.random_uniform(tf.shape(ims)[:1], 0.5, 1.5), [-1, 1, 1, 1])
    ims = (ims * (1./mx))**up
    return tf.cast(ims * 255., tf.uint8)
