import tensorflow as tf

from tensorflow.contrib import layers


def fc_mnist_encoder(x, latent_dim):
    """
    Inference network q(z|x) which encodes a mini batch of data points
    to a parameterization of a diagonal Gaussian using a network with 
    fully connected layers.

    :param x: Mini batch of data points to encode.
    :param latent_dim: dimension of latent space into which we encode
    :return: e: Encoded mini batch.
    """
    e = layers.fully_connected(x, 500, scope='fc-01')
    e = layers.fully_connected(e, 500, scope='fc-02')
    e = layers.fully_connected(e, 200, scope='fc-03')
    e = layers.fully_connected(e, 2 * latent_dim, activation_fn=None,
                               scope='fc-final')

    return e


def fc_mnist_decoder(z):
    """
    Generative network p(x|z) which decodes a sample z from
    the latent space using a network with fully connected layers.
    
    :param z: Latent variable sampled from latent space.
    :return: x: Decoded latent variable.
    """
    x = layers.fully_connected(z, 200, scope='fc-01')
    x = layers.fully_connected(x, 500, scope='fc-02')
    x = layers.fully_connected(x, 500, scope='fc-03')
    x = layers.fully_connected(x, 28 * 28, activation_fn=tf.nn.sigmoid,
                               scope='fc-final')

    return x


def conv_mnist_encoder(x, latent_dim):
    """
    Inference network q(z|x) which encodes a mini batch of data points
    to a parameterization of a diagonal Gaussian using a network with 
    convolutional layers.

    :param x: Mini batch of data points to encode.
    :param latent_dim: dimension of latent space into which we encode
    :return: e: Encoded mini batch.
    """
    e = tf.reshape(x, [-1, 28, 28, 1])
    e = layers.conv2d(e, 32, 5, stride=2, scope='conv-01')
    e = layers.conv2d(e, 64, 5, stride=2, scope='conv-02')
    e = layers.conv2d(e, 128, 3, stride=2, scope='conv-03')
    e = layers.flatten(e)
    e = layers.fully_connected(e, 500, scope='fc-01')
    e = layers.fully_connected(e, 200, scope='fc-02')
    e = layers.fully_connected(e, 2 * latent_dim, activation_fn=None,
                               scope='fc-final')

    return e


def conv_mnist_decoder(z):
    """
    Generative network p(x|z) which decodes a sample z from
    the latent space using a network with convolutional layers.
    
    :param z: Latent variable sampled from latent space.
    :return: x: Decoded latent variable.
    """
    x = tf.expand_dims(z, 1)
    x = tf.expand_dims(x, 1)
    x = layers.conv2d_transpose(x, 128, 3, padding='VALID', scope='conv-transpose-01')
    x = layers.conv2d_transpose(x, 64, 5, padding='VALID', scope='conv-transpose-02')
    x = layers.conv2d_transpose(x, 32, 5, stride=2, scope='conv-transpose-03')
    x = layers.conv2d_transpose(x, 1, 5, stride=2, activation_fn=tf.nn.sigmoid,
                                scope='conv-transpose-final')
    x = layers.flatten(x)

    return x
