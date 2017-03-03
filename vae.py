import numpy as np
import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope


class VAE(object):

    def __init__(self, latent_dim, batch_size,
                 encoder_architecture,
                 decoder_architecture):
        """
        Implementation of Variational Autoencoder (VAE) for  MNIST,
        as outlined in https://arxiv.org/abs/1312.6114.

        :param latent_dim: (int) Dimension of latent space.
        :param batch_size: (int) Number of data points per mini batch.
        :param encoder_architecture: (str) Which encoder architecture to use.
            One of 'fc' or 'conv'.
        :param decoder_architecture: (str) Which decoder architecture to use.
            One of 'fc' or 'conv'.
        """
        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._encoder_architecture = encoder_architecture
        self._decoder_architecture = decoder_architecture
        self._build_graph()

    def _build_graph(self):
        """
        Build tensorflow computational graph for VAE.
        x -> encode(x) -> latent parameterization & KL divergence ->
        z -> decode(z) -> distribution over x -> reconstruction loss ->
        total loss & train step
        """
        # placeholder for MNIST inputs
        self.x = tf.placeholder(tf.float32, shape=[None, 28 * 28])

        with arg_scope([layers.fully_connected,
                        layers.conv2d, layers.conv2d_transpose],
                       activation_fn=tf.nn.relu):
            # encode inputs (map to parameterization of diagonal Gaussian)
            self.encoded = self._encode(self.x)

        # extract mean and (diagonal) log variance of latent variable
        self.mean = self.encoded[:, :self._latent_dim]
        self.logvar = self.encoded[:, self._latent_dim:]
        # also calculate standard deviation for practical use
        self.stddev = tf.sqrt(tf.exp(self.logvar))

        # calculate KL divergence between approximate posterior
        # q(z|x) ~ N(mu, sigma^T * I)
        # and prior
        # p(z) ~ N(0, I)
        kl_div = self._kl_divergence(self.mean, self.stddev)

        # use the reparameterization trick to sample from latent space
        epsilon = tf.random_normal([self._batch_size, self._latent_dim])
        self.z = self.mean + self.stddev * epsilon

        with arg_scope([layers.fully_connected,
                        layers.conv2d, layers.conv2d_transpose],
                       activation_fn=tf.nn.relu):
            # decode sample
            self.decoded = self._decode(self.z)

        # calculate reconstruction error between decoded sample
        # and original input batch
        rec_error = self._reconstruction_error(self.x, self.decoded)

        # set up training steps
        optimizer = tf.train.AdamOptimizer(learning_rate=3e-3)
        self._loss = kl_div + rec_error
        self._train = optimizer.minimize(self._loss)

        # start tensorflow session
        self._sesh = tf.Session()
        init = tf.global_variables_initializer()
        self._sesh.run(init)

    def _encode(self, x):
        """
        Inference network q(z|x) which encodes a mini batch of data points.

        :param x: (tf.Tensor(tf.float32) Mini batch of data points to encode.
        :return: encoded: (tf.Tensor(tf.float32)) Encoded mini batch.
        """
        if self._encoder_architecture == 'fc':
            encoded = layers.fully_connected(x, 500)
            encoded = layers.fully_connected(encoded, 500)
            encoded = layers.fully_connected(encoded, 200)
            encoded = layers.fully_connected(encoded, 2 * self._latent_dim,
                                             activation_fn=None)

        elif self._encoder_architecture == 'conv':
            encoded = tf.reshape(x, [-1, 28, 28, 1])
            encoded = layers.conv2d(encoded, 32, 5, stride=2)
            encoded = layers.conv2d(encoded, 64, 5, stride=2)
            encoded = layers.conv2d(encoded, 128, 3, stride=2)
            encoded = layers.flatten(encoded)
            encoded = layers.fully_connected(encoded, 500)
            encoded = layers.fully_connected(encoded, 200)
            encoded = layers.fully_connected(encoded, 2 * self._latent_dim,
                                             activation_fn=None)

        else:
            raise ValueError("Invalid encoder architecture specified!"
                             "Must be one of 'fc' or 'conv'.")

        return encoded

    def _decode(self, z):
        """
        Generative network p(x|z) which decodes a sample z from
        the latent space.

        :param z: (tf.Tensor(tf.float32)) Latent variable sampled from latent space.
        :return: decoded: (tf.Tensor(tf.float32)) Decoded latent variable.
        """
        if self._decoder_architecture == 'fc':
            decoded = layers.fully_connected(z, 200)
            decoded = layers.fully_connected(decoded, 500)
            decoded = layers.fully_connected(decoded, 500)
            decoded = layers.fully_connected(decoded, 28 * 28,
                                             activation_fn=tf.nn.sigmoid)

        elif self._decoder_architecture == 'conv':
            decoded = tf.expand_dims(z, 1)
            decoded = tf.expand_dims(decoded, 1)
            decoded = layers.conv2d_transpose(decoded, 128, 3, padding='VALID')
            decoded = layers.conv2d_transpose(decoded, 64, 5, padding='VALID')
            decoded = layers.conv2d_transpose(decoded, 32, 5, stride=2)
            decoded = layers.conv2d_transpose(decoded, 1, 5, stride=2,
                                              activation_fn=tf.nn.sigmoid)
            decoded = layers.flatten(decoded)

        else:
            raise ValueError("Invalid decoder architecture!"
                             "Must be one of 'fc' or 'conv'.")

        return decoded

    @staticmethod
    def _kl_divergence(mu, sigma, eps=1e-8):
        """
        Calculates KL Divergence between q~N(mu, sigma^T * I) and p~N(0, I).
        q(z|x) is the approximate posterior over the latent variable z,
        and p(z) is the prior on z.

        :param mu: (tf.Tensor(tf.float32)) Mean of z under approximate posterior.
        :param sigma: (tf.Tensor(tf.float32)) Standard deviation of z
            under approximate posterior.
        :param eps: (float) Small value to prevent log(0).
        :return: kl: (float) KL Divergence between q(z|x) and p(z).
        """
        var = tf.square(sigma)
        kl = 0.5 * tf.reduce_sum(tf.square(mu) + var - 1. - tf.log(var + eps))
        return kl

    @staticmethod
    def _reconstruction_error(targets, outputs, eps=1e-8):
        """
        Calculates negative log likelihood -log(p(x|z)) of outputs,
        assuming a Bernoulli distribution.

        :param targets: (tf.Tensor(tf.float32)) MNIST images.
        :param outputs: (tf.Tensor(tf.float32)) Probability distribution over outputs.
        :return: rec_error: (float) -log(p(x|z)) (negative log likelihood or
            'reconstruction error')
        """
        rec_error = -tf.reduce_sum(targets * tf.log(outputs + eps)
                                   + (1. - targets) * tf.log((1. - outputs) + eps))
        return rec_error

    def update(self, x):
        """
        Performs one mini batch update of parameters for both inference
        and generative networks.

        :param x: (tf.Tensor(tf.float32)) Mini batch of input data points.
        :return: loss: (float) Total loss (KL + reconstruction) for mini batch.
        """
        _, loss = self._sesh.run([self._train, self._loss],
                                 feed_dict={self.x: x})
        return loss

    def embed(self, x):
        """
        Maps a data point x to mean in latent space.

        :return: mean: (tf.Tensor(tf.float32)) mu such that q(z|x)~N(mu, .).
        """
        mean = self._sesh.run([self.mean], feed_dict={self.x: x})
        return mean

    def generate(self, z):
        """
        Maps a point in latent space to an image.

        :param z: (tf.Tensor(tf.float32)) Point in latent space.
        :return: x: (tf.Tensor(tf.float32)) Corresponding image generated from z.
        """
        x = self._sesh.run([self.decoded],
                           feed_dict={self.z: z})
        # need to reshape since our network processes batches of 1-D 28 * 28 arrays
        x = np.array(x)[:, 0, :].reshape(28, 28)
        return x
