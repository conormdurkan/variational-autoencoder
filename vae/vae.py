import numpy as np
import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope


class VAE:

    def __init__(self, latent_dim, batch_size, encoder, decoder):
        """
        Implementation of Variational Autoencoder (VAE) for  MNIST.
        Paper (Kingma & Welling): https://arxiv.org/abs/1312.6114.

        :param latent_dim: Dimension of latent space.
        :param batch_size: Number of data points per mini batch.
        :param encoder: function which encodes a batch of inputs to a 
            parameterization of a diagonal Gaussian
        :param decoder: function which decodes a batch of samples from 
            the latent space and returns the corresponding batch of images.
        """
        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._encode = encoder
        self._decode = decoder
        self._build_graph()

    def _build_graph(self):
        """
        Build tensorflow computational graph for VAE.
        x -> encode(x) -> latent parameterization & KL divergence ->
        z -> decode(z) -> distribution over x -> log likelihood ->
        total loss -> train step
        """
        with tf.variable_scope('vae'):
            # placeholder for MNIST inputs
            self.x = tf.placeholder(tf.float32, shape=[None, 28 * 28])

            # encode inputs (map to parameterization of diagonal Gaussian)
            with tf.variable_scope('encoder'):
                # use relu activations
                with arg_scope([layers.fully_connected,
                                layers.conv2d, layers.conv2d_transpose],
                               activation_fn=tf.nn.relu):
                    self.encoded = self._encode(self.x, self._latent_dim)

            with tf.variable_scope('sampling'):
                # extract mean and (diagonal) log variance of latent variable
                self.mean = self.encoded[:, :self._latent_dim]
                self.logvar = self.encoded[:, self._latent_dim:]
                # also calculate standard deviation for practical use
                self.stddev = tf.sqrt(tf.exp(self.logvar))

                # sample from latent space
                epsilon = tf.random_normal([self._batch_size, self._latent_dim])
                self.z = self.mean + self.stddev * epsilon

            # decode batch
            with tf.variable_scope('decoder'):
                # use relu activations
                with arg_scope([layers.fully_connected,
                                layers.conv2d, layers.conv2d_transpose],
                               activation_fn=tf.nn.relu):
                    self.decoded = self._decode(self.z)

            with tf.variable_scope('loss'):
                # calculate KL divergence between approximate posterior q and prior p
                with tf.variable_scope('kl-divergence'):
                    kl = self._kl_diagnormal_stdnormal(self.mean, self.stddev)

                # calculate reconstruction error between decoded sample
                # and original input batch
                with tf.variable_scope('log-likelihood'):
                    log_like = self._bernoulli_log_likelihood(self.x, self.decoded)

                self._loss = (kl + log_like) / self._batch_size

            with tf.variable_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=2e-4)
            with tf.variable_scope('training-step'):
                self._train = optimizer.minimize(self._loss)

            # start tensorflow session
            self._sesh = tf.Session()
            init = tf.global_variables_initializer()
            self._sesh.run(init)

    @staticmethod
    def _kl_diagnormal_stdnormal(mu, sigma, eps=1e-8):
        """
        Calculates KL Divergence between q~N(mu, sigma^T * I) and p~N(0, I).
        q(z|x) is the approximate posterior over the latent variable z,
        and p(z) is the prior on z.

        :param mu: Mean of z under approximate posterior.
        :param sigma: Standard deviation of z
            under approximate posterior.
        :param eps: Small value to prevent log(0).
        :return: kl: KL Divergence between q(z|x) and p(z).
        """
        var = tf.square(sigma)
        kl = 0.5 * tf.reduce_sum(tf.square(mu) + var - 1. - tf.log(var + eps))
        return kl

    @staticmethod
    def _bernoulli_log_likelihood(targets, outputs, eps=1e-8):
        """
        Calculates negative log likelihood -log(p(x|z)) of outputs,
        assuming a Bernoulli distribution.

        :param targets: MNIST images.
        :param outputs: Probability distribution over outputs.
        :return: log_like: -log(p(x|z)) (negative log likelihood)
        """
        log_like = -tf.reduce_sum(targets * tf.log(outputs + eps)
                                  + (1. - targets) * tf.log((1. - outputs) + eps))
        return log_like

    def update(self, x):
        """
        Performs one mini batch update of parameters for both inference
        and generative networks.

        :param x: Mini batch of input data points.
        :return: loss: Total loss (KL + NLL) for mini batch.
        """
        _, loss = self._sesh.run([self._train, self._loss],
                                 feed_dict={self.x: x})
        return loss

    def x2z(self, x):
        """
        Maps a data point x (i.e. an image) to mean in latent space.

        :return: mean: mu such that q(z|x) ~ N(mu, .).
        """
        mean = self._sesh.run([self.mean], feed_dict={self.x: x})
        return np.asarray(mean).reshape(-1, self._latent_dim)

    def z2x(self, z):
        """
        Maps a point in latent space to a 28*28 image.

        :param z: Point in latent space.
        :return: x: Corresponding image generated from z.
        """
        x = self._sesh.run([self.decoded],
                           feed_dict={self.z: z})
        # need to reshape since our network processes batches of 1-D 28 * 28 arrays
        x = np.array(x)[:, 0, :].reshape(28, 28)
        return x
