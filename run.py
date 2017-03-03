import tensorflow as tf

from plot import plot_canvas, plot_spread
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
from vae import VAE


def main():
    flags = tf.flags

    # VAE params
    flags.DEFINE_integer("latent_dim", 2, "Dimension of latent space.")
    flags.DEFINE_integer("batch_size", 128, "Batch size.")
    flags.DEFINE_string("encoder_architecture", 'fc', "Architecture to use for encoder.")
    flags.DEFINE_string("decoder_architecture", 'fc', "Architecture to use for decoder.")

    # training params
    flags.DEFINE_integer("epochs", 200,
                         "Total number of epochs for which to train the model.")
    flags.DEFINE_integer("updates_per_epoch", 100,
                         "Number of (mini batch) updates performed per epoch.")

    # data params
    flags.DEFINE_string("data_dir", '../MNIST', "Directory containing MNIST data.")

    FLAGS = flags.FLAGS

    # define model
    kwargs = {
        'latent_dim': FLAGS.latent_dim,
        'batch_size': FLAGS.batch_size,
        'encoder_architecture': FLAGS.encoder_architecture,
        'decoder_architecture': FLAGS.decoder_architecture
    }
    vae = VAE(**kwargs)

    # read data
    mnist = input_data.read_data_sets(train_dir=FLAGS.data_dir, one_hot=True)

    # set up progress bar
    tbar = tqdm(range(FLAGS.epochs))
    for epoch in tbar:
        training_loss = 0.

        for _ in range(FLAGS.updates_per_epoch):
            x, _ = mnist.train.next_batch(FLAGS.batch_size)
            loss = vae.update(x)
            training_loss += loss

        # loss over most recent epoch
        training_loss /= (FLAGS.batch_size * FLAGS.updates_per_epoch)
        # update progress bar
        s = "Loss: {:.4f}".format(training_loss)
        tbar.set_description(s)

        if FLAGS.latent_dim == 2:
            plot_canvas(vae=vae, batch_size=FLAGS.batch_size, epoch=epoch)
            plot_spread()

if __name__ == '__main__':
    main()
