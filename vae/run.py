import networks as nets
import tensorflow as tf

from plot import make_canvas, make_spread, make_canvas_gif, make_spread_gif
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
from vae import VAE


def main():
    flags = tf.flags

    # VAE params
    flags.DEFINE_integer("latent_dim", 2, "Dimension of latent space.")
    flags.DEFINE_integer("batch_size", 128, "Batch size.")
    # architectures
    flags.DEFINE_string("encoder_architecture", 'fc', "Architecture to use for encoder.")
    flags.DEFINE_string("decoder_architecture", 'fc', "Architecture to use for decoder.")

    # training params
    flags.DEFINE_integer("epochs", 100,
                         "Total number of epochs for which to train the model.")
    flags.DEFINE_integer("updates_per_epoch", 100,
                         "Number of (mini batch) updates performed per epoch.")

    # data params
    flags.DEFINE_string("data_dir", '../mnist', "Directory containing MNIST data.")
    FLAGS = flags.FLAGS

    # viz params
    flags.DEFINE_bool("do_viz", True, "Whether to make visualisations for 2D.")

    architectures = {
        'encoders': {
            'fc': nets.fc_mnist_encoder,
            'conv': nets.conv_mnist_encoder
        },
        'decoders': {
            'fc': nets.fc_mnist_decoder,
            'conv': nets.conv_mnist_decoder
        }
    }

    # define model
    kwargs = {
        'latent_dim': FLAGS.latent_dim,
        'batch_size': FLAGS.batch_size,
        'encoder': architectures['encoders'][FLAGS.encoder_architecture],
        'decoder': architectures['decoders'][FLAGS.decoder_architecture]
    }
    vae = VAE(**kwargs)

    # data provider
    provider = input_data.read_data_sets(train_dir=FLAGS.data_dir)

    # do training
    tbar = tqdm(range(FLAGS.epochs))
    for epoch in tbar:
        training_loss = 0.

        # iterate through batches
        for _ in range(FLAGS.updates_per_epoch):
            x, _ = provider.train.next_batch(FLAGS.batch_size)
            loss = vae.update(x)
            training_loss += loss

        # loss over most recent epoch
        training_loss /= (FLAGS.batch_size * FLAGS.updates_per_epoch)
        # update progress bar
        s = "Loss: {:.4f}".format(training_loss)
        tbar.set_description(s)

        # make pretty pictures if latent dim. is 2-dimensional
        if FLAGS.latent_dim == 2 and FLAGS.do_viz:
            make_canvas(vae=vae, batch_size=FLAGS.batch_size, epoch=epoch)
            make_spread(vae, provider, epoch)

    # make
    if FLAGS.latent_dim == 2 and FLAGS.do_viz:
        make_canvas_gif()
        make_spread_gif()


if __name__ == '__main__':
    main()
