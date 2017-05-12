# MNIST VAE using Tensorflow
Tensorflow Implementation of the Variational Autoencoder using the MNIST data set, first introduced in [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).

The code is heavily documented since the implementation was used as a learning process.

Both fully connected and convolutional encoders and decoders are available in `networks.py`. The VAE architecture is (fairly) modular, so implementing your own encoder/decoder pairs is easy. Changing to another data set needs a little more work, since the data provider, likelihood function, and plotting utilities all need tweaking.

Defaults for the model are as follows:

| Parameter        | Name | Default Value |
| :------------- | :------------- | :-----|
| Latent dimension      | `latent_dim` | 2 |
| Batch size      | `batch_size` |   128 |
| Encoder architecture | `encoder_architecture`      |    'fc' |
| Decoder architecture | `decoder_architecture`      |    'fc' |
| Epochs | `epochs`      |    100 |
| Updates per epoch| `updates_per_epoch`      |    100 |
| Data directory | `data_dir`      |  '../mnist'   |
| Perform visualisation | `do_viz`|    True |

Architectures can be `'fc'` or `'conv'` for each of the encoder and decoder.

The MNIST data set will be automatically downloaded to `data_dir` if the data is not found in this directory. Change to your own location if you don't want multiple copies of MNIST hanging around.

If you'd prefer not to generate `2 * epochs` images and corresponding gifs, then run `python --do_viz=False run.py`.

## Visualisation
Plotting functionality is included for visualising a 2-D latent space. This runs by default whenever the latent dimension is 2.

The first gif shows how 5000 test images are embedded in the latent space over 100 epochs of training.
![](spread.gif)

The second gif shows the result of decoding a grid of points in the latent space at each epoch of training (this example rendered in log time).
![aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa](canvas.gif)

Lower loss is achievable using a slightly bigger learning rate, but the visualisations suffer a bit since the convergence is too fast.

## References
The implementation was based on the two (immensely helpful) resources:

* Ilya Kostrikov's [vae-gan-draw](https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW) repo.
* Fast Forward Labs' blog series, [Introducing Variational Autoencoders](http://blog.fastforwardlabs.com/2016/08/12/introducing-variational-autoencoders-in-prose-and.html) and [Under the Hood of the Variational Autoencoder](http://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html).
