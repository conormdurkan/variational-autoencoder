import matplotlib.pyplot as plt
import numpy as np


def plot_canvas(vae, batch_size, epoch, n=15, bound=2, make_gif=False):
    nx = ny = n
    spaced_x = np.linspace(-bound, bound, nx)
    spaced_y = np.linspace(-bound, bound, ny)
    canvas = np.empty((28 * ny, 28 * nx))

    for i, xi in enumerate(spaced_x):
        for j, yi in enumerate(spaced_y):
            img = vae.generate(np.array([[xi, yi]] * batch_size))
            canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = img

    plt.figure(figsize=(10, 10))
    plt.imshow(canvas)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('latent.pdf')
    if make_gif:
        plt.savefig('./latent/latent-' + str(epoch + 1000) + '.jpeg')


def plot_spread():
    return
