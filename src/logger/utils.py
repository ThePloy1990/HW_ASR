import io

import matplotlib.pyplot as plt
import PIL
from torchvision.transforms import ToTensor

plt.switch_backend("agg")


def plot_images(imgs, config):
    """
    Combine several images into one figure.

    Args:
        imgs (Tensor): array of images (B X C x H x W).
        config (DictConfig): hydra experiment config.
    Returns:
        image (Tensor): a single figure with imgs plotted side-to-side.
    """
    names = config.writer.names
    figsize = config.writer.figsize
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        img = imgs[i].permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(names[i])
        axes[i].axis("off")
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = ToTensor()(PIL.Image.open(buf))

    plt.close()

    return image


def plot_spectrogram(spectrogram, name=None):
    """
    Plot spectrogram

    Args:
        spectrogram (Tensor): spectrogram tensor.
        name (None | str): optional name.
    Returns:
        image (Image): image of the spectrogram
    """
    plt.figure(figsize=(20, 5))
    plt.pcolormesh(spectrogram)
    plt.title(name)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # convert buffer to Tensor
    image = ToTensor()(PIL.Image.open(buf))

    plt.close()

    return image
