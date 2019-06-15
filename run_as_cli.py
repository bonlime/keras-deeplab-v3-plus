from generate_labels import generate_labels

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import click


@click.command()
@click.argument('image_path')
def run_as_cli(image_path):
    img_array = np.array(Image.open(image_path))

    result = generate_labels(img_array)

    plt.imshow(result)
    plt.waitforbuttonpress()


if __name__ == "__main__":
    run_as_cli()

