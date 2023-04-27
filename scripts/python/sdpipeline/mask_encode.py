import numpy
from skimage.transform import resize


def run(input_mask, latent_width, latent_height, invert):

    mask = input_mask
    mask[mask < 0.5] = 0.0
    mask[mask >= 0.5] = 1.0

    resampled_data = resize(mask, (latent_width, latent_height))
    mask = numpy.flip(numpy.tile(resampled_data,(4,1,1)),2)

    if invert:
        mask = 1.0-mask

    return mask
