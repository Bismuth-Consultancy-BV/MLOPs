import numpy
from scipy import ndimage

def run(input_mask, latent_width, latent_height, invert):

    mask = input_mask
    mask[mask < 0.5] = 0.0
    mask[mask >= 0.5] = 1.0

    zoom_factors = [new_dim / old_dim for new_dim, old_dim in zip((latent_width, latent_height), mask.shape)]
    resampled_data = ndimage.zoom(mask, zoom_factors, order=3)  # order=1 for linear interpolation
    mask = numpy.tile(resampled_data,(4,1,1))

    if invert:
        mask = 1.0-mask

    return mask