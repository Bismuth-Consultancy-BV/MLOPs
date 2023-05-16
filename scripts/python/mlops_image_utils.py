import numpy
from PIL import Image

from torchmetrics import StructuralSimilarityIndexMeasure
import numpy as np
import torchvision.transforms.functional as G

def colors_numpy_array_to_pil(input_colors):
    # Transpose into (width, height, channels)
    input_colors = input_colors.transpose(1, 2, 0)
    # Correct Orientation
    input_colors = input_colors[:, ::-1, :]
    # Gamma Correct
    input_colors = pow(input_colors, 1.0/2.2)
    # Convert to RGB space
    input_colors = (input_colors * 255).round().astype("uint8") 
    return Image.fromarray(input_colors)

def pil_to_colors_numpy_array(pil_image):
    # Convert to Numpy Array
    input_colors = numpy.asarray(pil_image)
    # Flip Horizontally
    input_colors = input_colors[:, ::-1, :]
    # Convert to 0-1 space
    input_colors = input_colors.astype(numpy.uint8) / 255
    # Reshape to (number_pixels, channels)
    input_colors = input_colors.reshape(-1, input_colors.shape[2])
    return input_colors

def SSIM(pil_image1, pil_image2, data_range=1.0, kernel_size=11, sigma=1.5, k1=0.01, k2=0.03):
    # make sure images are same size
    # Assuming you have two image files
    image1 = pil_image1
    image2 = pil_image2

    # Get the sizes of the images
    size1 = image1.size
    size2 = image2.size

    # Check if the sizes are different
    if size1 != size2:
        # Determine the maximum size among the two images
        max_width = max(size1[0], size2[0])
        max_height = max(size1[1], size2[1])

        # Resize the images to the maximum size
        image1 = image1.resize((max_width, max_height))
        image2 = image2.resize((max_width, max_height))

    # Now both images have the same size

    image1_array = image1
    image2_array = image2

    # Convert the NumPy arrays to tensors
    image1_tensor = G.to_tensor(image1_array).unsqueeze(0)  # Add extra dimension for batch size
    image2_tensor = G.to_tensor(image2_array).unsqueeze(0)  # Add extra dimension for batch size

    # Create an instance of the SSIM metric
    ssim_metric = StructuralSimilarityIndexMeasure(data_range, kernel_size, sigma, K=(k1, k2))

    # Compute SSIM
    ssim_value_tensor = ssim_metric(image1_tensor, image2_tensor)
    ssim_value = ssim_value_tensor.item()

    # remap cosine similarity (-1. to +1.) into 0 to 1 range
    ssim_value = (ssim_value + 1.) * .5

    # Print the similarity measure
    return ssim_value

def embedding_similarity(pil_image1, pil_image2):
    # make sure images are same size
    # Assuming you have two image files
    image1 = pil_image1
    image2 = pil_image2

    # Get the sizes of the images
    size1 = image1.size
    size2 = image2.size

    # Check if the sizes are different
    if size1 != size2:
        # Determine the maximum size among the two images
        max_width = max(size1[0], size2[0])
        max_height = max(size1[1], size2[1])

        # Resize the images to the maximum size
        image1 = image1.resize((max_width, max_height))
        image2 = image2.resize((max_width, max_height))

    # Now both images have the same size