import numpy
from PIL import Image

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