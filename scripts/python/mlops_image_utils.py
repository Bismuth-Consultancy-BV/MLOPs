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

def ensure_same_pil_image_dimensions(pil_image1, pil_image2):
        # Get the sizes of the images
    size1 = pil_image1.size
    size2 = pil_image2.size

    # Check if the sizes are different
    if size1 != size2:
        # Determine the maximum size among the two images
        max_width = max(size1[0], size2[0])
        max_height = max(size1[1], size2[1])

        # Resize the images to the maximum size
        pil_image1 = pil_image1.resize((max_width, max_height))
        pil_image2 = pil_image2.resize((max_width, max_height))

    return pil_image1, pil_image2


def colored_points_to_numpy_array(geo):
    width = int(geo.attribValue("image_dimension")[0])
    height = int(geo.attribValue("image_dimension")[1])

    r = numpy.array(geo.pointFloatAttribValues("r"), dtype=numpy.float32).reshape(width, height)
    g = numpy.array(geo.pointFloatAttribValues("g"), dtype=numpy.float32).reshape(width, height)
    b = numpy.array(geo.pointFloatAttribValues("b"), dtype=numpy.float32).reshape(width, height)
    
    input_colors = numpy.stack((r,g,b), axis=0)
    return input_colors
