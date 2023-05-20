import numpy
from PIL import Image


def colors_numpy_array_to_pil(input_colors):
    # Transpose into (width, height, channels)
    input_colors = input_colors.transpose(1, 2, 0)
    # Gamma Correct
    input_colors = pow(input_colors, 1.0 / 2.2)
    # Convert to RGB space
    input_colors = (input_colors * 255).round().astype("uint8")
    return Image.fromarray(input_colors)


def pil_to_colors_numpy_array(pil_image):
    # Convert to Numpy Array
    input_colors = numpy.asarray(pil_image)
    # Convert to 0-1 space
    input_colors = input_colors.astype(numpy.uint8) / 255
    # Transpose into (channels, width, height)
    input_colors = input_colors.transpose(2, 0, 1)
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
    width = geo.attribValue("image_dimension")[0]
    height = geo.attribValue("image_dimension")[1]

    r = numpy.array(geo.pointFloatAttribValues("r"), dtype=numpy.float32).reshape(
        height, width
    )
    g = numpy.array(geo.pointFloatAttribValues("g"), dtype=numpy.float32).reshape(
        height, width
    )
    b = numpy.array(geo.pointFloatAttribValues("b"), dtype=numpy.float32).reshape(
        height, width
    )

    input_colors = numpy.stack((r, g, b), axis=0)
    return input_colors


def pil_to_colored_points(geo, pil_image):
    # Convert the PIL image to a numpy array
    cd_array = numpy.array(pil_image)

    # Split the color data into separate "r", "g", and "b" arrays
    r_attrib = cd_array[0:, :].ravel() / 255.0
    g_attrib = cd_array[1:, :].ravel() / 255.0
    b_attrib = cd_array[2:, :].ravel() / 255.0

    # Set the "r", "g", and "b" attributes on the points
    geo.setPointFloatAttribValues("r", r_attrib)
    geo.setPointFloatAttribValues("g", g_attrib)
    geo.setPointFloatAttribValues("b", b_attrib)


def numpy_array_to_colored_points(geo, cd_array, scale_factor=255.0):
    # Split the color data into separate "r", "g", and "b" arrays
    r_attrib = cd_array[0, :, :].ravel() / scale_factor
    g_attrib = cd_array[1, :, :].ravel() / scale_factor
    b_attrib = cd_array[2, :, :].ravel() / scale_factor

    # Set the "r", "g", and "b" attributes on the points
    geo.setPointFloatAttribValues("r", list(map(float, r_attrib)))
    geo.setPointFloatAttribValues("g", list(map(float, g_attrib)))
    geo.setPointFloatAttribValues("b", list(map(float, b_attrib)))
