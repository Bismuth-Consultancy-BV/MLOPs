import numpy
from PIL import Image


def run(input_image, image_width, image_height, path):
    pixel_array = numpy.transpose(
        input_image.reshape(3, image_width, image_height), (1, 2, 0)
    )
    pixel_array = (
        (numpy.clip(pixel_array / 2 + 0.5, 0, 1) * 255).round().astype("uint8")
    )
    pil_image = Image.fromarray(pixel_array)
    # pil_image.show()
    pil_image.save(path)
