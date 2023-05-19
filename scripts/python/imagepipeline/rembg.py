import os

import hou
import mlops_image_utils
import numpy

os.environ["U2NET_HOME"] = hou.text.expandString("$MLOPS_MODELS/rembg/")
import rembg


def run(
    input_colors,
    model_name,
    post_process,
    alphamatting,
    erodesize,
    foregroundthreshold,
    backgroundthreshold,
):
    image = mlops_image_utils.colors_numpy_array_to_pil(input_colors).convert("RGB")
    session = rembg.new_session(model_name)

    output = rembg.remove(
        image,
        session=session,
        post_process_mask=post_process,
        alpha_matting=alphamatting,
        alpha_matting_foreground_threshold=foregroundthreshold,
        alpha_matting_background_threshold=backgroundthreshold,
        alpha_matting_erode_size=erodesize,
    )

    return mlops_image_utils.pil_to_colors_numpy_array(output)
