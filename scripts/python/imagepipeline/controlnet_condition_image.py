# import hou
# import os
# import sys
# sys.path.append(os.path.normpath(hou.text.expandString("$MLOPS/data/dependencies/python/controlnet_aux/")))

import mlops_image_utils
from controlnet_aux import (
    CannyDetector,
    ContentShuffleDetector,
    HEDdetector,
    LineartAnimeDetector,
    LineartDetector,
    MediapipeFaceDetector,
    MidasDetector,
    MLSDdetector,
    NormalBaeDetector,
    OpenposeDetector,
    PidiNetDetector,
    ZoeDetector,
)


def run(model, mode, input_colors):
    img = mlops_image_utils.colors_numpy_array_to_pil(input_colors)

    if mode == "hed":
        hed = HEDdetector.from_pretrained(model)
        processed_image = hed(img)
    elif mode == "midas":
        midas = MidasDetector.from_pretrained(model)
        processed_image = midas(img)
    elif mode == "mlsd":
        mlsd = MLSDdetector.from_pretrained(model)
        processed_image = mlsd(img)
    elif mode == "openpose":
        open_pose = OpenposeDetector.from_pretrained(model)
        processed_image = open_pose(img, hand_and_face=True)
    elif mode == "pidi":
        pidi = PidiNetDetector.from_pretrained(model)
        processed_image = pidi(img, safe=True)
    elif mode == "bae":
        normal_bae = NormalBaeDetector.from_pretrained(model)
        processed_image = normal_bae(img)
    elif mode == "lineart":
        lineart = LineartDetector.from_pretrained(model)
        processed_image = lineart(img, coarse=True)
    elif mode == "lineartanime":
        lineart_anime = LineartAnimeDetector.from_pretrained(model)
        processed_image = lineart_anime(img)
    elif mode == "zoe":
        zoe = ZoeDetector.from_pretrained(model)
        processed_image = zoe(img)
    elif mode == "canny":
        canny = CannyDetector()
        processed_image = canny(img)
    elif mode == "content":
        content = ContentShuffleDetector()
        processed_image = content(img)
    elif mode == "facedetector":
        face_detector = MediapipeFaceDetector()
        processed_image = face_detector(img)
    else:
        processed_image = img

    return processed_image
