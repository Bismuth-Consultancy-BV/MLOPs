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
import torch
import numpy
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation


def run(model, mode, input_colors, cache_only):
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
    elif mode == "segment":
        image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small", local_files_only=cache_only)
        image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small", local_files_only=cache_only)

        def ade_palette():
            return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                    [102, 255, 0], [92, 0, 255]]


        pixel_values = image_processor(img, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = image_segmentor(pixel_values)
        seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]
        color_seg = numpy.zeros((seg.shape[0], seg.shape[1], 3), dtype=numpy.uint8)
        palette = numpy.array(ade_palette())
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        processed_image = Image.fromarray(color_seg.astype(numpy.uint8))
    else:
        processed_image = img

    return processed_image
