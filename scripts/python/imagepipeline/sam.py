import mlops_image_utils
import torch
import numpy
from transformers import pipeline

def run(model, input_colors, torch_device, local_cache_only):
    if torch_device == "cuda":
        torch.cuda.empty_cache()
        torch_device += ":0"

    image = mlops_image_utils.colors_numpy_array_to_pil(input_colors).convert("RGB")

    generator = pipeline("mask-generation", model=model, device=torch_device)
    
    with torch.no_grad():
        outputs = generator(image, points_per_batch=64)

    if torch_device.startswith("cuda"):
        torch.cuda.empty_cache()

    masks = outputs["masks"]
    h,w = masks[0].shape[-2:]

    out = numpy.zeros((h,w,1))

    for i, mask in enumerate(masks):
        value = numpy.full((h,w, 1), i, dtype=numpy.uint8)
        mask = mask.reshape(h, w, 1)
        out += value * mask

    return out