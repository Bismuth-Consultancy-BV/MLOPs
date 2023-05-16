import mlops_image_utils
import hou
import os
from clip_interrogator import Config, Interrogator


def run(model, mode, input_colors, torch_device):
    cache_dir = os.path.join(hou.text.expandString("$MLOPS_MODELS"), "CLIP")
    config = Config(clip_model_name=model, device=torch_device, quiet=True, cache_path=cache_dir)
    clip_interrogator = Interrogator(config)

    image = mlops_image_utils.colors_numpy_array_to_pil(input_colors)
    if mode == 'best':
        prompt = clip_interrogator.interrogate(image)
    elif mode == 'classic':
        prompt = clip_interrogator.interrogate_classic(image)
    elif mode == 'fast':
        prompt = clip_interrogator.interrogate_fast(image)
    elif mode == 'negative':
        prompt = clip_interrogator.interrogate_negative(image)
    else:
        prompt = ""

    return prompt
