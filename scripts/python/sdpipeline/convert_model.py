import logging

logging.disable(logging.WARNING)
import torch
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
)


def convert(checkpoint_file, config_file, export_path):
    from_safetensor = checkpoint_file.endswith(".safetensors")
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict=checkpoint_file,
        original_config_file=config_file,
        load_safety_checker=False,
        from_safetensors=from_safetensor,
    )

    pipe.save_pretrained(export_path)
