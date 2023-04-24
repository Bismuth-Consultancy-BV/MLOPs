import logging
logging.disable(logging.WARNING)
import torch
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

def convert(checkpoint_file, config_file, export_path):
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path=checkpoint_file,
        original_config_file=config_file,
        load_safety_checker=False
    )

    pipe.save_pretrained(export_path)
