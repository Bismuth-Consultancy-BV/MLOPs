import numpy
from diffusers import AutoencoderKL
import torch


def run(input_latents, latent_dimension, torch_device, model="CompVis/stable-diffusion-v1-4", local_cache_only=True):
    vae = AutoencoderKL.from_pretrained(model, subfolder="vae", local_files_only=local_cache_only)
    vae.to(torch_device)

    latents = torch.from_numpy(numpy.array([input_latents.reshape(4, latent_dimension[0], latent_dimension[1])])).to(torch_device)
    latents = (1.0 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    return image.cpu().numpy()[0]
