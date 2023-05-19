import numpy
import torch
from diffusers import AutoencoderKL


def run(input_vectors, torch_device, model, local_cache_only=True):
    vae = AutoencoderKL.from_pretrained(
        model, subfolder="vae", local_files_only=local_cache_only
    )
    vae.to(torch_device)
    input_vectors_flipped = torch.from_numpy(numpy.flip(input_vectors, 2).copy())
    input_vectors_flipped = (input_vectors_flipped) * 2.0 - 1.0
    input_vectors_flipped = input_vectors_flipped.to(torch_device)

    with torch.no_grad():
        latent = vae.encode(input_vectors_flipped.unsqueeze(0))

    latents = vae.config.scaling_factor * latent.latent_dist.sample()
    return latents[0]
