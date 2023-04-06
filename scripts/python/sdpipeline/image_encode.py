from diffusers import AutoencoderKL
import torch
import numpy

def run(input_vectors, torch_device, model="CompVis/stable-diffusion-v1-4", local_cache_only=True):
    # scale and decode the image latents with vae
    # 1. Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained(model, subfolder="vae", local_files_only=local_cache_only)
    vae.to(torch_device)
    input_vectors =  (torch.from_numpy(input_vectors) * 2.0) - 1.0
    input_vectors = input_vectors.to(torch_device)


    with torch.no_grad():
        latent = vae.encode(input_vectors.unsqueeze(0)) # Note scaling

    latents = 0.18215 * latent.latent_dist.sample()
    return latents[0]
