from diffusers import AutoencoderKL
import torch


def run(input_vectors, mask, torch_device, model, local_cache_only=True):
    vae = AutoencoderKL.from_pretrained(model, subfolder="vae", local_files_only=local_cache_only)
    vae.to(torch_device)
    input_vectors =  (torch.from_numpy(input_vectors) * 2.0) - 1.0
    input_vectors = input_vectors * (1.0-mask)

    input_vectors = input_vectors.to(torch_device)


    with torch.no_grad():
        latent = vae.encode(input_vectors.unsqueeze(0))

    latents = vae.config.scaling_factor * latent.latent_dist.sample()
    return latents[0]